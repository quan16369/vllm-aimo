"""
AIMO Custom Scheduler Policies

Optimized scheduling for Tool-Integrated Reasoning (TIR) workload:
1. Low-latency scheduling for interactive tool execution
2. Priority boosting for sequences with tool calls  
3. Efficient batch management for K parallel samples
"""

from typing import List, Optional, Tuple
from enum import Enum
import time

from vllm.logger import init_logger
from vllm.sequence import SequenceGroup, SequenceStatus

logger = init_logger(__name__)


class SequenceType(Enum):
    """Types of sequences in TIR workload."""
    PREFILL = "prefill"  # Initial prompt processing
    DECODE = "decode"    # Token generation
    TOOL_PENDING = "tool_pending"  # Waiting for tool execution
    TOOL_RESULT = "tool_result"    # Processing tool results


class AIMOSchedulerPolicy:
    """
    Custom scheduler optimized for AIMO TIR workload.
    
    Key characteristics of TIR:
    - K samples run in parallel (K=4 typically)
    - Each sample may call Python tools multiple times
    - Tool execution happens on CPU (not GPU), creating gaps
    - Latency is critical (5-minute budget per problem)
    
    Optimization strategies:
    1. Prioritize decode over prefill for low latency
    2. Boost priority for sequences with pending tool results
    3. Batch samples from same problem for prefix cache benefits
    4. Preempt long-running sequences to reduce tail latency
    """
    
    def __init__(
        self,
        max_num_seqs: int = 32,
        max_num_batched_tokens: int = 8192,
        enable_priority_boost: bool = True,
    ):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_priority_boost = enable_priority_boost
        
        # Track sequence priorities
        self.seq_priorities: dict[str, float] = {}
        
        # Statistics
        self.total_scheduled = 0
        self.tool_boosts = 0
        self.preemptions = 0
        
        logger.info(
            "[AIMO Scheduler] Initialized: max_seqs=%d, max_tokens=%d, "
            "priority_boost=%s",
            max_num_seqs, max_num_batched_tokens, enable_priority_boost
        )
    
    def compute_priority(
        self, 
        seq_group: SequenceGroup,
        current_time: float
    ) -> float:
        """
        Compute scheduling priority for a sequence group.
        
        Lower priority value = scheduled earlier.
        
        Priority factors:
        1. Base: arrival time (FCFS)
        2. Boost: -1000 if has tool result pending  
        3. Boost: -500 if in decode phase (vs prefill)
        4. Penalty: +100 per minute of age (prevent starvation)
        """
        seq_id = seq_group.request_id
        
        # Base priority: arrival time
        arrival_time = seq_group.arrival_time
        priority = arrival_time
        
        # Check if in decode phase
        is_decoding = any(
            seq.get_len() > seq.get_prompt_len()
            for seq in seq_group.get_seqs()
        )
        
        if is_decoding and self.enable_priority_boost:
            # Boost decode sequences for lower latency
            priority -= 500
        
        # Check for tool result (simulated by checking metadata)
        # In real implementation, this would check if sequence has
        # a pending tool execution result
        if hasattr(seq_group, 'sampling_params'):
            metadata = getattr(seq_group.sampling_params, 'metadata', {})
            if metadata.get('has_tool_result', False):
                priority -= 1000
                self.tool_boosts += 1
                logger.debug(
                    "[AIMO] Tool result boost: seq=%s, new_priority=%.1f",
                    seq_id[:8], priority
                )
        
        # Age penalty (prevent starvation)
        age_seconds = current_time - arrival_time
        age_penalty = (age_seconds // 60) * 100  # +100 per minute
        priority += age_penalty
        
        # Cache priority
        self.seq_priorities[seq_id] = priority
        
        return priority
    
    def schedule_batch(
        self,
        waiting: List[SequenceGroup],
        running: List[SequenceGroup],
        current_time: float,
    ) -> Tuple[List[SequenceGroup], List[SequenceGroup]]:
        """
        Schedule next batch of sequences to run.
        
        Returns:
            (sequences_to_run, preempted_sequences)
        """
        # Compute priorities for all sequences
        all_seqs = waiting + running
        seq_priorities = [
            (self.compute_priority(seq, current_time), seq)
            for seq in all_seqs
        ]
        
        # Sort by priority (lower = higher priority)
        seq_priorities.sort(key=lambda x: x[0])
        
        # Select sequences to run
        selected = []
        total_tokens = 0
        preempted = []
        
        for priority, seq_group in seq_priorities:
            # Estimate tokens needed
            if any(seq.get_len() == seq.get_prompt_len() 
                   for seq in seq_group.get_seqs()):
                # Prefill: use prompt length
                num_tokens = max(
                    seq.get_prompt_len() 
                    for seq in seq_group.get_seqs()
                )
            else:
                # Decode: estimate 1 token per sequence
                num_tokens = len(seq_group.get_seqs())
            
            # Check if we can add this sequence
            if (len(selected) < self.max_num_seqs and 
                total_tokens + num_tokens <= self.max_num_batched_tokens):
                selected.append(seq_group)
                total_tokens += num_tokens
            else:
                # Sequence cannot fit - preempt if running
                if seq_group in running:
                    preempted.append(seq_group)
                    self.preemptions += 1
        
        self.total_scheduled += len(selected)
        
        logger.debug(
            "[AIMO Scheduler] Scheduled %d seqs, %d tokens, %d preempted",
            len(selected), total_tokens, len(preempted)
        )
        
        return selected, preempted
    
    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        return {
            "total_scheduled": self.total_scheduled,
            "tool_boosts": self.tool_boosts,
            "preemptions": self.preemptions,
            "cached_priorities": len(self.seq_priorities),
        }


class TIRBatchOptimizer:
    """
    Optimize batching for TIR workload where K samples process same question.
    
    Benefits:
    - Group samples from same problem for prefix cache benefits
    - Synchronize prefill phases to maximize cache hits
    - Balance load across samples
    """
    
    def __init__(self, K: int = 4):
        self.K = K  # Number of parallel samples
        self.problem_groups: dict[str, List[SequenceGroup]] = {}
        
        logger.info("[AIMO] TIRBatchOptimizer initialized with K=%d", K)
    
    def group_by_problem(
        self, 
        sequences: List[SequenceGroup]
    ) -> dict[str, List[SequenceGroup]]:
        """
        Group sequences by problem ID.
        
        Sequences from same problem should be batched together
        to maximize prefix cache benefits.
        """
        groups: dict[str, List[SequenceGroup]] = {}
        
        for seq_group in sequences:
            # Extract problem ID from metadata
            problem_id = "default"
            if hasattr(seq_group, 'sampling_params'):
                metadata = getattr(seq_group.sampling_params, 'metadata', {})
                problem_id = metadata.get('problem_id', 'default')
            
            if problem_id not in groups:
                groups[problem_id] = []
            groups[problem_id].append(seq_group)
        
        return groups
    
    def optimize_batch(
        self,
        sequences: List[SequenceGroup],
        max_batch_size: int = 32
    ) -> List[List[SequenceGroup]]:
        """
        Create optimized batches that group same-problem sequences.
        
        Returns:
            List of batches, each batch is a list of sequences
        """
        # Group by problem
        problem_groups = self.group_by_problem(sequences)
        
        batches = []
        current_batch = []
        
        # Pack complete problem groups into batches
        for problem_id, seqs in problem_groups.items():
            # If problem group fits in current batch, add it
            if len(current_batch) + len(seqs) <= max_batch_size:
                current_batch.extend(seqs)
            else:
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = seqs[:max_batch_size]
        
        # Add remaining batch
        if current_batch:
            batches.append(current_batch)
        
        logger.debug(
            "[AIMO] Created %d optimized batches from %d problems",
            len(batches), len(problem_groups)
        )
        
        return batches


def create_aimo_scheduler(
    max_num_seqs: int = 32,
    max_num_batched_tokens: int = 8192,
    K: int = 4
) -> Tuple[AIMOSchedulerPolicy, TIRBatchOptimizer]:
    """
    Create AIMO-optimized scheduler components.
    
    Args:
        max_num_seqs: Maximum concurrent sequences
        max_num_batched_tokens: Maximum tokens per batch
        K: Number of parallel samples
    
    Returns:
        (scheduler_policy, batch_optimizer)
    """
    policy = AIMOSchedulerPolicy(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_priority_boost=True
    )
    
    optimizer = TIRBatchOptimizer(K=K)
    
    logger.info("[AIMO] Scheduler components created successfully")
    
    return policy, optimizer
