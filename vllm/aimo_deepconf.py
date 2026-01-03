"""
DeepConf: Deep Think with Confidence for AIMO Competition

Based on the paper "Deep Think with Confidence" (Meta AI, 2025)
https://arxiv.org/abs/2508.15260

Key improvements over standard majority voting:
1. Group Confidence: Local confidence over sliding windows (2048 tokens)
2. Bottom-10% Confidence: Focus on weakest reasoning segments  
3. Tail Confidence: Emphasize final reasoning steps
4. Confidence-weighted voting with adaptive filtering (10%/90%)

Results on AIME 2025:
- GPT-OSS-120B: 91.8% (pass@1) → 99.9% (DeepConf@512)
- DeepSeek-8B: 76.9% (pass@1) → 87.4% (DeepConf@512)
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from collections import Counter, defaultdict
from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class DeepConfConfig:
    """Configuration for DeepConf confidence measurements."""
    # Group confidence settings
    group_size: int = 2048  # Sliding window size for group confidence
    overlap: bool = True     # Use overlapping windows
    
    # Confidence measurement type
    use_bottom_10_percent: bool = True   # Use bottom 10% group confidence
    use_tail_confidence: bool = True      # Use tail confidence (last 2048 tokens)
    use_mean_confidence: bool = False     # Use average trace confidence (baseline)
    
    # Filtering settings
    top_k_percent: float = 0.90  # Keep top 90% (conservative) or 0.10 (aggressive)
    
    # Weighted voting
    confidence_weighted: bool = True  # Use confidence weighting in voting


class TokenConfidence:
    """
    Compute token-level confidence from logprobs.
    
    Token confidence C_i = -1/k * sum(log P_i(j)) for top-k tokens
    High confidence → peaked distribution → greater certainty
    """
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
    
    def compute_token_confidence(self, logprobs: Dict[str, float]) -> float:
        """
        Compute confidence for a single token position.
        
        Args:
            logprobs: Dict mapping tokens to their log probabilities
        
        Returns:
            Token confidence score (higher = more confident)
        """
        if not logprobs:
            return 0.0
        
        # Get top-k log probabilities
        sorted_logprobs = sorted(logprobs.values(), reverse=True)
        top_k_logprobs = sorted_logprobs[:min(self.top_k, len(sorted_logprobs))]
        
        # Average negative log probability (convert to positive confidence)
        token_conf = -sum(top_k_logprobs) / len(top_k_logprobs)
        return token_conf
    
    def compute_trace_confidence(self, all_logprobs: List[Dict[str, float]]) -> List[float]:
        """
        Compute confidence for all tokens in a trace.
        
        Args:
            all_logprobs: List of logprob dicts for each token position
        
        Returns:
            List of token confidence scores
        """
        return [self.compute_token_confidence(logprobs) for logprobs in all_logprobs]


class GroupConfidence:
    """
    Compute group confidence over sliding windows.
    
    Group confidence C_G = average token confidence over group window
    Provides localized, smoother signal than single-token confidence
    """
    
    def __init__(self, group_size: int = 2048, overlap: bool = True):
        self.group_size = group_size
        self.overlap = overlap
        self.token_conf = TokenConfidence()
    
    def compute_group_confidences(self, token_confidences: List[float]) -> List[float]:
        """
        Compute group confidence for each position using sliding window.
        
        Args:
            token_confidences: Token-level confidence scores
        
        Returns:
            Group confidence for each token position (looking back group_size tokens)
        """
        if not token_confidences:
            return []
        
        group_confs = []
        n = len(token_confidences)
        
        for i in range(n):
            # Get window of previous tokens (overlapping sliding window)
            start_idx = max(0, i - self.group_size + 1)
            end_idx = i + 1
            window = token_confidences[start_idx:end_idx]
            
            # Average confidence in this group
            group_conf = sum(window) / len(window) if window else 0.0
            group_confs.append(group_conf)
        
        return group_confs


class ConfidenceMeasurements:
    """
    Compute various trace-level confidence measurements.
    
    Implements three key metrics from DeepConf paper:
    1. Average Trace Confidence (baseline, global measure)
    2. Bottom-10% Group Confidence (captures worst reasoning segments)
    3. Tail Confidence (focuses on final reasoning steps)
    """
    
    def __init__(self, config: DeepConfConfig):
        self.config = config
        self.group_conf = GroupConfidence(
            group_size=config.group_size,
            overlap=config.overlap
        )
    
    def compute_mean_confidence(self, token_confidences: List[float]) -> float:
        """
        Average Trace Confidence (Eq. 3 in paper).
        
        C_avg = 1/N * sum(C_i) for all tokens
        Simple global measure, but can mask local failures
        """
        if not token_confidences:
            return 0.0
        return sum(token_confidences) / len(token_confidences)
    
    def compute_bottom_10_percent_confidence(self, group_confidences: List[float]) -> float:
        """
        Bottom-10% Group Confidence (Eq. 5 in paper).
        
        C_bottom-10 = mean of lowest 10% of group confidences
        Captures effect of extremely low confidence groups
        """
        if not group_confidences:
            return 0.0
        
        # Get bottom 10%
        k = max(1, int(len(group_confidences) * 0.10))
        sorted_confs = sorted(group_confidences)
        bottom_10_percent = sorted_confs[:k]
        
        return sum(bottom_10_percent) / len(bottom_10_percent)
    
    def compute_tail_confidence(self, token_confidences: List[float], tail_size: int = 2048) -> float:
        """
        Tail Confidence (Eq. 7 in paper).
        
        C_tail = mean confidence over last tail_size tokens
        Final reasoning steps are critical for correctness
        """
        if not token_confidences:
            return 0.0
        
        # Get last tail_size tokens
        tail_tokens = token_confidences[-tail_size:] if len(token_confidences) > tail_size else token_confidences
        
        return sum(tail_tokens) / len(tail_tokens)
    
    def compute_lowest_group_confidence(self, group_confidences: List[float]) -> float:
        """
        Lowest Group Confidence (Eq. 6 in paper).
        
        C_least = min(C_G) for all groups
        Special case of bottom-10%, focuses on single weakest segment
        """
        if not group_confidences:
            return 0.0
        return min(group_confidences)
    
    def compute_all_measurements(self, token_confidences: List[float]) -> Dict[str, float]:
        """
        Compute all confidence measurements for a trace.
        
        Returns dict with keys:
        - mean_conf: Average trace confidence
        - bottom_10_conf: Bottom 10% group confidence  
        - tail_conf: Tail confidence
        - lowest_group_conf: Minimum group confidence
        """
        # Compute group confidences first
        group_confs = self.group_conf.compute_group_confidences(token_confidences)
        
        return {
            "mean_conf": self.compute_mean_confidence(token_confidences),
            "bottom_10_conf": self.compute_bottom_10_percent_confidence(group_confs),
            "tail_conf": self.compute_tail_confidence(token_confidences, self.config.group_size),
            "lowest_group_conf": self.compute_lowest_group_confidence(group_confs),
        }


class DeepConfVoting:
    """
    Confidence-aware voting with adaptive filtering.
    
    Implements DeepConf offline thinking algorithms:
    1. Confidence-weighted majority voting
    2. Confidence filtering (top-k% retention)
    """
    
    def __init__(self, config: DeepConfConfig):
        self.config = config
        self.measurements = ConfidenceMeasurements(config)
    
    def filter_by_confidence(
        self, 
        traces: List[Any], 
        confidences: List[float],
        top_k_percent: float = 0.90
    ) -> Tuple[List[Any], List[float]]:
        """
        Filter traces by confidence, keeping top-k%.
        
        Args:
            traces: List of trace objects
            confidences: Confidence score for each trace
            top_k_percent: Keep top k% traces (0.10 for aggressive, 0.90 for conservative)
        
        Returns:
            (filtered_traces, filtered_confidences)
        """
        if not traces:
            return [], []
        
        # Sort by confidence (descending)
        paired = list(zip(confidences, traces))
        paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)
        
        # Keep top k%
        k = max(1, int(len(paired_sorted) * top_k_percent))
        top_k = paired_sorted[:k]
        
        filtered_confs, filtered_traces = zip(*top_k) if top_k else ([], [])
        return list(filtered_traces), list(filtered_confs)
    
    def confidence_weighted_voting(
        self,
        answers: List[int],
        confidences: List[float]
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        Confidence-weighted majority voting.
        
        V(a) = sum(C_t * I(answer(t) = a)) for all traces t
        Final answer = argmax V(a)
        
        Args:
            answers: List of final answers from each trace
            confidences: Confidence score for each trace
        
        Returns:
            (final_answer, winning_confidence, debug_info)
        """
        if not answers:
            return 0, 0.0, {"error": "No answers"}
        
        # Compute weighted votes for each answer
        vote_weights = defaultdict(float)
        vote_counts = Counter()
        
        for ans, conf in zip(answers, confidences):
            vote_weights[ans] += conf
            vote_counts[ans] += 1
        
        # Find answer with highest weighted vote
        best_answer = max(vote_weights.items(), key=lambda x: x[1])
        final_answer = best_answer[0]
        total_weight = best_answer[1]
        
        # Compute consensus ratio
        total_all_weights = sum(vote_weights.values())
        consensus = total_weight / total_all_weights if total_all_weights > 0 else 0.0
        
        debug_info = {
            "vote_weights": dict(vote_weights),
            "vote_counts": dict(vote_counts),
            "consensus": consensus,
            "num_traces": len(answers),
            "num_unique_answers": len(vote_weights),
        }
        
        return final_answer, consensus, debug_info
    
    def deepconf_vote(
        self,
        responses: List[str],
        answers: List[int],
        token_confidences_list: List[List[float]],
        question: str = "",
        confidence_type: str = "bottom_10"
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        DeepConf offline voting with confidence filtering.
        
        Args:
            responses: List of full text responses
            answers: List of extracted final answers
            token_confidences_list: Token confidence scores for each response
            question: Problem text (for validation)
            confidence_type: "mean", "bottom_10", "tail", or "lowest_group"
        
        Returns:
            (final_answer, consensus, debug_info)
        """
        if not answers or not token_confidences_list:
            return 0, 0.0, {"error": "No data"}
        
        # Compute trace-level confidence for each response
        trace_confidences = []
        for token_confs in token_confidences_list:
            measurements = self.measurements.compute_all_measurements(token_confs)
            
            # Select confidence measure based on config
            if confidence_type == "mean":
                conf = measurements["mean_conf"]
            elif confidence_type == "bottom_10":
                conf = measurements["bottom_10_conf"]
            elif confidence_type == "tail":
                conf = measurements["tail_conf"]
            elif confidence_type == "lowest_group":
                conf = measurements["lowest_group_conf"]
            else:
                conf = measurements["bottom_10_conf"]  # Default
            
            trace_confidences.append(conf)
        
        # Apply confidence filtering
        filtered_answers, filtered_confs = self.filter_by_confidence(
            answers,
            trace_confidences,
            top_k_percent=self.config.top_k_percent
        )
        
        # Confidence-weighted voting on filtered traces
        final_answer, consensus, debug_info = self.confidence_weighted_voting(
            filtered_answers,
            filtered_confs
        )
        
        debug_info.update({
            "confidence_type": confidence_type,
            "top_k_percent": self.config.top_k_percent,
            "original_num_traces": len(answers),
            "filtered_num_traces": len(filtered_answers),
            "confidence_stats": {
                "mean": np.mean(trace_confidences) if trace_confidences else 0,
                "std": np.std(trace_confidences) if trace_confidences else 0,
                "min": min(trace_confidences) if trace_confidences else 0,
                "max": max(trace_confidences) if trace_confidences else 0,
            }
        })
        
        return final_answer, consensus, debug_info


# Global instance for easy import
_deepconf_config = DeepConfConfig(
    group_size=2048,
    use_bottom_10_percent=True,
    use_tail_confidence=True,
    top_k_percent=0.90,  # Conservative: keep top 90%
    confidence_weighted=True
)

_deepconf_voting = DeepConfVoting(_deepconf_config)


def enable_deepconf(
    group_size: int = 2048,
    top_k_percent: float = 0.90,
    confidence_type: str = "bottom_10"
):
    """
    Enable DeepConf with custom settings.
    
    Args:
        group_size: Window size for group confidence (default 2048)
        top_k_percent: Keep top k% traces (0.10=aggressive, 0.90=conservative)
        confidence_type: "mean", "bottom_10", "tail", or "lowest_group"
    """
    global _deepconf_config, _deepconf_voting
    
    _deepconf_config = DeepConfConfig(
        group_size=group_size,
        top_k_percent=top_k_percent,
        use_bottom_10_percent=(confidence_type == "bottom_10"),
        use_tail_confidence=(confidence_type == "tail"),
        use_mean_confidence=(confidence_type == "mean"),
    )
    
    _deepconf_voting = DeepConfVoting(_deepconf_config)
    
    logger.info(f"✅ DeepConf enabled: group_size={group_size}, "
                f"top_k={top_k_percent*100}%, type={confidence_type}")


def get_deepconf_voting() -> DeepConfVoting:
    """Get global DeepConf voting instance."""
    return _deepconf_voting


def compute_token_confidence_from_logprobs(logprobs_data: List[Dict]) -> List[float]:
    """
    Helper to extract token confidences from vLLM logprobs output.
    
    Args:
        logprobs_data: List of logprob dicts from vLLM response.logprobs
    
    Returns:
        List of token confidence scores
    """
    token_conf = TokenConfidence()
    confidences = []
    
    for logprob_entry in logprobs_data:
        # vLLM format: logprob_entry has 'token' and 'logprob' keys
        # or it might be a dict of {token: logprob}
        if isinstance(logprob_entry, dict):
            token_confs = token_conf.compute_token_confidence(logprob_entry)
            confidences.append(token_confs)
    
    return confidences


# Export key classes and functions
__all__ = [
    "DeepConfConfig",
    "TokenConfidence", 
    "GroupConfidence",
    "ConfidenceMeasurements",
    "DeepConfVoting",
    "enable_deepconf",
    "get_deepconf_voting",
    "compute_token_confidence_from_logprobs",
]
