"""
DeepConf-Lite: Heuristic confidence for AIMO (no logprobs needed)

Simplified version of DeepConf for vLLM streaming API without logprobs.
Uses response characteristics to estimate confidence instead of token logprobs.
"""

import re
import math
from typing import List, Tuple, Dict, Any
from collections import Counter, defaultdict

from vllm.logger import init_logger

logger = init_logger(__name__)


class HeuristicConfidence:
    """
    Estimate trace confidence from response characteristics.
    
    No logprobs needed - works with streaming API!
    
    Confidence factors:
    1. Response length (longer = more thorough)
    2. Verification keywords (check, verify, confirm)
    3. Python usage (executable reasoning)
    4. Mathematical notation (proper formulas)
    5. Multiple approaches (robust reasoning)
    6. Clear conclusion (organized thinking)
    """
    
    def __init__(self):
        self.verification_keywords = [
            "check", "verify", "confirm", "let me verify",
            "double check", "let's check", "testing",
            "validation", "確認", "検証"  # multilingual support
        ]
        
        self.reasoning_keywords = [
            "because", "since", "therefore", "thus",
            "it follows", "we can see", "this means",
            "consequently", "as a result"
        ]
        
        self.uncertainty_keywords = [
            "wait", "hmm", "actually", "let me reconsider",
            "on second thought", "I think", "maybe",
            "probably", "might be", "could be"
        ]
    
    def compute_confidence(self, response: str, answer: int, question: str = "") -> float:
        """
        Compute heuristic confidence score [0, 1].
        
        Args:
            response: Full reasoning trace text
            answer: Extracted final answer
            question: Problem text
        
        Returns:
            Confidence score (higher = more confident)
        """
        if not response:
            return 0.0
        
        score = 0.5  # Base confidence
        
        # 1. Length-based confidence (longer responses with detailed reasoning)
        response_length = len(response)
        if response_length > 3000:
            score += 0.15  # Very detailed
        elif response_length > 1500:
            score += 0.10  # Detailed
        elif response_length > 500:
            score += 0.05  # Moderate
        elif response_length < 200:
            score -= 0.10  # Too short, likely rushed
        
        # 2. Verification keywords (self-checking)
        verification_count = sum(1 for kw in self.verification_keywords if kw.lower() in response.lower())
        score += min(0.15, verification_count * 0.03)
        
        # 3. Reasoning quality keywords
        reasoning_count = sum(1 for kw in self.reasoning_keywords if kw.lower() in response.lower())
        score += min(0.10, reasoning_count * 0.02)
        
        # 4. Python code usage (executable reasoning)
        python_code_blocks = response.count("```python") + response.count("```py")
        if python_code_blocks > 0:
            score += 0.15
            if python_code_blocks > 2:
                score += 0.05  # Multiple code verifications
        
        # 5. Mathematical notation density
        math_symbols = ["\\frac", "\\sum", "\\int", "\\sqrt", "^", "_", "\\times"]
        math_count = sum(response.count(sym) for sym in math_symbols)
        if math_count > 10:
            score += 0.10
        elif math_count > 5:
            score += 0.05
        
        # 6. Multiple approaches (considering alternatives)
        approach_indicators = [
            "approach", "method", "alternatively", "another way",
            "let's try", "different approach", "or we can"
        ]
        approach_count = sum(1 for ind in approach_indicators if ind.lower() in response.lower())
        if approach_count >= 2:
            score += 0.10
        
        # 7. Clear conclusion structure
        conclusion_indicators = ["therefore", "thus", "final answer", "the answer is", "boxed"]
        has_conclusion = any(ind.lower() in response.lower() for ind in conclusion_indicators)
        if has_conclusion:
            score += 0.05
        
        # 8. Uncertainty penalties (doubt signals)
        uncertainty_count = sum(1 for kw in self.uncertainty_keywords if kw.lower() in response.lower())
        if uncertainty_count > 5:
            score -= 0.20  # High uncertainty
        elif uncertainty_count > 3:
            score -= 0.10  # Moderate uncertainty
        elif uncertainty_count > 1:
            score -= 0.05  # Some uncertainty
        
        # 9. Repeated calculations (thorough verification)
        answer_str = str(answer)
        answer_mentions = response.count(answer_str)
        if answer_mentions > 3:
            score += 0.10  # Answer derived/verified multiple times
        elif answer_mentions > 1:
            score += 0.05
        
        # 10. Step-by-step structure (organized thinking)
        step_patterns = [r"Step \d+", r"\d+\.", r"\(\d+\)"]
        step_count = sum(len(re.findall(pattern, response)) for pattern in step_patterns)
        if step_count > 10:
            score += 0.10
        elif step_count > 5:
            score += 0.05
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    def compute_group_confidence(self, response: str, group_size: int = 2048) -> float:
        """
        Simulate group confidence by splitting response into chunks.
        Returns minimum chunk confidence (mimics bottom group confidence).
        """
        if len(response) <= group_size:
            return self.compute_confidence(response, 0, "")
        
        # Split into chunks
        chunks = []
        for i in range(0, len(response), group_size):
            chunk = response[i:i+group_size]
            chunks.append(chunk)
        
        # Compute confidence for each chunk
        chunk_confs = []
        for chunk in chunks:
            conf = self.compute_confidence(chunk, 0, "")
            chunk_confs.append(conf)
        
        # Return minimum (weakest segment)
        return min(chunk_confs) if chunk_confs else 0.0
    
    def compute_tail_confidence(self, response: str, tail_size: int = 2048) -> float:
        """
        Compute confidence over final tail_size characters.
        """
        tail = response[-tail_size:] if len(response) > tail_size else response
        return self.compute_confidence(tail, 0, "")


class DeepConfLiteVoting:
    """
    DeepConf voting using heuristic confidence (no logprobs).
    """
    
    def __init__(self, top_k_percent: float = 0.90):
        self.top_k_percent = top_k_percent
        self.heuristic_conf = HeuristicConfidence()
    
    def filter_by_confidence(
        self, 
        traces: List[Any], 
        confidences: List[float],
        top_k_percent: float = None
    ) -> Tuple[List[Any], List[float]]:
        """Filter traces, keeping top-k% by confidence."""
        if top_k_percent is None:
            top_k_percent = self.top_k_percent
        
        if not traces:
            return [], []
        
        # Sort by confidence
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
        """Confidence-weighted majority voting."""
        if not answers:
            return 0, 0.0, {"error": "No answers"}
        
        # Weighted votes
        vote_weights = defaultdict(float)
        vote_counts = Counter()
        
        for ans, conf in zip(answers, confidences):
            vote_weights[ans] += conf
            vote_counts[ans] += 1
        
        # Winner
        best_answer = max(vote_weights.items(), key=lambda x: x[1])
        final_answer = best_answer[0]
        total_weight = best_answer[1]
        
        # Consensus
        total_all_weights = sum(vote_weights.values())
        consensus = total_weight / total_all_weights if total_all_weights > 0 else 0.0
        
        return final_answer, consensus, {
            "vote_weights": dict(vote_weights),
            "vote_counts": dict(vote_counts),
            "consensus": consensus,
            "num_traces": len(answers),
        }
    
    def deepconf_vote(
        self,
        responses: List[str],
        answers: List[int],
        question: str = "",
        confidence_type: str = "full"  # "full", "tail", or "group"
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        DeepConf voting with heuristic confidence.
        
        Args:
            responses: Full text responses
            answers: Extracted final answers
            question: Problem text
            confidence_type: "full" (whole response), "tail" (last 2k chars), 
                            "group" (min chunk confidence)
        
        Returns:
            (final_answer, consensus, debug_info)
        """
        if not answers or not responses:
            return 0, 0.0, {"error": "No data"}
        
        # Compute confidence for each response
        trace_confidences = []
        for resp, ans in zip(responses, answers):
            if confidence_type == "tail":
                conf = self.heuristic_conf.compute_tail_confidence(resp)
            elif confidence_type == "group":
                conf = self.heuristic_conf.compute_group_confidence(resp)
            else:  # "full"
                conf = self.heuristic_conf.compute_confidence(resp, ans, question)
            
            trace_confidences.append(conf)
        
        # Filter by confidence
        filtered_answers, filtered_confs = self.filter_by_confidence(
            answers,
            trace_confidences,
            top_k_percent=self.top_k_percent
        )
        
        # Weighted voting
        final_answer, consensus, debug_info = self.confidence_weighted_voting(
            filtered_answers,
            filtered_confs
        )
        
        # Add metrics
        debug_info.update({
            "confidence_type": confidence_type,
            "top_k_percent": self.top_k_percent,
            "original_traces": len(answers),
            "filtered_traces": len(filtered_answers),
            "conf_mean": sum(trace_confidences) / len(trace_confidences),
            "conf_min": min(trace_confidences),
            "conf_max": max(trace_confidences),
        })
        
        return final_answer, consensus, debug_info


# Global instances
_deepconf_lite_voting = DeepConfLiteVoting(top_k_percent=0.90)


def enable_deepconf_lite(top_k_percent: float = 0.90):
    """Enable DeepConf-Lite with custom settings."""
    global _deepconf_lite_voting
    _deepconf_lite_voting = DeepConfLiteVoting(top_k_percent=top_k_percent)
    logger.info(f"✅ DeepConf-Lite enabled: top_k={top_k_percent*100}%")


def get_deepconf_lite_voting() -> DeepConfLiteVoting:
    """Get global DeepConf-Lite voting instance."""
    return _deepconf_lite_voting


__all__ = [
    "HeuristicConfidence",
    "DeepConfLiteVoting",
    "enable_deepconf_lite",
    "get_deepconf_lite_voting",
]
