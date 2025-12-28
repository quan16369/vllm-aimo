"""
AIMO Math-Optimized Sampling Parameters

Enhanced sampling strategies for mathematical reasoning:
1. Dynamic temperature based on reasoning stage
2. Boosted precision for numerical outputs
3. Repetition penalty tuned for math expressions
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class MathSamplingPreset:
    """
    Predefined sampling presets optimized for mathematical reasoning.
    """
    name: str
    temperature: float
    top_p: float
    min_p: float
    repetition_penalty: float
    description: str


# Preset configurations for different stages of math reasoning
MATH_SAMPLING_PRESETS = {
    "explore": MathSamplingPreset(
        name="explore",
        temperature=1.0,
        top_p=0.95,
        min_p=0.02,
        repetition_penalty=1.0,
        description="Exploratory sampling for initial problem analysis"
    ),
    
    "reason": MathSamplingPreset(
        name="reason",
        temperature=0.9,
        top_p=0.92,
        min_p=0.03,
        repetition_penalty=1.02,
        description="Balanced sampling for step-by-step reasoning"
    ),
    
    "compute": MathSamplingPreset(
        name="compute",
        temperature=0.7,
        top_p=0.90,
        min_p=0.05,
        repetition_penalty=1.05,
        description="Focused sampling for numerical computations"
    ),
    
    "verify": MathSamplingPreset(
        name="verify",
        temperature=0.5,
        top_p=0.85,
        min_p=0.08,
        repetition_penalty=1.1,
        description="Conservative sampling for answer verification"
    ),
    
    "tool_call": MathSamplingPreset(
        name="tool_call",
        temperature=0.3,
        top_p=0.80,
        min_p=0.10,
        repetition_penalty=1.0,
        description="Precise sampling for tool invocation syntax"
    ),
}


class DynamicMathSampler:
    """
    Dynamically adjust sampling parameters based on generation context.
    
    Optimized for AIMO TIR (Tool-Integrated Reasoning) workload:
    - Early iterations: more exploratory (high temperature)
    - Middle iterations: balanced reasoning
    - Late iterations: focused on answer (low temperature)
    - Tool calls: very precise (lowest temperature)
    """
    
    def __init__(
        self,
        max_iterations: int = 100,
        enable_dynamic: bool = True,
    ):
        self.max_iterations = max_iterations
        self.enable_dynamic = enable_dynamic
        self.current_iteration = 0
        
        logger.info(
            "[AIMO Sampler] DynamicMathSampler initialized: "
            "max_iter=%d, dynamic=%s",
            max_iterations, enable_dynamic
        )
    
    def get_params_for_iteration(
        self, 
        iteration: int,
        context_type: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get sampling parameters for current iteration.
        
        Args:
            iteration: Current iteration number (0-based)
            context_type: Optional context hint 
                         ("explore", "reason", "compute", "verify", "tool_call")
        
        Returns:
            Dictionary of sampling parameters
        """
        if not self.enable_dynamic:
            # Return default balanced params
            preset = MATH_SAMPLING_PRESETS["reason"]
            return self._preset_to_dict(preset)
        
        # If explicit context provided, use that preset
        if context_type and context_type in MATH_SAMPLING_PRESETS:
            preset = MATH_SAMPLING_PRESETS[context_type]
            logger.debug(
                "[AIMO] Using %s preset: temp=%.2f, top_p=%.2f",
                context_type, preset.temperature, preset.top_p
            )
            return self._preset_to_dict(preset)
        
        # Otherwise, dynamically select based on iteration
        progress = iteration / self.max_iterations
        
        if progress < 0.3:
            # Early: explore
            preset = MATH_SAMPLING_PRESETS["explore"]
        elif progress < 0.6:
            # Middle: reason
            preset = MATH_SAMPLING_PRESETS["reason"]
        elif progress < 0.85:
            # Late: compute
            preset = MATH_SAMPLING_PRESETS["compute"]
        else:
            # Final: verify
            preset = MATH_SAMPLING_PRESETS["verify"]
        
        logger.debug(
            "[AIMO] Iteration %d/%d (%.1f%%), using %s preset",
            iteration, self.max_iterations, progress * 100, preset.name
        )
        
        return self._preset_to_dict(preset)
    
    def get_params_for_tool_call(self) -> Dict[str, float]:
        """Get sampling parameters optimized for tool calls."""
        preset = MATH_SAMPLING_PRESETS["tool_call"]
        logger.debug("[AIMO] Using tool_call preset")
        return self._preset_to_dict(preset)
    
    def _preset_to_dict(self, preset: MathSamplingPreset) -> Dict[str, float]:
        """Convert preset to dictionary."""
        return {
            "temperature": preset.temperature,
            "top_p": preset.top_p,
            "min_p": preset.min_p,
            "repetition_penalty": preset.repetition_penalty,
        }
    
    def update_iteration(self, iteration: int) -> None:
        """Update current iteration counter."""
        self.current_iteration = iteration


class NumericalTokenBooster:
    """
    Boost logits for numerical tokens to improve accuracy in math.
    
    When generating numbers, we want higher precision and less randomness.
    This booster increases logit values for numerical tokens.
    """
    
    def __init__(
        self,
        boost_factor: float = 1.2,
        numerical_token_ids: Optional[set] = None
    ):
        self.boost_factor = boost_factor
        
        # Default numerical token IDs (adjust based on tokenizer)
        # These are approximate - should be calibrated per tokenizer
        self.numerical_token_ids = numerical_token_ids or set(range(15, 25))
        
        logger.info(
            "[AIMO] NumericalTokenBooster: boost=%.2f, num_tokens=%d",
            boost_factor, len(self.numerical_token_ids)
        )
    
    def boost_logits(
        self,
        logits: "torch.Tensor",  # type: ignore
        context_is_numerical: bool = False
    ) -> "torch.Tensor":  # type: ignore
        """
        Boost logits for numerical tokens.
        
        Args:
            logits: Logit tensor [vocab_size]
            context_is_numerical: True if previous tokens were numbers
        
        Returns:
            Modified logits
        """
        if not context_is_numerical:
            return logits
        
        # Boost numerical token logits
        for token_id in self.numerical_token_ids:
            if token_id < logits.shape[-1]:
                logits[..., token_id] *= self.boost_factor
        
        return logits


def create_aimo_sampler(
    max_iterations: int = 100,
    K: int = 4,
    enable_dynamic: bool = True,
) -> DynamicMathSampler:
    """
    Create AIMO-optimized sampler.
    
    Args:
        max_iterations: Maximum reasoning iterations
        K: Number of parallel samples
        enable_dynamic: Enable dynamic parameter adjustment
    
    Returns:
        DynamicMathSampler instance
    """
    sampler = DynamicMathSampler(
        max_iterations=max_iterations,
        enable_dynamic=enable_dynamic
    )
    
    logger.info(
        "[AIMO] Created math sampler: max_iter=%d, K=%d, dynamic=%s",
        max_iterations, K, enable_dynamic
    )
    
    return sampler


def get_default_math_params() -> Dict[str, Any]:
    """Get default sampling parameters for math reasoning."""
    return {
        "temperature": 1.0,
        "top_p": 0.95,
        "min_p": 0.02,
        "repetition_penalty": 1.0,
        "max_tokens": 2048,
    }


def print_available_presets() -> None:
    """Print all available sampling presets."""
    logger.info("[AIMO] Available sampling presets:")
    for name, preset in MATH_SAMPLING_PRESETS.items():
        logger.info(
            "  %s: temp=%.2f, top_p=%.2f, min_p=%.2f, rep_pen=%.2f - %s",
            preset.name,
            preset.temperature,
            preset.top_p,
            preset.min_p,
            preset.repetition_penalty,
            preset.description
        )


# Example usage patterns for AIMO notebook:
"""
# In your inference code:

from vllm.aimo_sampling import create_aimo_sampler, MATH_SAMPLING_PRESETS

# Create dynamic sampler
sampler = create_aimo_sampler(max_iterations=100, K=4)

# For each iteration:
for iteration in range(max_iterations):
    # Get dynamic parameters
    params = sampler.get_params_for_iteration(iteration)
    
    # Or explicitly specify context:
    # params = sampler.get_params_for_iteration(iteration, context_type="tool_call")
    
    # Use in vLLM call
    response = client.completions.create(
        model="gpt-oss",
        prompt=prompt,
        temperature=params["temperature"],
        top_p=params["top_p"],
        min_p=params["min_p"],
        repetition_penalty=params["repetition_penalty"],
        ...
    )
"""
