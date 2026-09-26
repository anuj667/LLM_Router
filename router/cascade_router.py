from typing import Callable, Optional
from router.base import BaseRouter, RouteDecision

class CascadeRouter(BaseRouter):
    """
    Executes a speculative cascade: calls a cheap model first, checks output validity,
    and falls back to an expensive model only if validation fails.
    """

    def __init__(self, evaluator_fn: Callable[[str], bool]):
        self.evaluator_fn = evaluator_fn

    def route(self, prompt: str, **kwargs) -> RouteDecision:
        # Default entry is always the cheap tier in a cascade pattern
        return RouteDecision(
            model_name="gpt-4o-mini",
            provider="openai",
            confidence=1.0,
            reason="Cascade initial tier: Attempting cheap execution first.",
            estimated_cost_per_1k_tokens=0.00015
        )

    def evaluate_or_escalate(self, candidate_response: str) -> Optional[RouteDecision]:
        """Call this after getting the response from the lower-tier model."""
        if self.evaluator_fn(candidate_response):
            return None  # Output accepted; no escalation required.

        return RouteDecision(
            model_name="gpt-4o",
            provider="openai",
            confidence=1.0,
            reason="Cascade escalation: Low-tier output failed validation criteria.",
            estimated_cost_per_1k_tokens=0.005
        )
