from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel

class RouteDecision(BaseModel):
    model_name: str
    provider: str
    confidence: float
    reason: str
    estimated_cost_per_1k_tokens: float
    metadata: Dict[str, Any] = {}

class BaseRouter(ABC):
    """ Abstract base class for all routing strategies."""

    @abstractmethod
    def route(self, prompt: str, **kwargs) -> RouteDecision:
        """ Evaluates input prompt and returns the optimal model route."""
        pass
