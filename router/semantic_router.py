import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from router.base import BaseRouter, RouteDecision

class SemanticRouter(BaseRouter):
    """
    Routes queries based on semantic embedding similarity against pre-defined intent clusters.
    """

    def __init__(self, routes: Dict[str, Dict[str, Any]], embedding_model: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embedding_model)
        self.routes = routes
        self._build_index()

    def _build_index(self):
        self.route_names = []
        self.embeddings = []
        for name, data in self.routes.items():
            for sample in data["samples"]:
                self.route_names.append(name)
                self.embeddings.append(self.encoder.encode(sample))
        self.embeddings = np.array(self.embeddings)

    def route(self, prompt: str, threshold: float = 0.3, **kwargs) -> RouteDecision:
        query_vec = self.encoder.encode(prompt)
        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
        sims = np.dot(self.embeddings, query_vec) / np.maximum(norms, 1e-9)

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        matched_route = self.route_names[best_idx]

        if best_score < threshold:
            return RouteDecision(
                model_name="claude-3-5-sonnet",
                provider="anthropic",
                confidence=best_score,
                reason="Fallback: No semantic cluster surpassed confidence threshold.",
                estimated_cost_per_1k_tokens=0.003
            )

        route_info = self.routes[matched_route]
        return RouteDecision(
            model_name=route_info["target_model"],
            provider=route_info["provider"],
            confidence=best_score,
            reason=f"Matched cluster: '{matched_route}'",
            estimated_cost_per_1k_tokens=route_info["cost_per_1k"],
            metadata={"cluster": matched_route}
        )
