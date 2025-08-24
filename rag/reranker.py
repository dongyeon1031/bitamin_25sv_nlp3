from typing import List, Dict
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = None):
    
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 8) -> List[Dict]:
    
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        rescored = []
        for c, s in zip(candidates, scores):
            new_c = c.copy()
            new_c["rerank_score"] = float(s)
            rescored.append(new_c)

        rescored.sort(key=lambda x: x["rerank_score"], reverse=True)

        return rescored[:top_k]
