# rag/embeddings.py
from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from configs.rag_config import EMBEDDING_MODEL_PATH

class LocalEmbedder:
    def __init__(self, model_path: str = EMBEDDING_MODEL_PATH, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_path, device=self.device)

    def encode(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
            vecs = vecs / norms
        return vecs
