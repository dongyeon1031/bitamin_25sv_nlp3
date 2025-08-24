# -*- coding: utf-8 -*-
import json, pickle
from typing import List, Dict, Tuple
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from rag.embeddings import LocalEmbedder
from configs.rag_config import (
    DOCSTORE_PATH, BM25_INDEX_PATH, CHROMA_DIR,
    TOP_K_VECTOR, TOP_K_BM25, MERGE_TOP_K, ALPHA_VECTOR
)

def _load_docstore() -> Dict[str, Dict]:
    ds = {}
    with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ds[obj["id"]] = obj
    return ds

def _tokenize_ko(text: str):
    import re
    text = text.lower()
    return re.findall(r"[가-힣]+|[a-z]+|\d+", text)

class HybridRetriever:
    def __init__(self):
        self.docstore = _load_docstore()

        with open(BM25_INDEX_PATH, "rb") as f:
            self.bm25 = pickle.load(f)["bm25"]

        self.embedder = LocalEmbedder()
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)

        # trick to give chroma our embedder
        outer = self
        class _EmbedFunc(embedding_functions.EmbeddingFunction):
            def __call__(self, texts):
                return outer.embedder.encode(texts).tolist()

        self.collection = self.client.get_collection("law_chunks", embedding_function=_EmbedFunc())

    def vector_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        res = self.collection.query(query_texts=[query], n_results=top_k)
        ids = res["ids"][0]
        dists = res.get("distances", [[]])[0]
        sims = [1.0 - float(d) for d in dists]  # cosine distance → similarity
        return list(zip(ids, sims))

    def bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        toks = _tokenize_ko(query)
        scores = self.bm25.get_scores(toks)
        idxs = np.argsort(scores)[::-1][:top_k]
        id_list = list(self.docstore.keys())
        return [(id_list[i], float(scores[i])) for i in idxs]

    @staticmethod
    def _minmax(xs: List[float]) -> List[float]:
        if not xs:
            return []
        mn, mx = min(xs), max(xs)
        if mx - mn < 1e-8:
            return [0.0 for _ in xs]
        return [(x - mn) / (mx - mn + 1e-8) for x in xs]

    def retrieve(self, query: str, merge_top_k: int = MERGE_TOP_K) -> List[Dict]:
        v_hits = self.vector_search(query, TOP_K_VECTOR)
        b_hits = self.bm25_search(query, TOP_K_BM25)

        vdict = {i: s for i, s in v_hits}
        bdict = {i: s for i, s in b_hits}
        all_ids = list(set(vdict) | set(bdict))

        v_norm = self._minmax([vdict.get(i, 0.0) for i in all_ids])
        b_norm = self._minmax([bdict.get(i, 0.0) for i in all_ids])

        merged = []
        for idx, cid in enumerate(all_ids):
            score = ALPHA_VECTOR * v_norm[idx] + (1 - ALPHA_VECTOR) * b_norm[idx]
            merged.append((cid, score))

        merged.sort(key=lambda x: x[1], reverse=True)
        merged = merged[:merge_top_k]

        out = []
        for cid, s in merged:
            d = self.docstore[cid]
            out.append({
                "id": cid,
                "score": float(s),
                "text": d["text"],
                "parent_text": d.get("parent_text", d["text"]),
                "meta": d.get("meta", {})
            })
        return out
