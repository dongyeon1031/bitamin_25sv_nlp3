import json, pickle
from typing import List, Dict, Tuple
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from rag.embeddings import LocalEmbedder
from configs.unified_rag_config import (
    UNIFIED_DOCSTORE_PATH, UNIFIED_BM25_INDEX_PATH, UNIFIED_CHROMA_DIR, UNIFIED_CHROMA_COLLECTION,
    TOP_K_VECTOR, TOP_K_BM25, MERGE_TOP_K
)

def _load_unified_docstore() -> Tuple[Dict[str, Dict], List[str]]:
    """통합 docstore 로드"""
    ds = {}
    ids = []
    with open(UNIFIED_DOCSTORE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ds[obj["id"]] = obj
            ids.append(obj["id"])
    return ds, ids

def _tokenize_ko(text: str):
    """한국어 토크나이저"""
    import re
    text = text.lower()
    return re.findall(r"[가-힣]+|[a-z]+|\d+", text)

class UnifiedHybridRetriever:
    """통합 하이브리드 리트리버 (법령 + 보안/경제)"""
    
    def __init__(self):
        self.docstore, self.id_list = _load_unified_docstore()

        with open(UNIFIED_BM25_INDEX_PATH, "rb") as f:
            bm25_obj = pickle.load(f)
            self.bm25 = bm25_obj["bm25"] if isinstance(bm25_obj, dict) else bm25_obj

        self.embedder = LocalEmbedder()
        self.client = chromadb.PersistentClient(path=UNIFIED_CHROMA_DIR)

        outer = self
        class _EmbedFunc(embedding_functions.EmbeddingFunction):
            def __call__(self, texts):
                return outer.embedder.encode(texts).tolist()

        try:
            self.collection = self.client.get_collection(
                UNIFIED_CHROMA_COLLECTION,
                embedding_function=_EmbedFunc()
            )
        except Exception:
            # 없으면 새로 생성하면서 metric 지정
            self.collection = self.client.create_collection(
                name=UNIFIED_CHROMA_COLLECTION,
                embedding_function=_EmbedFunc(),
                metadata={"hnsw:space": "cosine"}
            )

    def vector_search(self, query: str, top_k: int) -> List[str]:
        """벡터 검색"""
        res = self.collection.query(query_texts=[query], n_results=top_k)
        ids = res["ids"][0]
        return [(cid, rank+1) for rank, cid in enumerate(ids)]

    def bm25_search(self, query: str, top_k: int) -> List[str]:
        """BM25 검색"""
        toks = _tokenize_ko(query)
        scores = self.bm25.get_scores(toks)
        idxs = np.argsort(scores)[::-1][:top_k]
        return [(self.id_list[i], rank+1) for rank, i in enumerate(idxs)]

    def retrieve(self, query: str, merge_top_k: int = MERGE_TOP_K, k: int = 60) -> List[Dict]:
        """하이브리드 검색 수행"""
        v_hits = self.vector_search(query, TOP_K_VECTOR)
        b_hits = self.bm25_search(query, TOP_K_BM25)

        rank_dicts = {"vector": dict(v_hits), "bm25": dict(b_hits)}
        all_ids = set(rank_dicts["vector"].keys()) | set(rank_dicts["bm25"].keys())

        merged = []
        for cid in all_ids:
            score = 0.0
            for rdict in rank_dicts.values():
                if cid in rdict:
                    score += 1.0 / (k + rdict[cid])
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
                "meta": {**d.get("meta", {}), "parent_id": d.get("parent_hash")}
            })
        return out

    def filter_by_doc_type(self, results: List[Dict], doc_type: str) -> List[Dict]:
        """문서 타입별 필터링"""
        return [r for r in results if r["meta"].get("doc_type") == doc_type]

    def get_statistics(self) -> Dict:
        """인덱스 통계 정보"""
        law_count = len([c for c in self.docstore.values() if c["meta"].get("doc_type") == "law"])
        security_count = len([c for c in self.docstore.values() if c["meta"].get("doc_type") == "security"])
        
        return {
            "total_chunks": len(self.docstore),
            "law_chunks": law_count,
            "security_chunks": security_count
        }
