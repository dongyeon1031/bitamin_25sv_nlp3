import os, json, pickle
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from rag.embeddings import LocalEmbedder
from finetune.security_configs.security_rag_config import CHROMA_DIR, BM25_INDEX_PATH, DOCSTORE_PATH, CHROMA_COLLECTION

def _tokenize_ko(text: str):
    """한국어 토크나이저"""
    import re
    text = text.lower()
    return re.findall(r"[가-힣]+|[a-z]+|\d+", text)

def build_and_persist_security_index(chunks: List[Dict]):
    """보안/경제 데이터용 인덱스 구축 및 저장"""
    os.makedirs(os.path.dirname(DOCSTORE_PATH), exist_ok=True)

    # 1) Docstore
    with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "id": c["id"],
                "text": c["text"],
                "parent_hash": c.get("parent_hash"),
                "meta": c["meta"]
            }, ensure_ascii=False) + "\n")

    # 2) BM25
    tokenized = [_tokenize_ko(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "ids": [c["id"] for c in chunks],
            "docs": [c["text"] for c in chunks],
            "metas": [c["meta"] for c in chunks],
        }, f)

    # 3) Chroma (cosine metric 보장)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception as e:
        print(f"[WARN] 기존 컬렉션 삭제 실패: {e}")

    embedder = LocalEmbedder()

    class _EmbedFunc(embedding_functions.EmbeddingFunction):
        def __call__(self, texts):
            return embedder.encode(texts).tolist()

    col = client.create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=_EmbedFunc(),
        metadata={"hnsw:space": "cosine"}   
    )

    ids = [c["id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [c["meta"] for c in chunks]

    col.add(ids=ids, documents=docs, metadatas=metas)
    print(f"[RAG] 보안 데이터 인덱스 구축 완료: {len(chunks)}개 청크")
