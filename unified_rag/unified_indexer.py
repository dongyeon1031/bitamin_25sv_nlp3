import os, json, pickle
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from rag.embeddings import LocalEmbedder
from configs.unified_rag_config import (
    UNIFIED_CHROMA_DIR, UNIFIED_BM25_INDEX_PATH, UNIFIED_DOCSTORE_PATH, 
    UNIFIED_CHROMA_COLLECTION
)

def _tokenize_ko(text: str):
    """한국어 토크나이저"""
    import re
    text = text.lower()
    return re.findall(r"[가-힣]+|[a-z]+|\d+", text)

def build_and_persist_unified_index(chunks: List[Dict]):
    """통합 인덱스 구축 및 저장 (법령 + 보안/경제)"""
    os.makedirs(os.path.dirname(UNIFIED_DOCSTORE_PATH), exist_ok=True)

    print(f"[UNIFIED RAG] 통합 인덱스 구축 시작: {len(chunks)}개 청크")

    # 1) Docstore
    print("[UNIFIED RAG] Docstore 구축 중...")
    with open(UNIFIED_DOCSTORE_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "id": c["id"],
                "text": c["text"],
                "parent_hash": c.get("parent_hash", c["text"]),
                "meta": c["meta"]
            }, ensure_ascii=False) + "\n")

    # 2) BM25
    print("[UNIFIED RAG] BM25 인덱스 구축 중...")
    tokenized = [_tokenize_ko(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(UNIFIED_BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "ids": [c["id"] for c in chunks],
            "docs": [c["text"] for c in chunks],
            "metas": [c["meta"] for c in chunks],
        }, f)

    # 3) Chroma (cosine metric 보장)
    print("[UNIFIED RAG] Chroma 벡터 인덱스 구축 중...")
    client = chromadb.PersistentClient(path=UNIFIED_CHROMA_DIR)
    try:
        client.delete_collection(UNIFIED_CHROMA_COLLECTION)
    except Exception as e:
        print(f"[WARN] 기존 컬렉션 삭제 실패: {e}")

    embedder = LocalEmbedder()

    class _EmbedFunc(embedding_functions.EmbeddingFunction):
        def __call__(self, texts):
            return embedder.encode(texts).tolist()

    col = client.create_collection(
        name=UNIFIED_CHROMA_COLLECTION,
        embedding_function=_EmbedFunc(),
        metadata={"hnsw:space": "cosine"}   
    )

        # 배치 크기 설정 (ChromaDB 제한보다 작게)
    BATCH_SIZE = 5000
    
    # 배치별로 ChromaDB에 추가
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        batch_ids = [c["id"] for c in batch_chunks]
        batch_docs = [c["text"] for c in batch_chunks]
        batch_metas = [c["meta"] for c in batch_chunks]
        
        print(f"[UNIFIED RAG] Chroma 배치 추가 중: {i+1}-{min(i+BATCH_SIZE, len(chunks))}/{len(chunks)}")
        col.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
    
    # 통계 정보 출력
    law_chunks = [c for c in chunks if c["meta"].get("doc_type") == "law"]
    security_chunks = [c for c in chunks if c["meta"].get("doc_type") == "security"]
    
    print(f"[UNIFIED RAG] 통합 인덱스 구축 완료!")
    print(f"  - 총 청크 수: {len(chunks)}개")
    print(f"  - 법령 청크: {len(law_chunks)}개")
    print(f"  - 보안/경제 청크: {len(security_chunks)}개")
    print(f"  - 저장 경로: {UNIFIED_CHROMA_DIR}")
