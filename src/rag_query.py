"""
검색 전용 rag_query.py (marker 표시 제거 + 필터 옵션)

예시:
  python rag_query.py \
    --persist_dir .,/db/chroma_laws \
    --collection laws-ko \
    --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    --q "가명정보 정의가 뭐야?" \
    --k 5 \
    --contains "가명정보" \
    --doc_type act
"""

import argparse, json
from typing import List, Dict, Any
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer

class STEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

def fmt_source(meta: Dict[str, Any]) -> str:
    doc = meta.get("doc_title") or ""
    art = meta.get("article_title") or ""
    head = meta.get("heading") or ""
    parts = [p for p in [doc, art, head] if p]
    return " · ".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist_dir", required=True)
    ap.add_argument("--collection", default="laws-ko")
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--q", required=True, help="Question")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--json", action="store_true", help="결과를 JSON Lines 형태로 출력")
    ap.add_argument("--score", action="store_true", help="유사도 점수/거리 표시")
    # 필터 옵션
    ap.add_argument("--contains", default=None, help="본문에 반드시 포함되어야 하는 키워드 (where_document $contains)")
    ap.add_argument("--title_contains", default=None, help="doc_title에 포함되어야 하는 문자열")
    ap.add_argument("--doc_type", default=None, choices=["act","decree","rule","other","unknown"], help="ingest 시 추론된 문서 타입으로 필터")
    ap.add_argument("--embed_filter", action="store_true", help="ingest 시 저장된 embed_model과 현재 모델명이 일치하는 문서만 검색")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=args.persist_dir)
    embed_fn = STEmbeddingFunction(args.model)
    try:
        col = client.get_collection(name=args.collection, embedding_function=embed_fn)
    except TypeError:
        col = client.get_collection(name=args.collection)

    where = {}
    if args.title_contains:
        where["doc_title"] = {"$contains": args.title_contains}
    if args.doc_type:
        where["doc_type"] = args.doc_type
    if args.embed_filter:
        where["embed_model"] = args.model

    query_kwargs = dict(query_texts=[args.q], n_results=args.k)
    if where:
        query_kwargs["where"] = where
    if args.contains:
        query_kwargs["where_document"] = {"$contains": args.contains}

    res = col.query(**query_kwargs)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] if res.get("distances") else None

    if args.json:
        for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
            out = {"rank": i, "source": fmt_source(meta), "document": doc, "metadata": meta}
            if args.score and dists is not None:
                out["distance"] = dists[i-1]
            print(json.dumps(out, ensure_ascii=False))
        return

    if not docs:
        print("No results found.")
        return

    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        src = fmt_source(meta)
        print(f"\n--- Result {i} ---")
        print(f"Source: {src}")
        if args.score and dists is not None:
            print(f"Distance: {dists[i-1]}")
        print(doc[:1200] + ("..." if len(doc) > 1200 else ""))

if __name__ == "__main__":
    main()
