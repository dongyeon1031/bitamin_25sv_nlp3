# -*- coding: utf-8 -*-
import os, json, pickle
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from rag.embeddings import LocalEmbedder
from configs.rag_config import CHROMA_DIR, BM25_INDEX_PATH, DOCSTORE_PATH

def _tokenize_ko(text: str):
    import re
    text = text.lower()
    return re.findall(r"[가-힣]+|[a-z]+|\d+", text)

def build_and_persist(chunks: List[Dict]):
    os.makedirs(os.path.dirname(DOCSTORE_PATH), exist_ok=True)

    # 1) Docstore (chunk 단위)
    with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 2) BM25
    tokenized = [_tokenize_ko(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25}, f)

    # 3) Chroma (벡터)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection("law_chunks")
    except Exception:
        pass

    embedder = LocalEmbedder()
    class _EmbedFunc(embedding_functions.EmbeddingFunction):
        def __call__(self, texts):
            return embedder.encode(texts).tolist()

    col = client.create_collection(name="law_chunks", embedding_function=_EmbedFunc())
    ids = [c["id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [c["meta"] for c in chunks]
    col.add(ids=ids, documents=docs, metadatas=metas)
