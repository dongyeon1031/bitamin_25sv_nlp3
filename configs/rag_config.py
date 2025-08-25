import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PDF_DIR = os.path.join(PROJECT_ROOT, "data")
PDF_LIST = [
    (os.path.normpath(os.path.join(PROJECT_ROOT, p)), name, kind)
    for (p, name, kind) in PDF_LIST
]

INDEX_DIR = os.path.join(PROJECT_ROOT, "indexes")
CHROMA_DIR = os.path.join(INDEX_DIR, "chroma")
BM25_INDEX_PATH = os.path.join(INDEX_DIR, "bm25.pkl")
DOCSTORE_PATH = os.path.join(INDEX_DIR, "docstore.jsonl")

# 청킹
CHUNK_SIZE_CHARS = 1200
CHUNK_OVERLAP_CHARS = 240
MAX_SECTION_SIZE_CHARS = 6000

# 검색 파라미터
TOP_K_VECTOR = 40
TOP_K_BM25 = 40
MERGE_TOP_K = 24
FINAL_CONTEXT_K = 8
ALPHA_VECTOR = 0.6


EMBEDDING_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bge-m3")
SEED = 42
CHROMA_COLLECTION = "law_chunks"

