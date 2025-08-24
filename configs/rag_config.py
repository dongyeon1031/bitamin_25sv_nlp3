import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

PDF_DIR = os.path.join(PROJECT_ROOT, "data")


PDF_LIST = [
    (r"data/신용정보의 이용 및 보호에 관한 법률 시행령(대통령령)(제35227호)(20250121).pdf", "신용정보법", "법률"),
    (r"data/전자금융거래법(법률)(제19734호)(20240915) (1).pdf", "전자금융거래법", "법률"),
    (r"data/정보통신망 이용촉진 및 정보보호 등에 관한 법률(법률)(제20678호)(20250722).pdf", "정보통신망법", "법률"),
    (r"data/전자서명법(법률)(제18479호)(20221020).pdf", "전자서명법", "법률"),
    (r"data/개인정보 보호법(법률)(제20897호)(20251002).pdf", "개인정보보호법", "법률"),
    (r"data/금융위원회와 그 소속기관 직제(대통령령)(제35517호)(20250520).pdf", "금융위원회와 그 소속기관 직제", "대통령령"),
]

PDF_LIST = [(os.path.normpath(p), name, kind) for (p, name, kind) in PDF_LIST]

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

#임베딩(로컬 경로 필수)
EMBEDDING_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bge-m3")
SEED = 42
CHROMA_COLLECTION = "law_chunks"

