import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PDF_DIR = os.path.join(PROJECT_ROOT, "data")

PDF_LIST = [
    (os.path.join(PROJECT_ROOT, path.replace("/", os.sep)), name, kind)
    for (path, name, kind) in [
        ("data/신용정보의 이용 및 보호에 관한 법률(법률)(제20304호)(20240814).pdf", "신용정보법", "법률"),
        ("data/전자금융거래법(법률)(제19734호)(20240915) (1).pdf", "전자금융거래법", "법률"),
        ("data/정보통신망 이용촉진 및 정보보호 등에 관한 법률(법률)(제20678호)(20250722).pdf", "정보통신망법", "법률"),
        ("data/전자서명법(법률)(제18479호)(20221020).pdf", "전자서명법", "법률"),
        ("data/개인정보 보호법(법률)(제20897호)(20251002).pdf", "개인정보보호법", "법률"),
        ("data/금융위원회와 그 소속기관 직제(대통령령)(제35517호)(20250520).pdf", "금융위원회와 그 소속기관 직제", "대통령령"),
        ("data/전자금융감독규정(금융위원회고시)(제2025-4호)(20250205).pdf", "전자금융감독규정", "고시"),
        ("data/전자금융감독규정시행세칙(금융감독원세칙)(20250205).pdf", "전자금융감독규정 시행세칙", "세칙"),
        ("data/신용정보업감독규정(금융위원회고시)(제2025-2호)(20250205).pdf", "신용정보업감독규정", "고시"),
    ]
]

INDEX_DIR = os.path.join(PROJECT_ROOT, "indexes")
CHROMA_DIR = os.path.join(INDEX_DIR, "chroma")
BM25_INDEX_PATH = os.path.join(INDEX_DIR, "bm25.pkl")
DOCSTORE_PATH = os.path.join(INDEX_DIR, "docstore.jsonl")

# 청킹
CHUNK_SIZE_CHARS = 1200
CHUNK_OVERLAP_CHARS = 240
MAX_SECTION_SIZE_CHARS = 6000

# 검색 설정
TOP_K_VECTOR = 40
TOP_K_BM25 = 40
MERGE_TOP_K = 24
FINAL_CONTEXT_K = 8
ALPHA_VECTOR = 0.6

# 모델 설정
EMBEDDING_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bge-m3-ko")
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"

# 레이어드 Ensemble Reranker 설정 (RRF + CrossEncoder 가중합)
RRF_WEIGHT = 0.4           # RRF 점수 가중치 (BM25+Vector 융합 결과)
CROSS_ENCODER_WEIGHT = 0.6 # CrossEncoder 점수 가중치

SEED = 42
CHROMA_COLLECTION = "law_chunks"



