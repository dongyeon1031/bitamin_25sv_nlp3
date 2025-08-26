import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 새로운 보안/경제 데이터 목록
SECURITY_DATA_LIST = [
    (os.path.join(DATA_DIR, "2024년 홈네트워크 보안가이드 개정본(2024.06.).pdf"), "홈네트워크 보안가이드", "보안가이드"),
    (os.path.join(DATA_DIR, "2024 하반기 사이버 위협 동향 보고서.pdf"), "사이버 위협 동향 보고서", "보고서"),
    (os.path.join(DATA_DIR, "20250815_시사경제용어사전.xlsx"), "시사경제용어사전", "용어사전"),
]

# 인덱스 저장 경로 (기존과 분리)
INDEX_DIR = os.path.join(PROJECT_ROOT, "finetune", "indexes")
CHROMA_DIR = os.path.join(INDEX_DIR, "chroma_security")
BM25_INDEX_PATH = os.path.join(INDEX_DIR, "bm25_security.pkl")
DOCSTORE_PATH = os.path.join(INDEX_DIR, "docstore_security.jsonl")

# 청킹 설정 (보안/경제 문서에 맞게 조정)
CHUNK_SIZE_CHARS = 1000  # 법령보다 조금 작게
CHUNK_OVERLAP_CHARS = 200
MAX_SECTION_SIZE_CHARS = 5000

# 검색 설정
TOP_K_VECTOR = 30
TOP_K_BM25 = 30
MERGE_TOP_K = 20
FINAL_CONTEXT_K = 6

# 모델 설정 (기존과 동일)
EMBEDDING_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bge-m3-ko")
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"

# 앙상블 설정
RRF_WEIGHT = 0.4
CROSS_ENCODER_WEIGHT = 0.6

SEED = 42
CHROMA_COLLECTION = "security_chunks"
