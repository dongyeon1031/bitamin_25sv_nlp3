import os, unicodedata, glob

def resolve_path(p: str) -> str:
    # 1) NFC 정규화
    p_nfc = unicodedata.normalize("NFC", p)
    if os.path.exists(p_nfc):
        return p_nfc

    d = os.path.dirname(p_nfc)
    b = os.path.basename(p_nfc)

    # 2) 디렉터리 내 실제 이름들과 NFC 매핑 비교
    try:
        names = os.listdir(d)
    except FileNotFoundError:
        return p_nfc  # 상위 디렉터리가 없으면 그대로 반환

    nfc2real = {unicodedata.normalize("NFC", nm): nm for nm in names}
    # 완전 일치(정규화 후)
    if b in nfc2real:
        return os.path.join(d, nfc2real[b])

    # 3) 흔한 변형 보정: " (1)" 제거 시도
    if " (1)" in b:
        b2 = b.replace(" (1)", "")
        if b2 in nfc2real:
            return os.path.join(d, nfc2real[b2])

    # 4) 널널한 글롭: 접두부 + *.pdf
    stem = b.rsplit(".pdf", 1)[0]
    cand = glob.glob(os.path.join(d, f"{stem}*.pdf"))
    if cand:
        return cand[0]

    # 5) 마지막 시도: 공백 연속 축약
    b3 = " ".join(stem.split())
    cand = glob.glob(os.path.join(d, f"{b3}*.pdf"))
    if cand:
        return cand[0]

    return p_nfc

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PDF_DIR = os.path.join(PROJECT_ROOT, "data")

RAW_LIST = [
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

PDF_LIST = []
for rel, name, kind in RAW_LIST:
    abs_path = os.path.join(PROJECT_ROOT, rel.replace("/", os.sep))
    resolved = resolve_path(abs_path)
    PDF_LIST.append((resolved, name, kind))

print("[CONFIG] PDF 파일 경로 확인 중...")
for pdf_path, name, kind in PDF_LIST:
    print(("[CONFIG] ✅ " if os.path.exists(pdf_path) else "[CONFIG] ❌ ") + f"{name}: {os.path.basename(pdf_path)} -> {pdf_path}")

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
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"

# 레이어드 Ensemble Reranker 설정 (RRF + CrossEncoder 가중합)
RRF_WEIGHT = 0.4           # RRF 점수 가중치 (BM25+Vector 융합 결과)
CROSS_ENCODER_WEIGHT = 0.6 # CrossEncoder 점수 가중치

SEED = 42
CHROMA_COLLECTION = "law_chunks"



