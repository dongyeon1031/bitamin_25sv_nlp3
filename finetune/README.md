# 보안/경제 데이터 RAG 시스템

기존 법령 RAG 시스템과 별개로 보안/경제 관련 데이터를 위한 RAG 시스템입니다.

## 포함된 데이터

1. **2024년 홈네트워크 보안가이드 개정본(2024.06.).pdf**
   - 홈네트워크 보안 설정 방법
   - 보안 체크리스트 및 가이드라인

2. **2024 하반기 사이버 위협 동향 보고서.pdf**
   - 최신 사이버 보안 위협 동향
   - 공격 패턴 및 통계 분석

3. **20250815_시사경제용어사전.xlsx**
   - 경제 관련 용어 정의
   - 시사경제 키워드 정리

## 시스템 특징

### 1. 기존 법령 RAG와 완전 분리
- 별도의 인덱스 저장 경로 (`./finetune/indexes/`)
- 독립적인 설정 파일 (`security_rag_config.py`)
- 기존 시스템에 영향 없음

### 2. 보안/경제 데이터 특화 처리
- **그림/표 제거**: PDF에서 텍스트만 추출
- **섹션별 분할**: 법령 조문이 아닌 문서 섹션 단위로 분할
- **Excel 지원**: 용어사전 Excel 파일 처리

### 3. 동일한 앙상블 기술 사용
- **BM25 + Vector 검색** 융합
- **CrossEncoder 재순위화**
- **조건부 RAG**: 보안/경제 관련 질문 자동 감지

## 사용법

### 1. 인덱스 구축
```bash
cd finetune/security_rag
python build_security_index.py
```

### 2. 추론 실행
```python
from finetune.security_inference.security_runner import run_security_inference_ensemble

# 보안/경제 데이터 RAG 추론
predictions = run_security_inference_ensemble(
    llm=your_llm_model,
    test_df=test_dataframe,
    score_threshold=0.01,
    use_ensemble=True,
    top_k_retrieve=20
)
```

## 파일 구조

```
finetune/
├── security_configs/
│   └── security_rag_config.py      # 보안 RAG 설정
├── security_rag/
│   ├── build_security_index.py     # 인덱스 구축 스크립트
│   ├── security_pdf_loader.py      # PDF/Excel 로더
│   ├── security_chunker.py         # 청킹
│   ├── security_indexer.py         # 인덱스 구축
│   ├── security_retriever.py       # 검색
│   └── security_ensemble_reranker.py # 재순위화
├── security_inference/
│   └── security_runner.py          # 추론 러너
└── indexes/                        # 인덱스 저장 경로
    ├── chroma_security/
    ├── bm25_security.pkl
    └── docstore_security.jsonl
```

## 설정 옵션

### 청킹 설정
- `CHUNK_SIZE_CHARS = 1000`: 기본 청크 크기
- `CHUNK_OVERLAP_CHARS = 200`: 오버랩 크기
- 용어사전: 70% 크기, 보고서: 120% 크기로 자동 조정

### 검색 설정
- `TOP_K_VECTOR = 30`: 벡터 검색 결과 수
- `TOP_K_BM25 = 30`: BM25 검색 결과 수
- `FINAL_CONTEXT_K = 6`: 최종 컨텍스트 수

### 앙상블 설정
- `RRF_WEIGHT = 0.4`: RRF 점수 가중치
- `CROSS_ENCODER_WEIGHT = 0.6`: CrossEncoder 점수 가중치

## 보안/경제 질문 자동 감지

다음 키워드가 포함된 질문은 자동으로 RAG를 사용합니다:

**보안 관련**: 보안, 해킹, 침입, 위협, 공격, 홈네트워크, 사이버, 바이러스, 악성코드, 방화벽, 암호화, 인증, 접근제어

**경제 관련**: 경제, 금융, 투자, 주식, 환율, 인플레이션, GDP, 경기, 시장

## 주의사항

1. **기존 법령 RAG와 완전 분리**: 서로 영향을 주지 않습니다.
2. **그림/표 제외**: PDF에서 텍스트만 추출하여 처리합니다.
3. **Excel 파일**: pandas를 사용하여 용어사전을 처리합니다.
4. **메모리 사용량**: 대용량 PDF 파일 처리 시 충분한 메모리가 필요합니다.
