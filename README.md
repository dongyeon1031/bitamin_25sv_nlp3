# Bitamin 25SV NLP3 - 금융 법규 QA 시스템

---

## 실행환경 구성법

### 리눅스 환경 (CUDA + llama-cpp-python)

```bash
chmod +x install.sh
./install.sh
```

### 윈도우 환경 (CUDA 설정 포함)

윈도우는 그런거 없어요.
---

아래 무시하고 실행하는법                          
python model.py                         
python rerank_model.py             
python model2.py                                    

python unified_rag/build_unified_index.py                              

python main.py                       

주의사항                
ERROR: Invalid requirement: 'nvidia-.*-cu12'              
이런 오류가 뜨면 pytorch 버전이 안맞아서 그럼              
python -m pip uninstall -y torch torchvision torchaudio xformers                                                          
python -m pip freeze | grep -E 'nvidia-(cuda|cudnn|cublas|cufft|curand|cusolver|cusparse|nvjitlink|nvtx|nccl).*cu12' \                   
  | cut -d= -f1 | xargs -r python -m pip uninstall -y                                    
python -m pip cache purge                                        

python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio --upgrade                             

이거한번해보셈


## 🎯 프로젝트 개요
금융 관련 법규 문서에 대한 질의응답 시스템으로, RAG(Retrieval-Augmented Generation) 기술을 활용하여 정확한 답변을 생성합니다.

## 🚀 주요 기능

### 📚 문서 처리
- PDF 문서 자동 로딩 및 텍스트 추출
- 법규 문서별 청킹 및 인덱싱
- 한국어 특화 임베딩 모델 사용

### 🔍 하이브리드 검색
- **BM25**: 키워드 기반 검색
- **Vector Search**: 의미적 유사도 검색 (dragonkue/BGE-m3-ko)
- **RRF (Reciprocal Rank Fusion)**: 두 검색 결과 융합

### 🎯 레이어드 Ensemble Reranker (신규!)
**RRF + CrossEncoder 가중합으로 noise를 줄이는 레이어드 재순위화 시스템**

#### 특징:
- **1단계**: RRF로 BM25 + Vector 융합 (기존 유지)
- **2단계**: RRF + CrossEncoder 가중합으로 재정렬 (추가)
- **단순함**: 복잡한 분석 기능 없이 핵심 기능만 구현

#### 가중치 설정:
```python
RRF_WEIGHT = 0.4           # RRF 점수 가중치 (BM25+Vector 융합 결과)
CROSS_ENCODER_WEIGHT = 0.6 # CrossEncoder 점수 가중치
```

#### 기존 방식 vs 개선된 방식:
- **기존**: RRF + CrossEncoder (2단계)
- **개선**: 레이어드 Ensemble (RRF + 가중합)

## 📁 프로젝트 구조

```
bitamin_25sv_nlp3/
├── configs/
│   ├── model_config.py      # 모델 설정
│   └── rag_config.py        # RAG 시스템 설정
├── rag/
│   ├── simple_ensemble_reranker.py # 🆕 레이어드 Ensemble Reranker
│   ├── retriever.py         # 하이브리드 검색 (RRF 포함)
│   ├── embeddings.py        # 임베딩 생성
│   ├── indexer.py           # 인덱스 구축
│   ├── chunker.py           # 문서 청킹
│   ├── pdf_loader.py        # PDF 로딩
│   └── build_index.py       # 인덱스 빌드
├── inference/
│   ├── model.py             # LLM 모델 로딩
│   └── runner_rag.py        # 추론 실행 (레이어드 Ensemble 지원)
├── data/
│   ├── loader.py            # 데이터 로딩
│   └── *.pdf               # 금융 법규 문서들
├── utils/
│   └── classify.py          # 질문 분류
├── prompts/
│   └── builder.py           # 프롬프트 생성
├── output/
│   └── save.py              # 결과 저장
├── main.py                  # 메인 실행 파일
├── test_simple_ensemble.py  # 🆕 레이어드 Ensemble 성능 테스트
├── model.py                 # 🆕 dragonkue/BGE-m3-ko 모델 다운로드
├── rerank_model.py          # CrossEncoder 모델 다운로드
├── model2.py                # LLM 모델 다운로드
└── requirements.txt         # 의존성 패키지
```

## 🛠️ 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 모델 다운로드
```bash
# dragonkue/BGE-m3-ko 임베딩 모델 (한국어 특화)
python model.py

# CrossEncoder 모델
python rerank_model.py

# LLM 모델 (Claude-3.7-Sonnet-Gemma)
python model2.py
```

### 3. 인덱스 구축
```bash
python rag/build_index.py
```

### 4. 추론 실행
```bash
# 레이어드 Ensemble Reranker 사용
python main.py

# 또는 직접 실행
python -c "
from inference.runner_rag import run_inference_ensemble
from inference.model import load_model_and_tokenizer
from data.loader import load_test_data
from configs.model_config import MODEL_NAME

llm = load_model_and_tokenizer(MODEL_NAME)
test_data = load_test_data()
results = run_inference_ensemble(llm, test_data)
print('추론 완료!')
"
```

## 🧪 성능 테스트

### 레이어드 Ensemble Reranker 성능 분석
```bash
python test_simple_ensemble.py
```

이 스크립트는 다음을 수행합니다:
- 레이어드 Ensemble vs 단일 CrossEncoder 성능 비교
- 가중치 설정 확인
- 결과 출력

## ⚙️ 설정 옵션

### 레이어드 Ensemble Reranker 설정 (`configs/rag_config.py`)
```python
# 모델 설정
EMBEDDING_MODEL_PATH = "dragonkue/BGE-m3-ko"  # 한국어 특화 임베딩
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"

# 가중치 조정
RRF_WEIGHT = 0.4           # RRF 점수 가중치
CROSS_ENCODER_WEIGHT = 0.6 # CrossEncoder 점수 가중치
```

### 검색 설정
```python
TOP_K_VECTOR = 40        # 벡터 검색 결과 수
TOP_K_BM25 = 40          # BM25 검색 결과 수
MERGE_TOP_K = 24         # RRF 융합 결과 수
FINAL_CONTEXT_K = 8      # 최종 컨텍스트 수
```

## 📊 성능 개선 효과

### Noise 감소 효과:
1. **RRF 융합**: BM25와 Vector 검색의 장점을 순위 기반으로 결합
2. **가중합 최적화**: RRF와 CrossEncoder의 장점을 극대화
3. **한국어 특화**: dragonkue/BGE-m3-ko로 한국어 법규 문서 최적화

### 예상 개선 효과:
- **검색 정확도**: 20-30% 향상
- **답변 품질**: 관련성 높은 문서 선택으로 개선  
- **안정성**: Noise 감소로 일관성 향상

## 🔧 커스터마이징

### 가중치 조정
```python
# CrossEncoder 중심 (정밀도 우선)
RRF_WEIGHT = 0.2
CROSS_ENCODER_WEIGHT = 0.8

# 균형잡힌 설정
RRF_WEIGHT = 0.5
CROSS_ENCODER_WEIGHT = 0.5

# RRF 중심 (검색 안정성 우선)
RRF_WEIGHT = 0.6
CROSS_ENCODER_WEIGHT = 0.4
```

