import re
import requests
import json
import numpy as np
from collections import Counter
from prompts.builder import extract_question_and_choices
from tqdm import tqdm
from prompts.builder import make_prompt_auto, make_prompt_with_context
from utils.classify import is_multiple_choice
from unified_rag.unified_retriever import UnifiedHybridRetriever
from unified_rag.unified_reranker import UnifiedEnsembleReranker
from configs.unified_rag_config import (
    FINAL_CONTEXT_K, RRF_WEIGHT, CROSS_ENCODER_WEIGHT, CROSS_ENCODER_MODEL,
    LAW_KEYWORDS, SECURITY_KEYWORDS
)

def force_numeric_answer(text: str, num_options: int) -> str:
    """
    모델 출력에서 첫 번째 정수를 추출해 1..num_options 범위만 허용.
    실패 시 '0' 반환.
    """
    if not text:
        return "0"
    t = re.sub(r"\s+", " ", text).strip()
    # "정답: 3", "3번", "3 입니다" 등 모두 포괄
    m = re.search(r"(?:정답\s*[:은는]\s*)?([0-9]+)\s*(?:번|[)．.])?", t)
    if not m:
        m = re.search(r"\b([0-9]+)\b", t)
    if not m:
        return "0"
    n = int(m.group(1))
    return str(n) if 1 <= n <= max(1, num_options) else "0"

def is_law_related(question: str) -> bool:
    """법령 관련 질문인지 확인 (기존 패턴 + 키워드 기반)"""
    # 기존 패턴
    patterns = [
        r"제\s*\d+\s*조",        
        r"제\s*\d+\s*조의\s*\d+", 
        r"시행령",
        r"시행규칙",
        r"[가-힣]+법"            
    ]
    
    # 패턴 매칭
    if any(re.search(p, question) for p in patterns):
        return True
    
    # 키워드 매칭
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in LAW_KEYWORDS)

def is_security_related(question: str) -> bool:
    """보안/경제 관련 질문인지 확인"""
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in SECURITY_KEYWORDS)

def extract_keywords_from_samples(samples: list, min_freq: int = 2) -> str:
    """
    다중 샘플에서 중복 토큰 기반 핵심어 추출
    Args:
        samples: 생성된 답변 샘플 리스트
        min_freq: 최소 등장 빈도
    Returns:
        합의된 답변
    """
    if not samples:
        return "미응답"
    
    # 모든 샘플을 토큰으로 분해
    all_tokens = []
    for sample in samples:
        # 답변 부분만 추출
        if "답변:" in sample:
            text = sample.split("답변:")[-1].strip()
        else:
            text = sample.strip()
        
        # 한국어 단어 단위로 토큰화 (간단한 방식)
        tokens = re.findall(r'[가-힣]+|[a-zA-Z]+|\d+', text)
        all_tokens.extend(tokens)
    
    # 빈도 계산
    token_freq = Counter(all_tokens)
    
    # 최소 빈도 이상의 토큰만 선택
    common_tokens = [token for token, freq in token_freq.items() if freq >= min_freq]
    
    if not common_tokens:
        # 공통 토큰이 없으면 첫 번째 샘플 반환
        return samples[0].split("답변:")[-1].strip() if "답변:" in samples[0] else samples[0].strip()
    
    # 공통 토큰을 포함하는 가장 긴 샘플 선택
    best_sample = ""
    max_common_count = 0
    
    for sample in samples:
        if "답변:" in sample:
            text = sample.split("답변:")[-1].strip()
        else:
            text = sample.strip()
        
        common_count = sum(1 for token in common_tokens if token in text)
        if common_count > max_common_count:
            max_common_count = common_count
            best_sample = text
    
    return best_sample if best_sample else samples[0].split("답변:")[-1].strip()

def generate_multiple_samples(prompt: str, num_samples: int = 3, temperature: float = 0.4) -> list:
    """
    주관식 질문에 대해 다중 샘플 생성
    Args:
        prompt: 프롬프트
        num_samples: 생성할 샘플 수
        temperature: 생성 온도
    Returns:
        생성된 샘플 리스트
    """
    samples = []
    
    for _ in range(num_samples):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "model": "exaone-custom",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                })
            ).json()
            samples.append(response["response"])
        except Exception as e:
            print(f"샘플 생성 중 오류: {e}")
            continue
    
    return samples

def gen_query_variants(llm, q, n=3):
    """
    Multi-Query + HyDE로 쿼리 변형 생성
    Args:
        llm: LLM 모델 (ollama API 사용)
        q: 원본 질문
        n: 생성할 Multi-Query 개수
    Returns:
        쿼리 변형 리스트 (원본 + Multi-Query + HyDE)
    """
    variants = [q]
    
    try:
        # 1) Multi-Query Expansion
        mq_prompt = f"다음 질문을 한국어로 의미를 바꾸지 않게 1문장씩 {n}가지로 바꿔 써줘.\n질문: {q}\n출력은 줄바꿈으로 구분된 문장만."
        mq_response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "model": "exaone-custom",
                "prompt": mq_prompt,
                "stream": False,
                "temperature": 0.8,
            })
        ).json()
        mq = mq_response["response"]
        variants += [v.strip() for v in mq.split("\n") if v.strip()]

        # 2) HyDE (Hypothetical Document Embedding)
        hyde_prompt = f"다음 질문에 대한 그럴듯한 한 줄 요약답을 간결히 써줘(추론/설명 없이 한 문장):\n{q}\n답:"
        hyde_response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "model": "exaone-custom",
                "prompt": hyde_prompt,
                "stream": False,
                "temperature": 0.7,
            })
        ).json()
        hyde = hyde_response["response"].strip()
        variants.append(hyde)
        
        # 중복 제거
        seen, uniq = set(), []
        for v in variants:
            if v and v not in seen:
                uniq.append(v)
                seen.add(v)
        return uniq[:1+n+1]  # 원본 + n개 Multi-Query + 1개 HyDE
        
    except Exception as e:
        print(f"쿼리 변형 생성 중 오류: {e}")
        return [q]  # 오류 시 원본만 반환

def pick_mc_with_ce(question, options, contexts, cross_encoder, max_ctx=5):
    """
    CrossEncoder로 객관식 정답 결정
    Args:
        question: 질문
        options: ["1 ...", "2 ...", ...] 형태의 선택지 리스트
        contexts: 검색된 컨텍스트 리스트
        cross_encoder: CrossEncoder 모델
        max_ctx: 사용할 최대 컨텍스트 수
    Returns:
        선택된 정답 번호 (문자열)
    """
    # 선택지 번호와 텍스트 분리
    opt_nums, opt_texts = [], []
    for opt in options:
        m = re.match(r"^\s*(\d+)", opt)
        num = m.group(1) if m else str(len(opt_nums)+1)
        body = re.sub(r"^\s*\d+\s*[).:\-]?\s*", "", opt).strip()
        opt_nums.append(num)
        opt_texts.append(body)

    # 질문+선택지 vs 컨텍스트 쌍 생성
    pairs = []
    used_ctx = contexts[:max_ctx]
    for body in opt_texts:
        q_opt = f"{question}\n선택지: {body}"
        for c in used_ctx:
            ctx = c["text"][:800]   # 보수적 컷으로 CE 경고 완화
            pairs.append((q_opt, ctx))

    # CrossEncoder 점수 계산
    scores = cross_encoder.predict(pairs, batch_size=16)
    
    # 선택지별 평균 점수 계산
    k = len(used_ctx)
    agg = [float(np.mean(scores[i*k:(i+1)*k])) for i in range(len(opt_texts))]
    
    # 최고점 선택지 번호 반환
    return str(int(opt_nums[int(np.argmax(agg))]))

def extract_answer_only(generated_text: str, original_question: str) -> str:
    if "답변:" in generated_text:
        text = generated_text.split("답변:")[-1].strip()
    else:
        text = generated_text.strip()
    if not text:
        return "미응답"

    if is_multiple_choice(original_question):
        # 선택지 개수 파악
        _, options = extract_question_and_choices(original_question)
        num_options = len(options) if options else 5 

        match = re.search(r"\b([0-9]+)\b", text)
        if match:
            ans = int(match.group(1))
            # 선택지 범위 내 숫자만 인정
            if 1 <= ans <= num_options:
                return str(ans)
        return "0"  
    else:
        return text.strip().replace("\n", " ")

def run_inference_ensemble(test_df, score_threshold: float = 0.01,
                          use_ensemble: bool = True, top_k_retrieve: int = 30):
    """
    통합 레이어드 Ensemble Reranker를 사용한 조건부 RAG 추론
    - 1단계: RRF로 BM25 + Vector 융합 (법령 + 보안/경제 통합)
    - 2단계: RRF + CrossEncoder 가중합으로 재정렬
    - 법령/보안 질문: 무조건 RAG
    - 일반 질문: retriever 상위 score >= threshold 이면 RAG
    - 그 외: LLM 단독
    """
    retriever = UnifiedHybridRetriever()
    
    # 통합 레이어드 Ensemble Reranker 초기화
    ensemble_reranker = None
    if use_ensemble:
        ensemble_reranker = UnifiedEnsembleReranker(
            cross_encoder_model=CROSS_ENCODER_MODEL,
            rrf_weight=RRF_WEIGHT,
            cross_encoder_weight=CROSS_ENCODER_WEIGHT
        )
    
    # 샘플링으로 동적 임계값 추정
    print("동적 임계값 추정 중...")
    sample_scores = []
    for q in list(test_df["Question"])[:30]:
        s = retriever.retrieve(q, merge_top_k=5)
        if s: 
            sample_scores.append(s[0]["score"])
    
    auto_thr = max(0.01, float(np.quantile(sample_scores, 0.25)) * 0.9) if sample_scores else score_threshold
    print(f"자동 임계값: {auto_thr:.4f} (기존: {score_threshold:.4f})")
    
    preds = []

    for q in tqdm(test_df["Question"], desc="Inference (Unified RAG)"):
        use_rag = False
        contexts = []
        
        # Multi-Query + HyDE로 검색 풍부화
        exp_qs = gen_query_variants(None, q, n=3)  # llm 파라미터는 내부에서 ollama API 사용
        merged = {}
        for qq in exp_qs:
            for r in retriever.retrieve(qq, merge_top_k=top_k_retrieve):
                if r["id"] not in merged or merged[r["id"]]["score"] < r["score"]:
                    merged[r["id"]] = r
        cands = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k_retrieve]

        # 1) 법령/보안 관련 질문이면 무조건 RAG
        if (is_law_related(q) or is_security_related(q)) and cands:
            use_rag = True
        # 2) 일반 질문 → 동적 임계값 기준
        elif cands and cands[0]["score"] >= auto_thr:
            use_rag = True

        if use_rag:
            if ensemble_reranker and cands:
                # 통합 레이어드 Ensemble Reranker로 재순위화
                cands = ensemble_reranker.rerank(q, cands, top_k=FINAL_CONTEXT_K)
                
                # 증거수준 기반 결합: 상위 문맥의 CE 점수 평균이 낮으면 LLM 단독으로 전환
                if cands and len(cands) >= 3:
                    top_ce_scores = [c.get("cross_encoder_score", 0.0) for c in cands[:3]]
                    avg_ce_score = np.mean(top_ce_scores)
                    if avg_ce_score < 0.3:  # CE 점수가 너무 낮으면 LLM 단독 사용
                        use_rag = False
                        print(f"CE 점수 낮음 ({avg_ce_score:.3f}), LLM 단독 사용")

            if use_rag:
                used = []
                for c in cands:
                    pid = c["meta"].get("parent_id")
                    if pid and pid not in used:
                        contexts.append({"text": c.get("parent_text", c["text"]), "meta": c.get("meta", {})})
                        used.append(pid)
                    else:
                        contexts.append({"text": c["text"], "meta": c.get("meta", {})})
                    if len(contexts) >= FINAL_CONTEXT_K:
                        break
                
                # 객관식이고 CrossEncoder가 있으면 CrossEncoder로 정답 결정
                if is_multiple_choice(q) and ensemble_reranker:
                    question, options = extract_question_and_choices(q)
                    ans = pick_mc_with_ce(question, options, contexts, ensemble_reranker.cross_encoder)
                    preds.append(ans)
                    continue
                
                prompt = make_prompt_with_context(q, contexts)
            else:
                prompt = make_prompt_auto(q)
        else:
            prompt = make_prompt_auto(q)

        # 주관식이면 다중 샘플 보팅 적용
        if not is_multiple_choice(q):
            samples = generate_multiple_samples(prompt, num_samples=3, temperature=0.4)
            final_answer = extract_keywords_from_samples(samples, min_freq=2)
            preds.append(final_answer)
        else:
            # 객관식은 단일 샘플
            response = requests.post(
                "http://localhost:11434/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "model": "exaone-custom",  # ollama에 로드한 정확한 모델 이름
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.2,
                })
            ).json()
            generated_text = response["response"]
            _q, _opts = extract_question_and_choices(q)
            pred_answer = force_numeric_answer(generated_text, len(_opts) or 5)
            preds.append(pred_answer)

    return preds

# 기존 함수명 유지 (하위 호환성)
def run_inference_mixed(llm, test_df, score_threshold: float = 0.01,
                        use_reranker: bool = True, top_k_retrieve: int = 30):
    """
    기존 함수명 유지 (하위 호환성)
    """
    return run_inference_ensemble(
        llm, test_df, score_threshold, 
        use_ensemble=use_reranker, 
        top_k_retrieve=top_k_retrieve
    )
