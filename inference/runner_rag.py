import re
import numpy as np
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

def gen_query_variants(llm, q, n=3):
    """Multi-Query + HyDE로 쿼리 확장"""
    variants = [q]
    
    # 1) Multi-Query
    mq_prompt = f"다음 질문을 한국어로 의미를 바꾸지 않게 1문장씩 {n}가지로 바꿔 써줘.\n질문: {q}\n출력은 줄바꿈으로 구분된 문장만."
    mq = llm(mq_prompt, max_tokens=128, temperature=0.8, top_p=0.9)["choices"][0]["text"]
    variants += [v.strip() for v in mq.split("\n") if v.strip()]

    # 2) HyDE 한 줄
    hyde_prompt = f"다음 질문에 대한 그럴듯한 한 줄 요약답을 간결히 써줘(추론/설명 없이 한 문장):\n{q}\n답:"
    hyde = llm(hyde_prompt, max_tokens=64, temperature=0.7, top_p=0.9)["choices"][0]["text"].strip()
    variants.append(hyde)
    
    # 중복 제거
    seen, uniq = set(), []
    for v in variants:
        if v and v not in seen:
            uniq.append(v); seen.add(v)
    return uniq[:1+n+1]

def pick_mc_with_ce(question, options, contexts, cross_encoder, max_ctx=5):
    """CrossEncoder로 객관식 정답 결정"""
    # options: ["1 ...", "2 ...", ...]
    opt_nums, opt_texts = [], []
    for opt in options:
        m = re.match(r"^\s*(\d+)", opt)
        num = m.group(1) if m else str(len(opt_nums)+1)
        body = re.sub(r"^\s*\d+\s*[).:\-]?\s*", "", opt).strip()
        opt_nums.append(num)
        opt_texts.append(body)

    pairs = []
    used_ctx = contexts[:max_ctx]
    for body in opt_texts:
        q_opt = f"{question}\n선택지: {body}"
        for c in used_ctx:
            ctx = c["text"][:1200]  # 안전 컷
            pairs.append((q_opt, ctx))

    scores = cross_encoder.predict(pairs, batch_size=16)
    k = len(used_ctx)
    agg = [float(np.mean(scores[i*k:(i+1)*k])) for i in range(len(opt_texts))]
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

def run_inference_ensemble(llm, test_df, score_threshold: float = 0.01,
                          use_ensemble: bool = True, top_k_retrieve: int = 30):
    """
    통합 레이어드 Ensemble Reranker를 사용한 조건부 RAG 추론
    - 1단계: RRF로 BM25 + Vector 융합 (법령 + 보안/경제 통합)
    - 2단계: RRF + CrossEncoder 가중합으로 재정렬
    - 법령/보안 질문: 무조건 RAG
    - 일반 질문: retriever 상위 score >= threshold 이면 RAG
    - 그 외: LLM 단독
    - 객관식: CrossEncoder로 정답 결정 (우선순위)
    - 검색: Multi-Query + HyDE로 풍부화
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
    
    preds = []

    for q in tqdm(test_df["Question"], desc="Inference (Unified RAG)"):
        use_rag = False
        contexts = []
        
        # Multi-Query + HyDE로 검색 풍부화
        exp_qs = gen_query_variants(llm, q, n=3)
        merged = {}
        for qq in exp_qs:
            for r in retriever.retrieve(qq, merge_top_k=top_k_retrieve):
                if r["id"] not in merged or merged[r["id"]]["score"] < r["score"]:
                    merged[r["id"]] = r
        cands = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k_retrieve]

        # 1) 법령/보안 관련 질문이면 무조건 RAG
        if (is_law_related(q) or is_security_related(q)) and cands:
            use_rag = True
        # 2) 일반 질문 → RRF score 기준
        elif cands and cands[0]["score"] >= score_threshold:
            use_rag = True

        if use_rag:
            if ensemble_reranker and cands:
                # 통합 레이어드 Ensemble Reranker로 재순위화
                cands = ensemble_reranker.rerank(q, cands, top_k=FINAL_CONTEXT_K)

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
            
            # 객관식이면 CrossEncoder로 정답 결정 (우선순위)
            if is_multiple_choice(q) and ensemble_reranker:
                question, options = extract_question_and_choices(q)
                ans = pick_mc_with_ce(question, options, contexts, ensemble_reranker.cross_encoder)
                preds.append(ans)
                continue
                
            prompt = make_prompt_with_context(q, contexts)
        else:
            prompt = make_prompt_auto(q)

        response = llm(prompt, max_tokens=512, temperature=0.2, top_p=0.9)
        generated_text = response["choices"][0].get("text", "") or response["choices"][0].get("content", "")
        preds.append(extract_answer_only(generated_text, q))

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

