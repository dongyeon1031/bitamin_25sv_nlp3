import re
from prompts.builder import extract_question_and_choices
from tqdm import tqdm
from prompts.builder import make_prompt_auto
from utils.classify import is_multiple_choice
from finetune.security_rag.security_retriever import SecurityHybridRetriever
from finetune.security_rag.security_ensemble_reranker import SecurityEnsembleReranker
from finetune.security_configs.security_rag_config import (
    FINAL_CONTEXT_K, RRF_WEIGHT, CROSS_ENCODER_WEIGHT, CROSS_ENCODER_MODEL
)

def is_security_related(question: str) -> bool:
    """보안/경제 관련 질문인지 판단"""
    security_patterns = [
        r"보안", r"해킹", r"침입", r"위협", r"공격",
        r"홈네트워크", r"사이버", r"바이러스", r"악성코드",
        r"방화벽", r"암호화", r"인증", r"접근제어",
        r"경제", r"금융", r"투자", r"주식", r"환율",
        r"인플레이션", r"GDP", r"경기", r"시장"
    ]
    return any(re.search(p, question) for p in security_patterns)

def make_security_prompt_with_context(text: str, contexts: list):
    """보안/경제 데이터용 프롬프트 생성"""
    ctx_block = "\n\n".join([
        f"- 출처:{c.get('meta',{}).get('doc_name','?')} "
        f"유형:{c.get('meta',{}).get('doc_type','?')}\n{c['text'][:400]}"
        for c in contexts[:2]
    ])

    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = (
            "당신은 사이버보안 및 경제 전문가입니다.\n"
            "아래 제공된 보안/경제 관련 문맥(Context)만 근거로 사용하여 **정답 선택지 번호 하나만 출력**하세요.\n"
            "정답은 반드시 1,2,3,4,5 중 하나 숫자만 출력합니다.\n"
            "추론 과정, 설명, 다른 텍스트는 절대 출력하지 마세요.\n\n"
            f"[Context]\n{ctx_block}\n\n"
            f"질문: {question}\n"
            "선택지:\n"
            f"{chr(10).join(options)}\n\n"
            "답변:"
        )
    else:
        prompt = (
            "당신은 사이버보안 및 경제 전문가입니다.\n"
            "아래 제공된 보안/경제 관련 문맥(Context)만 근거로 사용하여 "
            "아래 주관식 질문에 대해 핵심 키워드 중심으로 정확하고 간략한 설명을 작성하세요\n"
            "중요: 답변은 반드시 한 문장으로 작성하고, 불필요한 서론/결론은 생략하세요.\n"
            "답변 이외의 부가적인 인사말은 생략하세요.\n\n"
            f"[Context]\n{ctx_block}\n\n"
            f"질문: {text}\n\n"
            "답변:"
        )
    return prompt

def extract_answer_only(generated_text: str, original_question: str) -> str:
    """답변만 추출"""
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

def run_security_inference_ensemble(llm, test_df, score_threshold: float = 0.01,
                                   use_ensemble: bool = True, top_k_retrieve: int = 20):
    """
    보안/경제 데이터용 레이어드 Ensemble Reranker를 사용한 조건부 RAG 추론
    - 1단계: RRF로 BM25 + Vector 융합
    - 2단계: RRF + CrossEncoder 가중합으로 재정렬
    - 보안/경제 질문: 무조건 RAG
    - 일반 질문: retriever 상위 score >= threshold 이면 RAG
    - 그 외: LLM 단독
    """
    retriever = SecurityHybridRetriever()
    
    # 레이어드 Ensemble Reranker 초기화
    ensemble_reranker = None
    if use_ensemble:
        ensemble_reranker = SecurityEnsembleReranker(
            cross_encoder_model=CROSS_ENCODER_MODEL,
            rrf_weight=RRF_WEIGHT,
            cross_encoder_weight=CROSS_ENCODER_WEIGHT
        )
    
    preds = []

    for q in tqdm(test_df["Question"], desc="보안 RAG 추론"):
        use_rag = False
        contexts = []
        cands = retriever.retrieve(q, merge_top_k=top_k_retrieve)

        # 1) 보안/경제 질문이면 무조건 RAG
        if is_security_related(q) and cands:
            use_rag = True
        # 2) 일반 질문 → RRF score 기준
        elif cands and cands[0]["score"] >= score_threshold:
            use_rag = True

        if use_rag:
            if ensemble_reranker and cands:
                # 레이어드 Ensemble Reranker로 재순위화
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
            prompt = make_security_prompt_with_context(q, contexts)
        else:
            prompt = make_prompt_auto(q)

        response = llm(prompt, max_tokens=512, temperature=0.2, top_p=0.9)
        generated_text = response["choices"][0].get("text", "") or response["choices"][0].get("content", "")
        preds.append(extract_answer_only(generated_text, q))

    return preds

# 기존 함수명 유지 (하위 호환성)
def run_security_inference_mixed(llm, test_df, score_threshold: float = 0.01,
                                use_reranker: bool = True, top_k_retrieve: int = 20):
    """
    기존 함수명 유지 (하위 호환성)
    """
    return run_security_inference_ensemble(
        llm, test_df, score_threshold, 
        use_ensemble=use_reranker, 
        top_k_retrieve=top_k_retrieve
    )
