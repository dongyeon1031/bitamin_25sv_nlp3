import re
from prompts.builder import extract_question_and_choices
from tqdm import tqdm
from prompts.builder import make_prompt_auto, make_prompt_with_context
from utils.classify import is_multiple_choice
from rag.retriever import HybridRetriever
from rag.simple_ensemble_reranker import SimpleEnsembleReranker
from configs.rag_config import (
    FINAL_CONTEXT_K, RRF_WEIGHT, CROSS_ENCODER_WEIGHT, CROSS_ENCODER_MODEL
)

def is_law_related(question: str) -> bool:
    patterns = [
        r"제\s*\d+\s*조",        
        r"제\s*\d+\s*조의\s*\d+", 
        r"시행령",
        r"시행규칙",
        r"[가-힣]+법"            
    ]
    return any(re.search(p, question) for p in patterns)

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
    레이어드 Ensemble Reranker를 사용한 조건부 RAG 추론
    - 1단계: RRF로 BM25 + Vector 융합 (기존 유지)
    - 2단계: RRF + CrossEncoder 가중합으로 재정렬 (추가)
    - 법령 질문: 무조건 RAG
    - 일반 질문: retriever 상위 score >= threshold 이면 RAG
    - 그 외: LLM 단독
    """
    retriever = HybridRetriever()
    
    # 레이어드 Ensemble Reranker 초기화
    ensemble_reranker = None
    if use_ensemble:
        ensemble_reranker = SimpleEnsembleReranker(
            cross_encoder_model=CROSS_ENCODER_MODEL,
            rrf_weight=RRF_WEIGHT,
            cross_encoder_weight=CROSS_ENCODER_WEIGHT
        )
    
    preds = []

    for q in tqdm(test_df["Question"], desc="Inference (Layered Ensemble)"):
        use_rag = False
        contexts = []
        cands = retriever.retrieve(q, merge_top_k=top_k_retrieve)

        # 1) 법령 질문이면 무조건 RAG
        if is_law_related(q) and cands:
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

