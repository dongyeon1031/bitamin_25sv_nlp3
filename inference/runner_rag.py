import re
import requests
import json
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
    
    preds = []

    for q in tqdm(test_df["Question"], desc="Inference (Unified RAG)"):
        use_rag = False
        contexts = []
        cands = retriever.retrieve(q, merge_top_k=top_k_retrieve)

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
            prompt = make_prompt_with_context(q, contexts)
        else:
            prompt = make_prompt_auto(q)

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
        # already defined above
        # generated_text = ...
        pred_answer = extract_answer_only(generated_text, original_question=q)
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

