import re
from prompts.builder import extract_question_and_choices
from tqdm import tqdm
from prompts.builder import make_prompt_auto, make_prompt_with_context
from utils.classify import is_multiple_choice

# 기존 법령 RAG
from rag.retriever import HybridRetriever
from rag.simple_ensemble_reranker import SimpleEnsembleReranker
from configs.rag_config import (
    FINAL_CONTEXT_K, RRF_WEIGHT, CROSS_ENCODER_WEIGHT, CROSS_ENCODER_MODEL
)

# 새로운 보안 RAG
from finetune.rag.security_retriever import SecurityHybridRetriever
from finetune.rag.security_ensemble_reranker import SecurityEnsembleReranker
from finetune.configs.security_rag_config import (
    FINAL_CONTEXT_K as SECURITY_FINAL_CONTEXT_K,
    RRF_WEIGHT as SECURITY_RRF_WEIGHT,
    CROSS_ENCODER_WEIGHT as SECURITY_CROSS_ENCODER_WEIGHT,
    CROSS_ENCODER_MODEL as SECURITY_CROSS_ENCODER_MODEL
)

def is_law_related(question: str) -> bool:
    """법령 관련 질문인지 판단"""
    patterns = [
        r"제\s*\d+\s*조",        
        r"제\s*\d+\s*조의\s*\d+", 
        r"시행령",
        r"시행규칙",
        r"[가-힣]+법"            
    ]
    return any(re.search(p, question) for p in patterns)

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

def run_unified_inference(llm, test_df, 
                         law_score_threshold: float = 0.01,
                         security_score_threshold: float = 0.01,
                         use_ensemble: bool = True,
                         top_k_retrieve: int = 30):
    """
    통합 RAG 추론: 법령 RAG + 보안 RAG
    
    전략:
    1. 법령 질문 → 법령 RAG 사용
    2. 보안/경제 질문 → 보안 RAG 사용  
    3. 일반 질문 → 두 RAG 모두 시도, 더 높은 점수 선택
    """
    
    # 두 RAG 시스템 초기화
    law_retriever = HybridRetriever()
    security_retriever = SecurityHybridRetriever()
    
    # 앙상블 리랭커 초기화
    law_reranker = None
    security_reranker = None
    if use_ensemble:
        law_reranker = SimpleEnsembleReranker(
            cross_encoder_model=CROSS_ENCODER_MODEL,
            rrf_weight=RRF_WEIGHT,
            cross_encoder_weight=CROSS_ENCODER_WEIGHT
        )
        security_reranker = SecurityEnsembleReranker(
            cross_encoder_model=SECURITY_CROSS_ENCODER_MODEL,
            rrf_weight=SECURITY_RRF_WEIGHT,
            cross_encoder_weight=SECURITY_CROSS_ENCODER_WEIGHT
        )
    
    preds = []

    for q in tqdm(test_df["Question"], desc="통합 RAG 추론"):
        use_law_rag = False
        use_security_rag = False
        contexts = []
        
        # 1) 법령 질문이면 법령 RAG 사용
        if is_law_related(q):
            use_law_rag = True
            cands = law_retriever.retrieve(q, merge_top_k=top_k_retrieve)
            
            if cands and cands[0]["score"] >= law_score_threshold:
                if law_reranker and cands:
                    cands = law_reranker.rerank(q, cands, top_k=FINAL_CONTEXT_K)
                
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
                
        # 2) 보안/경제 질문이면 보안 RAG 사용
        elif is_security_related(q):
            use_security_rag = True
            cands = security_retriever.retrieve(q, merge_top_k=top_k_retrieve)
            
            if cands and cands[0]["score"] >= security_score_threshold:
                if security_reranker and cands:
                    cands = security_reranker.rerank(q, cands, top_k=SECURITY_FINAL_CONTEXT_K)
                
                used = []
                for c in cands:
                    pid = c["meta"].get("parent_id")
                    if pid and pid not in used:
                        contexts.append({"text": c.get("parent_text", c["text"]), "meta": c.get("meta", {})})
                        used.append(pid)
                    else:
                        contexts.append({"text": c["text"], "meta": c.get("meta", {})})
                    if len(contexts) >= SECURITY_FINAL_CONTEXT_K:
                        break
                
                # 보안 RAG용 프롬프트 생성
                ctx_block = "\n\n".join([
                    f"- 출처:{c.get('meta',{}).get('doc_name','?')} "
                    f"유형:{c.get('meta',{}).get('doc_type','?')}\n{c['text'][:400]}"
                    for c in contexts[:2]
                ])
                
                if is_multiple_choice(q):
                    question, options = extract_question_and_choices(q)
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
                        f"질문: {q}\n\n"
                        "답변:"
                    )
            else:
                prompt = make_prompt_auto(q)
                
        # 3) 일반 질문 → 두 RAG 모두 시도, 더 높은 점수 선택
        else:
            law_cands = law_retriever.retrieve(q, merge_top_k=top_k_retrieve)
            security_cands = security_retriever.retrieve(q, merge_top_k=top_k_retrieve)
            
            law_score = law_cands[0]["score"] if law_cands else 0
            security_score = security_cands[0]["score"] if security_cands else 0
            
            # 더 높은 점수의 RAG 사용
            if law_score >= law_score_threshold and law_score > security_score:
                use_law_rag = True
                cands = law_cands
                if law_reranker and cands:
                    cands = law_reranker.rerank(q, cands, top_k=FINAL_CONTEXT_K)
                
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
                
            elif security_score >= security_score_threshold and security_score > law_score:
                use_security_rag = True
                cands = security_cands
                if security_reranker and cands:
                    cands = security_reranker.rerank(q, cands, top_k=SECURITY_FINAL_CONTEXT_K)
                
                used = []
                for c in cands:
                    pid = c["meta"].get("parent_id")
                    if pid and pid not in used:
                        contexts.append({"text": c.get("parent_text", c["text"]), "meta": c.get("meta", {})})
                        used.append(pid)
                    else:
                        contexts.append({"text": c["text"], "meta": c.get("meta", {})})
                    if len(contexts) >= SECURITY_FINAL_CONTEXT_K:
                        break
                
                # 보안 RAG용 프롬프트 생성
                ctx_block = "\n\n".join([
                    f"- 출처:{c.get('meta',{}).get('doc_name','?')} "
                    f"유형:{c.get('meta',{}).get('doc_type','?')}\n{c['text'][:400]}"
                    for c in contexts[:2]
                ])
                
                if is_multiple_choice(q):
                    question, options = extract_question_and_choices(q)
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
                        f"질문: {q}\n\n"
                        "답변:"
                    )
            else:
                # 두 RAG 모두 점수가 낮으면 LLM 단독
                prompt = make_prompt_auto(q)

        response = llm(prompt, max_tokens=512, temperature=0.2, top_p=0.9)
        generated_text = response["choices"][0].get("text", "") or response["choices"][0].get("content", "")
        preds.append(extract_answer_only(generated_text, q))

    return preds
