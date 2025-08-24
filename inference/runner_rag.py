# inference/runner_rag.py
import re
from tqdm import tqdm
from prompts.builder import make_prompt_auto, make_prompt_with_context
from utils.classify import is_multiple_choice
from rag.retriever import HybridRetriever
from configs.rag_config import FINAL_CONTEXT_K

def is_law_related(question: str) -> bool:
    keywords = ["조항", "법", "제", "시행령", "시행규칙"]
    return any(kw in question for kw in keywords)

def extract_answer_only(generated_text: str, original_question: str) -> str:
    if "답변:" in generated_text:
        text = generated_text.split("답변:")[-1].strip()
    else:
        text = generated_text.strip()
    if not text:
        return "미응답"
    if is_multiple_choice(original_question):
        match = re.search(r"([0-9]+)", text)
        return match.group(1) if match else "0"
    return text

def run_inference_mixed(llm, test_df, score_threshold: float = 0.01):
    """
    조건부 RAG 추론 (RRF 기반 retriever 사용)
    - 법령 관련 질문: 무조건 RAG
    - 일반 질문: retriever 상위 score >= threshold 이면 RAG
    - 그 외: LLM 단독
    """
    retriever = HybridRetriever()
    preds = []

    for q in tqdm(test_df["Question"], desc="Inference (Mixed, RRF)"):
        use_rag = False
        contexts = []
        cands = retriever.retrieve(q)

        # 1) 법령 질문이면 무조건 RAG
        if is_law_related(q) and cands:
            use_rag = True
        # 2) 일반 질문 → RRF score 기준
        elif cands and cands[0]["score"] >= score_threshold:
            use_rag = True

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
            prompt = make_prompt_with_context(q, contexts)
        else:
            prompt = make_prompt_auto(q)

        response = llm(prompt, max_tokens=512, temperature=0.2, top_p=0.9)
        generated_text = response["choices"][0].get("text", "") or response["choices"][0].get("content", "")
        preds.append(extract_answer_only(generated_text, q))

    return preds
