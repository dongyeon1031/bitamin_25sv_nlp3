# -*- coding: utf-8 -*-
import re
from tqdm import tqdm
from prompts.builder import make_prompt_with_context
from utils.classify import is_multiple_choice
from rag.retriever import HybridRetriever
from configs.rag_config import FINAL_CONTEXT_K

def extract_answer_only(generated_text: str, original_question: str) -> str:
    if "답변:" in generated_text:
        text = generated_text.split("답변:")[-1].strip()
    else:
        text = generated_text.strip()
    if not text:
        return "미응답"
    if is_multiple_choice(original_question):
        match = re.match(r"\D*([1-9][0-9]?)", text)
        return match.group(1) if match else "0"
    return text

def run_inference_with_rag(llm, test_df):
    retriever = HybridRetriever()
    preds = []
    for q in tqdm(test_df['Question'], desc="Inference+RAG"):
        # 1) Retrieve (hybrid)
        cands = retriever.retrieve(q)


        # 3) Parent 확대(같은 parent_id 중 대표 하나만 뽑혔다면 parent_text로 확장)
        used = []
        contexts = []
        for c in cands:
            pid = c["meta"].get("parent_id")
            if pid and pid not in used:
                contexts.append({"text": c.get("parent_text", c["text"]), "meta": c.get("meta", {})})
                used.append(pid)
            else:
                contexts.append({"text": c["text"], "meta": c.get("meta", {})})
            if len(contexts) >= FINAL_CONTEXT_K:
                break

        # 4) Prompt
        prompt = make_prompt_with_context(q, contexts)

        # 5) LLM 호출
        response = llm(prompt, max_tokens=512, temperature=0.2, top_p=0.9)
        generated_text = response['choices'][0]['text']
        pred_answer = extract_answer_only(generated_text, original_question=q)
        preds.append(pred_answer)
    return preds
