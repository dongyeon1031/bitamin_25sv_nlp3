import re
from tqdm import tqdm
from prompts.builder import make_prompt_auto
from utils.classify import is_multiple_choice

def extract_answer_only(generated_text: str, original_question: str) -> str:
    if "답변:" in generated_text:
        text = generated_text.split("답변:")[-1].strip()
    else:
        text = generated_text.strip()

    if not text:
        return "미응답"

    is_mc = is_multiple_choice(original_question)

    if is_mc:
        match = re.match(r"\D*([1-9][0-9]?)", text)
        if match:
            return match.group(1)
        else:
            return "0"
    else:
        return text

def run_inference(llm, test_df):
    preds = []
    for q in tqdm(test_df['Question'], desc="Inference"):
        prompt = make_prompt_auto(q)
        response = llm(prompt, max_tokens=512, temperature=0.2, top_p=0.9)
        generated_text = response['choices'][0]['text']
        pred_answer = extract_answer_only(generated_text, original_question=q)
        preds.append(pred_answer)
    return preds