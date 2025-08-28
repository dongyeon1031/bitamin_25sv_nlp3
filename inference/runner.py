import re
import requests
import json
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

def run_inference(test_df):
    preds = []
    for q in tqdm(test_df['Question'], desc="Inference"):
        prompt = make_prompt_auto(q)
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "model": "exaone-custom",
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
