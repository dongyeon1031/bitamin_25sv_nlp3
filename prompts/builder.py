import re
from utils.classify import is_multiple_choice

# 질문과 선택지 분리 함수
def extract_question_and_choices(full_text):
    """
    전체 질문 문자열에서 질문 본문과 선택지 리스트를 분리
    """
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []

    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?\s", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    
    question = " ".join(q_lines)
    return question, options

# 질문과 선택지 분리 함수
def make_prompt_with_context(text: str, contexts: list):
    """
    RAG용 프롬프트. contexts: [{"text":..., "meta":...}, ...]
    """
    ctx_block = "\n\n".join([f"- 출처:{c.get('meta',{}).get('source','?')} p.{c.get('meta',{}).get('page','?')}\n{c['text']}" 
                             for c in contexts])

    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = (
            "당신은 금융보안 및 관련 법규 전문가입니다.\n"
            "다음 문맥(Context)을 근거로 정답 **선택지 번호만** 출력하세요.\n"
            "추론 과정, 설명, 기타 텍스트는 출력하지 마세요.\n\n"
            f"[Context]\n{ctx_block}\n\n"
            f"질문: {question}\n"
            "선택지:\n"
            f"{chr(10).join(options)}\n\n"
            "답변:"
        )
    else:
        prompt = (
            "당신은 금융보안 및 관련 법규 전문가입니다.\n"
            "다음 문맥(Context)을 근거로 한국어 한 문장으로만 간결히 답하세요.\n"
            "불필요한 서론/결론/면책고지는 금지합니다.\n\n"
            f"[Context]\n{ctx_block}\n\n"
            f"질문: {text}\n\n"
            "답변:"
        )
    return prompt