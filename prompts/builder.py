#propts/builder.py
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


# 프롬프트 생성기 (기본)
def make_prompt_auto(text):
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = (
            "당신은 금융보안 전문가입니다.\n"
            "아래 질문에 대해 **제공된 선택지 중 정답 번호 하나**만 출력하세요.\n"
            "중요: 숫자 하나만 출력하세요. 다른 설명, 기호, 문장은 절대 출력하지 마세요.\n\n"
            f"질문: {question}\n"
            "선택지:\n"
            f"{chr(10).join(options)}\n\n"
            "답변:"
        )
    else:
        prompt = (
            "당신은 금융보안 전문가입니다.\n"
            "모든 응답은 한국어로 작성하세요.\n"
            "아래 주관식 질문에 대해 핵심 키워드 중심으로 정확하고 간략한 설명을 작성하세요\n"
            "중요: 답변은 반드시 한 문장으로 작성하고, 불필요한 서론/결론은 생략하세요.\n"
            "답변 이외의 부가적인 인사말은 생략하세요.\n\n"
            f"질문: {text}\n\n"
            "답변:"
        )   
    return prompt


def make_prompt_with_context(text: str, contexts: list):
    ctx_block = "\n\n".join([
        f"- 출처:{c.get('meta',{}).get('source','?')} "
        f"조문:{c.get('meta',{}).get('article_no','?')}\n{c['text'][:500]}"
        for c in contexts[:2]
    ])

    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = (
            "당신은 금융보안 및 법규 전문가입니다.\n"
            "아래 제공된 문맥(Context)만 근거로 사용하여 **정답 선택지 번호 하나만 출력**하세요.\n"
            "정답은 반드시 제공된 선택지 번호 중 하나의 숫자만 출력합니다.\n"
            "추론 과정, 설명, 다른 텍스트는 절대 출력하지 마세요.\n\n"
            f"[Context]\n{ctx_block}\n\n"
            f"질문: {question}\n"
            "선택지:\n"
            f"{chr(10).join(options)}\n\n"
            "답변:"
        )
    else:
        prompt = (
            "당신은 금융보안 및 법규 전문가입니다.\n"
            "아래 제공된 문맥(Context)만 근거로 사용하여 "
             "아래 주관식 질문에 대해 핵심 키워드 중심으로 정확하고 간략한 설명을 작성하세요\n"
            "중요: 답변은 반드시 한 문장으로 작성하고, 불필요한 서론/결론은 생략하세요.\n"
            "답변 이외의 부가적인 인사말은 생략하세요.\n\n"
            f"[Context]\n{ctx_block}\n\n"
            f"질문: {text}\n\n"
            "답변:"
        )
    return prompt