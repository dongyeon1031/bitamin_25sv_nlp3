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

# 프롬프트 생성기
def make_prompt_auto(text):
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = (
                "당신은 금융보안 전문가입니다.\n"
                "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
                f"질문: {question}\n"
                "선택지:\n"
                f"{chr(10).join(options)}\n\n"
                "답변:"
                )
    else:
        prompt = (
                "당신은 금융보안 전문가입니다.\n"
                "모든 응답은 한국어로 작성하세요.\n"
                "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n\n"
                f"질문: {text}\n\n"
                "답변:"
                )   
    return prompt