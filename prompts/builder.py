#propts/builder.py
import re
from utils.classify import is_multiple_choice, OPTION_PREFIX

# 질문과 선택지 분리 함수
_CIRCLED_MAP = {c: str(i) for i, c in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳", start=1)}

def extract_question_and_choices(full_text: str):
    """
    전체 질문에서 본문과 선택지를 분리하고,
    선택지는 '숫자 공백 내용' 형태로 표준화해서 반환.
    """
    lines = [ln.rstrip() for ln in full_text.strip().split("\n") if ln.strip()]
    q_lines, options = [], []

    for line in lines:
        m = OPTION_PREFIX.match(line)
        if not m:
            q_lines.append(line.strip())
            continue
        # 번호 추출
        num = None
        # ① 같은 경우
        circ = re.match(r"^\s*([①-⑳])", line)
        if circ:
            num = _CIRCLED_MAP.get(circ.group(1))
        if num is None:
            dig = re.match(r"^\s*([1-9][0-9]?)", line)
            if dig:
                num = dig.group(1)
        # 본문 텍스트 정리 (선두 번호/기호/번 제거)
        body = re.sub(r"^\s*[1-9][0-9]?\s*(?:번|[).:\-])\s*", "", line)
        body = re.sub(r"^\s*[①-⑳]\s*", "", body).strip()
        # 최종 표준화: "1 내용"
        if num:
            options.append(f"{num} {body}")
        else:
            # 혹시 실패하면 원문을 그대로 옵션에 담되, 뒤 파이프라인이 fallback
            options.append(line.strip())

    question = " ".join(q_lines).strip()
    return question, options

# 프롬프트 생성기 (기본)
def make_prompt_auto(text):
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = (
                "당신은 금융보안 전문가입니다.\n"
                "아래 질문에 대해 **제공된 선택지 중 정답 번호 하나**만 출력하세요.\n"
+               "정답은 반드시 숫자 하나만 출력합니다(예: 3).\n\n"
                f"질문: {question}\n"
                "선택지:\n"
                f"{chr(10).join(options)}\n\n"
                "답변:"
                )
    else:
        prompt = (
                "당신은 금융보안 전문가입니다.\n"
                "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n\n"
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
                "당신은 금융보안 전문가입니다.\n"
                "아래 질문에 대해 **제공된 선택지 중 정답 번호 하나**만 출력하세요.\n\n"
                f"질문: {question}\n"
                "선택지:\n"
                f"{chr(10).join(options)}\n\n"
                "답변:"
                )
    else:
        prompt = (
                "당신은 금융보안 전문가입니다.\n"
                "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n\n"
                f"질문: {text}\n\n"
                "답변:"
                )
    return prompt