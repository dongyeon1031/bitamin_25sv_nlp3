import re

# 선택지 접두부 패턴: 1) / 1. / 1- / 1번 / "1 " / ①②③ …
OPTION_PREFIX = re.compile(
    r"^\s*(?:"
    r"[1-9][0-9]?\s*(?:번|[).:\-])\s+"   # 1번 / 1) / 1. / 1-
    r"|[1-9][0-9]?\s+"                  # 1<space>
    r"|[①-⑳]\s*"                       # ①~⑳
    r")"
)

def is_multiple_choice(question_text: str) -> bool:
    """
    줄 단위로 2개 이상 OPTION_PREFIX가 발견되면 객관식으로 간주
    """
    lines = question_text.strip().split("\n")
    option_count = sum(bool(OPTION_PREFIX.match(line)) for line in lines)
    return option_count >= 2