#pdf_processing.py
import re
from typing import List, Optional, Tuple
import fitz  # PyMuPDF

LAW_CENTER_LINE_PAT = re.compile(
    r"(?m)^\s*(?:법제처\s*\d+\s*국가법령정보센터|국가법령정보센터(?:\s*\d+)?|www\.law\.go\.kr)\s*$"
)
PAGE_NUMBER_LINE_PAT = re.compile(r"(?m)^\s*[-–—]?\s*\d+\s*[-–—]?\s*$")
PHONE_LINE_PAT = re.compile(r"(?m)^\s*.*\d{2,3}-\d{3,4}-\d{4}.*$")

AMEND_TAG_PAT_ANGLE = re.compile(r"<\s*(개정|전문개정|전부개정|신설|삭제)\b[^>]*>", re.UNICODE)
AMEND_TAG_PAT_BRACK = re.compile(
    r"[〈〈<\[]\s*.*?(개정|전문개정|전부개정|신설|삭제).*?[〉〉>\]]|[〔【]\s*.*?(개정|전문개정|전부개정|신설|삭제).*?[〕】]",
    re.UNICODE | re.IGNORECASE,
)

ARTICLE_HEADER_PAT = re.compile(
    r"(?m)^\s*(제\s*\d+(?:\s*조의\s*\d+|\s*조)\s*(?:\([^)]+\)|\s+[^\n()]+)?)\s*$",
    re.UNICODE,
)
CHAPTER_HEADER_PAT = re.compile(r"(?m)^\s*(제\s*\d+\s*장\s*[^\n]*)\s*$", re.UNICODE)

def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")

def _fix_hyphen_linebreak(text: str) -> str:
    return re.sub(r"-\s*\n\s*", "", text)

def _collapse_intra_paragraph_newlines(text: str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

def _enforce_header_newlines(text: str) -> str:
    text = re.sub(r"\s*(제\s*\d+\s*장[^\n]*)", r"\n\1\n", text)
    text = re.sub(
        r"\s*(제\s*\d+(?:\s*조의\s*\d+|\s*조)\s*(?:\([^)]+\)|\s+[^\n()]+)?)",
        r"\n\1\n",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def _standardize_article_header_spacing(text: str) -> str:
    text = re.sub(r"제\s*(\d+)\s*조\s*의\s*(\d+)", r"제\1조의\2", text)
    text = re.sub(r"제\s*(\d+)\s*조\s*\(\s*([^)]+?)\s*\)", r"제\1조(\2)", text)
    return text

def _strip_header_footer_lines(text: str) -> str:
    text = LAW_CENTER_LINE_PAT.sub("", text)
    text = PAGE_NUMBER_LINE_PAT.sub("", text)
    text = PHONE_LINE_PAT.sub("", text)
    return text

def _strip_amend_tags(text: str) -> str:
    text = AMEND_TAG_PAT_ANGLE.sub("", text)
    text = AMEND_TAG_PAT_BRACK.sub("", text)
    return text

def clean_page_text(text: str) -> str:
    text = _normalize_line_endings(text)
    text = _strip_header_footer_lines(text)
    text = _strip_amend_tags(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def normalize_full_text(full_text: str) -> str:
    full_text = _normalize_line_endings(full_text)
    full_text = _fix_hyphen_linebreak(full_text)
    full_text = _collapse_intra_paragraph_newlines(full_text)
    full_text = _standardize_article_header_spacing(full_text)
    full_text = _enforce_header_newlines(full_text)
    full_text = full_text.replace("ㆍ", "·")
    full_text = re.sub(r"[ \t]+", " ", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    return full_text.strip()

def extract_text_with_pages(pdf_path: str) -> List[str]:
    pages: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return pages

def split_by_article(full_text: str) -> Tuple[str, List[Tuple[str, str]]]:
    parts = ARTICLE_HEADER_PAT.split(full_text)
    if len(parts) < 3:
        return "", []
    preface = parts[0].strip()
    articles: List[Tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        articles.append((header, body))
    return preface, articles

def parse_article_header(header_line: str) -> Tuple[Optional[str], str]:
    h = re.sub(r"\s+", "", header_line)
    m = re.match(r"제(\d+)(?:조의(\d+)|조)", h)
    art_no = None
    if m:
        art_no = f"{m.group(1)}의{m.group(2)}" if m.group(2) else m.group(1)
    t = ""
    tm = re.search(r"\(([^)]+)\)", header_line)
    if tm:
        t = tm.group(1).strip()
    else:
        m2 = re.search(r"(?:제\s*\d+(?:\s*조의\s*\d+|\s*조))\s+([^\n()]+)", header_line)
        if m2:
            t = m2.group(1).strip()
    return art_no, t
