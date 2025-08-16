# extract.py
import re
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
import fitz  # PyMuPDF

# =============== 공통 유틸 ===============
# ①~⑳ → 1~20 정규화 (필요시 확장)
CIRCLED_MAP = {ord(k): str(i) for i, k in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳", start=1)}

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r", "")

    s = s.translate(CIRCLED_MAP)
    s = re.sub(r"<[^>]*>", "", s, flags=re.DOTALL)
    s = re.sub(r"\[[^\]]*\]", "", s, flags=re.DOTALL)
    s = re.sub(r"^\s*법제처.*국가법령정보센터.*$", "", s, flags=re.MULTILINE)
    s = re.sub(r'([가-힣])\n(다[.)])', r'\1\2', s)
    s = re.sub(r'(\d+)\s*의\s*\n\s*(\d+)([.)])', r'\1의\2\3', s)
    s = re.sub(r'의\s*\n\s*(\d+)', r'의\1', s)
    s = re.sub(r'(\d+)\s*\n\s*([.)])', r'\1\2', s)
    s = re.sub(
        r'([가-힣])\n(?!\s*(?:[가-힣]\.|[가-힣]\)|\([가-힣]\)))'  # 다음 줄이 마커면 합치지 않음
        r'([가-힣])',
        r'\1\2',
        s
    )

    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

# =============== 조문(제N조) 추출 ===============
ARTICLE_RE = re.compile(
    r'^\s*(?P<heading>(?P<title>제\s*\d+\s*조(?:\s*의\s*\d+)?)\s*\([^)]+\))'   # 라인 시작 + 괄호 표제 필수
    r'\s*(?P<body>.*?)(?=^\s*(?:제\s*\d+\s*조(?:\s*의\s*\d+)?\s*\([^)]+\))\s*|$\Z)',
    re.DOTALL | re.MULTILINE
)

def extract_articles_from_pdf(pdf_path: str):
    """PDF 전체에서 '제N조(…)' 단위 블록을 추출"""
    doc = fitz.open(pdf_path)
    pages = [p.get_text() for p in doc]
    doc.close()
    text = "\n".join(pages)

    articles = []
    for m in ARTICLE_RE.finditer(text):
        heading = clean_text(m.group("heading"))               # 예: '제3조(영상정보처리기기의 범위)'
        title   = clean_text(m.group("title")).replace(" ", "")# 예: '제3조', '제3조의2'
        body    = clean_text(m.group("body") or "")
        articles.append({"title": title, "heading": heading, "body": body})
    return articles

# =============== 패턴 정의 (계층 파서) ===============
P0_PATTERNS = [
    r'\s*\((\d+)\)\s+',
    r'\s*(\d+)\)\s+',
    r'\s*(\d+)\s+(?![.\)])',
]

L1_PATTERNS = [
    r'\s*((?:\d+의\d+|\d+))\.\s+',   # 1의2.  또는  1.
    r'\s*\(((?:\d+의\d+|\d+))\)\s+', # (1의2)  또는  (1)
    r'\s*((?:\d+의\d+|\d+))\)\s+',   # 1의2)  또는  1)
]

_ENUM_KR = "가나다라마바사아자차카타파하"
L2_PATTERNS = [
    rf'\s*([{_ENUM_KR}])\.\s+',
    rf'\s*\(([{_ENUM_KR}])\)\s+',
    rf'\s*([{_ENUM_KR}])\)\s+',
]

def _split_by_patterns(text: str, patterns):
    """패턴 중 하나로 '라인 시작'에서 매치되는 구간을 (marker, text)로 분할"""
    if not text.strip():
        return []
    union = "|".join(f"(?:{p})" for p in patterns)
    combined = re.compile(rf"(?m)^(?:{union})")

    parts = []
    last_idx = 0
    for m in combined.finditer(text):
        start = m.start()
        if start != last_idx:
            if not parts:
                head = text[last_idx:start]
                if head.strip():
                    parts.append({"marker": None, "text": head})
            else:
                parts[-1]["text"] += text[last_idx:start]
        marker = m.group(0)
        parts.append({"marker": marker, "text": ""})
        last_idx = m.end()

    tail = text[last_idx:]
    if parts:
        parts[-1]["text"] += tail
    else:
        parts.append({"marker": None, "text": text})

    for p in parts:
        p["text"] = clean_text(p["text"])
        if p["marker"] is not None:
            p["marker"] = clean_text(p["marker"])
    return [p for p in parts if p["text"]]


def _parse_level(text: str, level_patterns, next_level_patterns=None):
    blocks = _split_by_patterns(text, level_patterns)
    items = []
    for blk in blocks:
        if blk["marker"] is None:
            items.append({"marker": None, "text": blk["text"], "children": []})
            continue
        num = re.sub(r"[^\w가-힣]", "", blk["marker"]).strip()
        node = {"marker": num, "text": blk["text"], "children": []}
        if next_level_patterns:
            children = _parse_level(blk["text"], next_level_patterns, None)
            if children and children[0]["marker"] is None:
                node["text"] = children[0]["text"]
                node["children"] = children[1:]
            else:
                node["children"] = children
        items.append(node)
    return items


# 복합번호(예: 1의2)를 직전 부모(예: 1)의 children으로 편입
def _nest_compound_markers(items):
    """
    items: [{'marker':'1', ...}, {'marker':'1의2', ...}, {'marker':'2', ...}, ...]
    -> '1의2'를 '1'의 children로 이동
    """
    new = []
    last_parent = None
    for it in items:
        mk = it.get("marker") or ""
        m = re.fullmatch(r'(\d+)의(\d+)', mk)
        if m and last_parent and last_parent.get("marker") == m.group(1):
            last_parent.setdefault("children", []).append(it)
        else:
            new.append(it)
            if re.fullmatch(r'\d+', mk):
                last_parent = new[-1]
    return new

"""
본문을 계층형으로 파싱.
1) 먼저 P0(① ② …) 단락으로 분해
2) 각 단락 내부는 L1(1. (1) 1)) → L2(가. (가) 가)) 로 재귀 파싱
P0가 없으면 L1/L2만 사용(기존 문서 호환)
"""
def parse_items_hierarchy(body: str):
    p0 = _parse_level(body, P0_PATTERNS, None)
    has_p0 = any(b["marker"] is not None for b in p0)

    if has_p0:
        lead = ""
        items = []
        if p0 and p0[0]["marker"] is None:
            lead = p0[0]["text"]
            p0_blocks = p0[1:]
        else:
            p0_blocks = p0

        for blk in p0_blocks:
            child_items = _parse_level(blk["text"], L1_PATTERNS, L2_PATTERNS)

            # child_items[0]이 lead(None)이면 본문으로 승격
            para_text = ""
            children = child_items
            if child_items and child_items[0]["marker"] is None:
                para_text = child_items[0]["text"]
                children = child_items[1:]
            children = _nest_compound_markers(children)

            items.append({
                "marker": blk["marker"],   # '1','2' …
                "text": para_text,         # 단락 설명(예: '법 제2조제7호에서 …')
                "children": children
            })
        return clean_text(lead), items

    # P0이 없으면: 기존 방식(L1 → L2)
    level1 = _parse_level(body, L1_PATTERNS, L2_PATTERNS)
    lead = ""
    items = []
    if level1 and level1[0]["marker"] is None:
        lead = level1[0]["text"]
        items = level1[1:]
    else:
        items = level1

    items = _nest_compound_markers(items)
    return clean_text(lead), items


# =============== JSONL 저장 ===============
"""
한 레코드 = 한 조문
text = 헤딩(예: '제3조(영상정보처리기기의 범위)')
metadata.lead = 머리말
metadata.items = P0(단락) → L1(호) → L2(목)
"""
def save_as_jsonl(articles, source_path, out_path, keep_raw=False):
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    now = datetime.utcnow().isoformat() + "Z"
    doc_title = os.path.basename(source_path)

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, a in enumerate(articles, 1):
            lead, items = parse_items_hierarchy(a["body"])
            rec = {
                "id": f"{Path(source_path).stem}#{a['title']}-{uuid.uuid4().hex[:8]}",
                "text": a["heading"],
                "metadata": {
                    "article_title": a["title"],
                    "heading": a["heading"],
                    "lead": lead,
                    "items": items,
                    # "source": os.path.abspath(source_path),   # ← 요청: 제거
                    "doc_title": doc_title,
                    "lang": "ko",
                    "created_at": now,
                    "chunk_index": idx
                }
            }
            if keep_raw:
                rec["metadata"]["raw"] = a["body"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[OK] JSONL 저장: {out_path}")