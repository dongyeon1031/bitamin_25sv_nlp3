# rag/pdf_loader.py
import os
from typing import List, Dict
from configs.rag_config import PDF_LIST
from index.pdf_processing import (
    extract_text_with_pages, clean_page_text, normalize_full_text,
    split_by_article, parse_article_header
)

def load_all_pdfs() -> List[Dict]:
    sections = []
    for pdf_path, law_name, kind in PDF_LIST:
        if not os.path.exists(pdf_path):
            print(f"[WARN] PDF not found: {pdf_path}")
            continue

        pages = extract_text_with_pages(pdf_path)
        pages = [clean_page_text(p) for p in pages]
        full = normalize_full_text("\n".join(pages))

        # 본문: 조 단위
        preface, articles = split_by_article(full)
        if not articles:
            sections.append({
                "source": os.path.basename(pdf_path),
                "law_name": law_name or "",
                "kind": kind or "",
                "section_type": "본문",
                "article_no": "",
                "para_no": "",
                "item_no": "",
                "title": "",
                "parent_text": full or "",
            })
            continue

        for header, body in articles:
            art_no, title = parse_article_header(header)
            art_no = f"{law_name}_{art_no or 'unknown'}"  # unique key

            parent_text = (header + "\n" + body).strip()
            sections.append({
                "source": os.path.basename(pdf_path),
                "law_name": law_name or "",
                "kind": kind or "",
                "section_type": "본문",
                "article_no": art_no,
                "para_no": "",
                "item_no": "",
                "title": title or "",
                "parent_text": parent_text or "",
            })
    return sections


def load_pdfs():
    return load_all_pdfs()
