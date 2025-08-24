# -*- coding: utf-8 -*-
import os
from typing import List, Dict
from configs.rag_config import PDF_LIST
from index.pdf_processing import (
    extract_text_with_pages, clean_page_text, normalize_full_text,
    split_by_article, parse_article_header
)

def load_all_pdfs() -> List[Dict]:
    """
    PDF_LIST에 있는 모든 PDF를 읽어 조문 단위 섹션으로 분리.
    반환: [{"source":..,"law_name":..,"kind":..,"article_no":..,"title":..,"parent_text":..}, ...]
    """
    sections = []
    for pdf_path, law_name, kind in PDF_LIST:
        if not os.path.exists(pdf_path):
            print(f"[WARN] PDF not found: {pdf_path}")
            continue
        pages = extract_text_with_pages(pdf_path)
        pages = [clean_page_text(p) for p in pages]
        full = normalize_full_text("\n".join(pages))
        _, articles = split_by_article(full)
        if not articles:
            # 조문 분리가 안 되면 통짜 섹션
            sections.append({
                "source": os.path.basename(pdf_path),
                "law_name": law_name,
                "kind": kind,
                "article_no": None,
                "title": "",
                "parent_text": full
            })
            continue

        for header, body in articles:
            art_no, title = parse_article_header(header)
            parent_text = (header + "\n" + body).strip()
            sections.append({
                "source": os.path.basename(pdf_path),
                "law_name": law_name,
                "kind": kind,
                "article_no": art_no,
                "title": title,
                "parent_text": parent_text
            })
    return sections

def load_pdfs():
    """Backward compatibility alias."""
    return load_all_pdfs()