import os, re, hashlib, uuid
from typing import List, Dict, Tuple
from configs.rag_config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS, MAX_SECTION_SIZE_CHARS

def _sliding_chunks(text: str, size: int, overlap: int) -> List[Tuple[str, int]]:
    chunks = []
    n = len(text)
    start = 0
    while start < n:
        end = min(n, start + size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start))
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def make_chunks_from_sections(sections: List[Dict]) -> List[Dict]:
    out = []
    for sec in sections:
        ptxt = sec["parent_text"][:MAX_SECTION_SIZE_CHARS]  

        size = CHUNK_SIZE_CHARS
        overlap = CHUNK_OVERLAP_CHARS
        if sec.get("section_type") in ("별표", "부칙"):
            size = int(CHUNK_SIZE_CHARS * 1.5)  
            overlap = int(CHUNK_OVERLAP_CHARS * 1.2)

        for chunk, offset in _sliding_chunks(ptxt, size, overlap):
            cid = uuid.uuid4().hex  
            out.append({
                "id": f"chunk_{cid}",
                "text": chunk,
                "parent_hash": hashlib.md5(ptxt.encode("utf-8")).hexdigest(),
                "meta": {
                    "parent_id": sec.get("section_id") or f"{sec['source']}::{sec.get('article_no','all')}",
                    "offset": offset,
                    "source": sec["source"],
                    "law_name": sec["law_name"],
                    "kind": sec["kind"],
                    "section_type": sec.get("section_type", "본문"),
                    "article_no": sec.get("article_no"),
                    "para_no": sec.get("para_no"),
                    "item_no": sec.get("item_no"),
                    "title": sec.get("title", "")
                }
            })
    return out

