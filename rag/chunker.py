import hashlib
from typing import List, Dict, Tuple
from configs.rag_config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS
import uuid

def _sliding_chunks(text: str) -> List[Tuple[str, int]]:
    chunks = []
    n = len(text)
    start = 0
    while start < n:
        end = min(n, start + CHUNK_SIZE_CHARS)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start))
        if end == n:
            break
        start = max(end - CHUNK_OVERLAP_CHARS, start + 1)
    return chunks



def make_chunks_from_sections(sections: List[Dict]) -> List[Dict]:
    out = []
    for sec in sections:
        ptxt = sec["parent_text"][:MAX_SECTION_SIZE_CHARS]
        for chunk, offset in _sliding_chunks(ptxt):
            out.append({
                "id": f"chunk_{uuid.uuid4().hex}", 
                "text": chunk,
                "parent_text": ptxt,
                "meta": {
                    "parent_id": f"{sec['source']}::제{sec['article_no']}조" if sec.get("article_no") else f"{sec['source']}::all",
                    "offset": offset,
                    "source": sec["source"],
                    "law_name": sec["law_name"],
                    "kind": sec["kind"],
                    "article_no": sec["article_no"],
                    "title": sec.get("title","")
                }
            })
    return out
