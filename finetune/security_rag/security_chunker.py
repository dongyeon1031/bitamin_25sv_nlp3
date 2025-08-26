import os, re, hashlib, uuid
from typing import List, Dict, Tuple
from finetune.security_configs.security_rag_config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS, MAX_SECTION_SIZE_CHARS

def _sliding_chunks(text: str, size: int, overlap: int) -> List[Tuple[str, int]]:
    """슬라이딩 윈도우 방식으로 텍스트를 청크로 분할"""
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

def make_chunks_from_security_sections(sections: List[Dict]) -> List[Dict]:
    """보안/경제 문서 섹션들을 청크로 분할"""
    out = []
    
    for sec in sections:
        ptxt = sec["parent_text"][:MAX_SECTION_SIZE_CHARS]  

        # 문서 유형에 따른 청크 크기 조정
        size = CHUNK_SIZE_CHARS
        overlap = CHUNK_OVERLAP_CHARS
        
        if sec.get("doc_type") == "용어사전":
            # 용어사전은 더 작은 청크로
            size = int(CHUNK_SIZE_CHARS * 0.7)
            overlap = int(CHUNK_OVERLAP_CHARS * 0.7)
        elif sec.get("doc_type") == "보고서":
            # 보고서는 더 큰 청크로
            size = int(CHUNK_SIZE_CHARS * 1.2)
            overlap = int(CHUNK_OVERLAP_CHARS * 1.2)

        for chunk, offset in _sliding_chunks(ptxt, size, overlap):
            cid = uuid.uuid4().hex  
            out.append({
                "id": f"security_chunk_{cid}",
                "text": chunk,
                "parent_hash": hashlib.md5(ptxt.encode("utf-8")).hexdigest(),
                "meta": {
                    "parent_id": sec.get("section_id") or f"{sec['source']}::{sec.get('title','all')}",
                    "offset": offset,
                    "source": sec["source"],
                    "doc_name": sec["doc_name"],
                    "doc_type": sec["doc_type"],
                    "section_type": sec.get("section_type", "보안문서"),
                    "section_id": sec.get("section_id"),
                    "title": sec.get("title", "")
                }
            })
    return out
