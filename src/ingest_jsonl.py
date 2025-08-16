"""
ingest_jsonl.py
- JSONL 법령 레코드를 읽어서 heading/lead + items를 청크로 분해
- item 트리는 그대로 평탄화하되, marker_path는 메타데이터/ID에서 완전히 제거
- SentenceTransformers 임베딩으로 ChromaDB에 영구 저장

실행:
  python ingest_jsonl.py \
    --input ../data/jsonl/*.jsonl \
    --persist_dir ../db/chroma_laws \
    --collection laws-ko \
    --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"""

import argparse, json, re, sys, glob, os, uuid
from typing import Dict, Any, List, Tuple, Iterable
from dataclasses import dataclass
from tqdm import tqdm

import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer

# ----------------------------- Embedding Function -----------------------------
class STEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

# ----------------------------- Helpers ---------------------------------------
def coalesce(*vals):
    for v in vals:
        if v not in (None, "", []):
            return v
    return None

def strip_ws(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def extract_json(line: str) -> Dict[str, Any]:
    """느슨한 JSONL 파서 (깨진 라인에 대해 {} 경계 복구 시도)"""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        start = line.find('{')
        end = line.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(line[start:end+1])
            except Exception:
                pass
        raise

@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]

def flatten_items_texts(items: List[Dict[str, Any]]) -> Iterable[str]:
    """items 트리를 텍스트 스트림으로 평탄화 (marker_path 제외)"""
    if not items:
        return
    for it in items:
        text = strip_ws(it.get("text", ""))
        if text:
            yield text
        for ch in it.get("children", []) or []:
            yield from flatten_items_texts([ch])

def chunk_long_text(text: str, max_chars: int = 1200, overlap: int = 120) -> List[str]:
    t = text.strip()
    if len(t) <= max_chars:
        return [t]
    chunks = []
    start = 0
    while start < len(t):
        end = min(len(t), start + max_chars)
        chunk = t[start:end]
        chunks.append(chunk)
        if end == len(t):
            break
        start = max(0, end - overlap)
    return chunks

def infer_doc_type(doc_title: str) -> str:
    """간단한 문서 타입 추론: 법 / 시행령 / 시행규칙 / 기타"""
    if not doc_title:
        return "unknown"
    if "시행규칙" in doc_title:
        return "rule"
    if "시행령" in doc_title:
        return "decree"
    if "법" in doc_title:
        return "act"
    return "other"

def build_chunks(record: Dict[str, Any], embed_model_name: str) -> List[Chunk]:
    """marker_path 없이 head + item 텍스트를 청크화"""
    rid = record.get("id") or f"rec-{uuid.uuid4()}"
    text_field = strip_ws(record.get("text", ""))
    md = record.get("metadata", {}) or {}

    heading = strip_ws(coalesce(md.get("heading"), md.get("article_title"), text_field, ""))
    lead = strip_ws(md.get("lead", "") or "")
    doc_title = md.get("doc_title", "")
    lang = md.get("lang", "")
    article_title = md.get("article_title", "")
    chunk_index = md.get("chunk_index")

    doc_type = infer_doc_type(doc_title)

    # Head chunk
    head_text_parts = []
    if heading:
        head_text_parts.append(heading)
    if lead:
        head_text_parts.append(lead)
    head_text = "\n\n".join(head_text_parts).strip() or text_field

    chunks: List[Chunk] = []
    if head_text:
        for i, sub in enumerate(chunk_long_text(head_text)):
            cid = f"{rid}::head::{i}"  # marker 없음
            chunks.append(Chunk(
                id=cid,
                text=sub,
                metadata={
                    "section": "head",
                    "heading": heading,
                    "article_title": article_title,
                    "doc_title": doc_title,
                    "doc_type": doc_type,
                    "lang": lang,
                    "source_id": rid,
                    "source_chunk_index": chunk_index,
                    "embed_model": embed_model_name,
                }
            ))

    # Item chunks (marker 제거)
    items = md.get("items", []) or []
    item_texts = list(flatten_items_texts(items))
    for j, item_text in enumerate(item_texts):
        for i, sub in enumerate(chunk_long_text(item_text)):
            cid = f"{rid}::item::{j}::{i}"
            chunks.append(Chunk(
                id=cid,
                text=sub,
                metadata={
                    "section": "item",
                    "heading": heading,
                    "article_title": article_title,
                    "doc_title": doc_title,
                    "doc_type": doc_type,
                    "lang": lang,
                    "source_id": rid,
                    "source_chunk_index": chunk_index,
                    "embed_model": embed_model_name,
                }
            ))
    return chunks

def batched(iterable, n: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

# ----------------------------- Main ------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="JSONL file(s) or globs (e.g., ./data/*.jsonl)")
    ap.add_argument("--persist_dir", required=True, help="Directory to persist Chroma DB")
    ap.add_argument("--collection", default="laws-ko", help="Collection name")
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="SentenceTransformer model")
    ap.add_argument("--batch_size", type=int, default=256, help="Batch size for Chroma add")
    args = ap.parse_args()

    files = []
    for p in args.input:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    if not files:
        print("No input files matched.", file=sys.stderr)
        sys.exit(1)

    client = chromadb.PersistentClient(path=args.persist_dir)
    embed_fn = STEmbeddingFunction(args.model)
    try:
        col = client.get_or_create_collection(name=args.collection, embedding_function=embed_fn)
    except TypeError:
        col = client.get_or_create_collection(name=args.collection)

    total_lines = 0
    total_chunks = 0
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Ingesting {os.path.basename(fp)}"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = extract_json(line)
                except Exception:
                    continue
                total_lines += 1
                chunks = build_chunks(rec, embed_model_name=args.model)
                total_chunks += len(chunks)

                for batch in batched(chunks, args.batch_size):
                    ids = [c.id for c in batch]
                    docs = [c.text for c in batch]
                    metas = [c.metadata for c in batch]
                    try:
                        if hasattr(col, "_embedding_function") and col._embedding_function:
                            col.add(ids=ids, documents=docs, metadatas=metas)
                        else:
                            embeddings = embed_fn(docs)
                            col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
                    except Exception:
                        continue

    print(f"Ingested lines: {total_lines}, chunks stored: {total_chunks}")
    print(f"Persisted at: {args.persist_dir}  (collection: {args.collection})")

if __name__ == "__main__":
    main()
