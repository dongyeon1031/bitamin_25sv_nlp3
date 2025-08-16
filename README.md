# Korean Laws RAG (JSONL -> ChromaDB)

This mini pipeline ingests your JSONL legal data into a Chroma vector DB using a multilingual SentenceTransformer, then queries it and optionally generates answers with OpenAI or Gemini.

## 1) Install

```bash
python -m venv .venv && source .venv/bin/activate   # or use your favorite environment
pip install -r requirements.txt
```

## 2) Ingest

```bash
python ingest_jsonl.py   --input ./laws/*.jsonl   --persist_dir ./db/chroma_laws   --collection laws-ko   --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Notes:
- The ingester makes a "head" chunk from `heading/lead` and additional chunks for each nested `metadata.items` entry.
- Overlong texts are split into ~1200-char chunks with small overlaps.
- Slightly broken JSON lines are skipped (and basic repair is attempted automatically).

## 3) Query (retrieve only)

```bash
python rag_query.py   --persist_dir ./db/chroma_laws   --collection laws-ko   --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2   --q "000의 정의가 뭐야?"   --k 5
```

## 4) Tips
- If you previously used a different collection name (e.g., `laws-ko`), keep it consistent across ingest and query.
- To re-ingest from scratch, delete the `--persist_dir` folder.
- You can switch to a larger multilingual embedding model later for quality improvements.
