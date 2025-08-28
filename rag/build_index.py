# rag/build_index.py

from rag.pdf_loader import load_all_pdfs
from rag.chunker import make_chunks_from_sections as make_chunks
from rag.indexer import build_and_persist

def main():
    print("[RAG] Loading PDFs & parsing articles...")
    sections = load_all_pdfs()  
    if not sections:
        raise FileNotFoundError("No sections loaded. Check configs/rag_config.PDF_LIST vs data/ folder.")
    print(f"[RAG] Sections(articles): {len(sections)}")

    print("[RAG] Chunking...")
    chunks = make_chunks(sections)
    print(f"[RAG] Chunks: {len(chunks)}")

    print("[RAG] Building BM25 + Chroma + Docstore...")
    build_and_persist(chunks)
    print("[RAG] Done. Indexes saved to ./indexes/")

if __name__ == "__main__":
    main()
