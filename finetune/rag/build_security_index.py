# finetune/rag/build_security_index.py

from finetune.rag.security_pdf_loader import load_all_security_data
from finetune.rag.security_chunker import make_chunks_from_security_sections as make_chunks
from finetune.rag.security_indexer import build_and_persist_security_index

def main():
    print("[보안 RAG] 보안/경제 데이터 로딩 중...")
    sections = load_all_security_data()  
    if not sections:
        raise FileNotFoundError("데이터를 로드할 수 없습니다. data/ 폴더를 확인해주세요.")
    print(f"[보안 RAG] 섹션 수: {len(sections)}")

    print("[보안 RAG] 청킹 중...")
    chunks = make_chunks(sections)
    print(f"[보안 RAG] 청크 수: {len(chunks)}")

    print("[보안 RAG] BM25 + Chroma + Docstore 구축 중...")
    build_and_persist_security_index(chunks)
    print("[보안 RAG] 완료. 인덱스가 ./finetune/indexes/에 저장되었습니다.")

if __name__ == "__main__":
    main()
