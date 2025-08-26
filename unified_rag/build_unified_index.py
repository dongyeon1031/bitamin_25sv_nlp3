# unified_rag/build_unified_index.py

from rag.pdf_loader import load_all_pdfs as load_law_pdfs
from rag.chunker import make_chunks_from_sections as make_law_chunks
from finetune.security_rag.security_pdf_loader import load_all_security_data as load_security_pdfs
from finetune.security_rag.security_chunker import make_chunks_from_security_sections as make_security_chunks
from unified_rag.unified_indexer import build_and_persist_unified_index

def main():
    print("[UNIFIED RAG] 통합 인덱스 구축 시작...")
    
    # 1. 법령 문서 로드 및 청킹
    print("[UNIFIED RAG] 법령 문서 처리 중...")
    law_sections = load_law_pdfs()
    if not law_sections:
        raise FileNotFoundError("법령 문서를 로드할 수 없습니다.")
    print(f"[UNIFIED RAG] 법령 섹션 수: {len(law_sections)}")
    
    law_chunks = make_law_chunks(law_sections)
    print(f"[UNIFIED RAG] 법령 청크 수: {len(law_chunks)}")
    
    # 2. 보안/경제 문서 로드 및 청킹
    print("[UNIFIED RAG] 보안/경제 문서 처리 중...")
    security_sections = load_security_pdfs()
    if not security_sections:
        raise FileNotFoundError("보안/경제 문서를 로드할 수 없습니다.")
    print(f"[UNIFIED RAG] 보안/경제 섹션 수: {len(security_sections)}")
    
    security_chunks = make_security_chunks(security_sections)
    print(f"[UNIFIED RAG] 보안/경제 청크 수: {len(security_chunks)}")
    
    # 3. 모든 청크 통합
    all_chunks = law_chunks + security_chunks
    print(f"[UNIFIED RAG] 총 청크 수: {len(all_chunks)}")
    
    # 4. 통합 인덱스 구축
    print("[UNIFIED RAG] 통합 인덱스 구축 중...")
    build_and_persist_unified_index(all_chunks)
    
    print("[UNIFIED RAG] 통합 인덱스 구축 완료!")

if __name__ == "__main__":
    main()
