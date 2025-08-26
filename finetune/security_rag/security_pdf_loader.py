import os
import re
import pandas as pd
from typing import List, Dict
from finetune.security_configs.security_rag_config import SECURITY_DATA_LIST
from index.pdf_processing import (
    extract_text_with_pages, clean_page_text, normalize_full_text
)

def clean_security_text(text: str) -> str:
    """보안 문서 텍스트 정리 - 그림/표 관련 텍스트 제거"""
    import re
    
    # 그림/표 관련 텍스트 제거
    text = re.sub(r'\[그림\s*\d+\].*?(?=\n|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'\[표\s*\d+\].*?(?=\n|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'<그림\s*\d+>.*?(?=\n|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'<표\s*\d+>.*?(?=\n|$)', '', text, flags=re.DOTALL)
    
    # 페이지 번호 제거
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # 연속된 공백 정리
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def split_security_sections(text: str) -> List[Dict]:
    """보안 문서를 섹션별로 분할"""
    sections = []
    
    # 제목 패턴들 (보안 문서에 맞게 조정)
    title_patterns = [
        r'^(\d+\.\s*[가-힣\s]+)$',  # 1. 제목
        r'^(\d+\.\d+\s*[가-힣\s]+)$',  # 1.1 제목
        r'^([가-힣\s]+)$',  # 제목만 있는 경우
    ]
    
    lines = text.split('\n')
    current_section = {"title": "", "content": []}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 제목인지 확인
        is_title = any(re.match(pattern, line) for pattern in title_patterns)
        
        if is_title and current_section["content"]:
            # 이전 섹션 저장
            if current_section["content"]:
                sections.append({
                    "title": current_section["title"],
                    "content": "\n".join(current_section["content"])
                })
            current_section = {"title": line, "content": []}
        else:
            current_section["content"].append(line)
    
    # 마지막 섹션 저장
    if current_section["content"]:
        sections.append({
            "title": current_section["title"],
            "content": "\n".join(current_section["content"])
        })
    
    return sections

def load_security_pdfs() -> List[Dict]:
    """보안/경제 PDF 파일들을 로드하고 섹션별로 분할"""
    sections = []
    
    for pdf_path, doc_name, doc_type in SECURITY_DATA_LIST:
        if not os.path.exists(pdf_path):
            print(f"[WARN] 파일을 찾을 수 없습니다: {pdf_path}")
            continue
            
        print(f"[INFO] 로딩 중: {doc_name}")
        
        try:
            # PDF 텍스트 추출
            pages = extract_text_with_pages(pdf_path)
            pages = [clean_page_text(p) for p in pages]
            full_text = normalize_full_text("\n".join(pages))
            
            # 보안 문서 특화 정리
            full_text = clean_security_text(full_text)
            
            # 섹션별로 분할
            doc_sections = split_security_sections(full_text)
            
            for i, section in enumerate(doc_sections):
                sections.append({
                    "source": os.path.basename(pdf_path),
                    "doc_name": doc_name,
                    "doc_type": doc_type,
                    "section_id": f"{doc_name}_{i+1}",
                    "section_type": "보안문서",
                    "title": section["title"],
                    "parent_text": section["content"],
                })
                
        except Exception as e:
            print(f"[ERROR] {pdf_path} 처리 중 오류: {e}")
            continue
    
    return sections

def load_excel_terms() -> List[Dict]:
    """Excel 용어사전 파일 로드"""
    sections = []
    
    # DATA_DIR 정의
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    
    excel_path = os.path.join(DATA_DIR, "20250815_시사경제용어사전.xlsx")
    
    if not os.path.exists(excel_path):
        print(f"[WARN] Excel 파일을 찾을 수 없습니다: {excel_path}")
        return sections
    
    try:
        # Excel 파일 읽기
        df = pd.read_excel(excel_path)
        
        # 컬럼명 확인 및 처리
        print(f"[INFO] Excel 컬럼: {df.columns.tolist()}")
        
        for idx, row in df.iterrows():
            # 용어와 정의 추출 (컬럼명에 따라 조정 필요)
            term = str(row.iloc[0]) if len(row) > 0 else ""
            definition = str(row.iloc[1]) if len(row) > 1 else ""
            
            if term and definition and term != "nan" and definition != "nan":
                sections.append({
                    "source": "20250815_시사경제용어사전.xlsx",
                    "doc_name": "시사경제용어사전",
                    "doc_type": "용어사전",
                    "section_id": f"용어사전_{idx+1}",
                    "section_type": "용어정의",
                    "title": term,
                    "parent_text": f"{term}: {definition}",
                })
                
    except Exception as e:
        print(f"[ERROR] Excel 파일 처리 중 오류: {e}")
    
    return sections

def load_all_security_data() -> List[Dict]:
    """모든 보안/경제 데이터 로드"""
    print("[RAG] 보안/경제 데이터 로딩 중...")
    
    # PDF 파일들 로드
    pdf_sections = load_security_pdfs()
    
    # Excel 파일 로드
    excel_sections = load_excel_terms()
    
    all_sections = pdf_sections + excel_sections
    print(f"[RAG] 총 {len(all_sections)}개 섹션 로드 완료")
    
    return all_sections
