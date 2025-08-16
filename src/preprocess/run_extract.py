# run_extract.py
import argparse
from pathlib import Path
from extract import extract_articles_from_pdf, save_as_jsonl

def main():
    ap = argparse.ArgumentParser(description="법률 PDF → JSONL(RAG용) 추출기")
    ap.add_argument("--pdf", required=True, help="입력 PDF 경로")
    ap.add_argument("--outdir", default="output", help="출력 디렉터리")
    ap.add_argument("--keep-raw", action="store_true", help="원본문을 metadata.raw에 포함")
    args = ap.parse_args()

    arts = extract_articles_from_pdf(args.pdf)
    if not arts:
        print("[WARN] 조문이 감지되지 않았습니다.")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{Path(args.pdf).stem}.jsonl"
    save_as_jsonl(arts, args.pdf, str(out_path), keep_raw=args.keep_raw)

if __name__ == "__main__":
    main()