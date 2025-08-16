import json
import sys

in_path = sys.argv[1] if len(sys.argv) > 1 else "input.jsonl"
out_path = sys.argv[2] if len(sys.argv) > 2 else "output.json"

with open(in_path, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print(f"변환 완료: {out_path}")