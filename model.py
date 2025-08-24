from sentence_transformers import SentenceTransformer

# 다운로드할 로컬 경로 지정
save_path = "model/bge-m3"

# 모델 다운로드
print(f"'{save_path}' 경로에 BAAI/bge-m3 모델 다운로드를 시작합니다.")
model = SentenceTransformer("BAAI/bge-m3")
model.save_pretrained(save_path)

print("\n모델 다운로드가 완료되었습니다.")
print(f"모델 파일이 '{save_path}'에 저장되었습니다.")