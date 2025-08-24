from huggingface_hub import snapshot_download

# 모델 다운로드
model_dir = snapshot_download(
    repo_id="BAAI/bge-reranker-base",
    local_dir="./models/bge-reranker-base",
    local_dir_use_symlinks=False
)

print("모델 다운로드 완료:", model_dir)
