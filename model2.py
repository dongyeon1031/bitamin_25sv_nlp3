from huggingface_hub import snapshot_download

local_dir = "models/claude3-gemma"

snapshot_download(
    repo_id="reedmayhew/claude-3.7-sonnet-reasoning-gemma3-12B",  # 모델 저장소
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    allow_patterns=["*.gguf"]  # gguf 파일만 받기
)

print("✅ 다운로드 완료:", local_dir)
