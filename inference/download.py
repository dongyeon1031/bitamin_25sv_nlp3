import os
from huggingface_hub import snapshot_download

def download_model():
    model_dir = "./models"
    if not os.path.exists(model_dir):
        snapshot_download(
            repo_id="LGAI-EXAONE/EXAONE-4.0-32B-GGUF",
            local_dir=model_dir,
            local_dir_use_symlinks=False,  # 하드카피로 저장
            allow_patterns=[
                "*Q6_K.gguf",
                "chat_template.jinja",
                "assets/*"
            ]  # Q6_K 모델, chat_template.jinja, assets 디렉토리 전체 다운로드
        )
