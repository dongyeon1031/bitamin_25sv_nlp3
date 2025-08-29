#!/bin/bash

# pip 및 requirements 설치
pip install --upgrade pip
pip install -r requirements.txt

# llama-cpp-python 설치
# CMAKE_ARGS="-DLLAMA_CUBLAS=OFF -DGGML_CUDA=ON" \
# pip install llama-cpp-python[cuda12]==0.3.16 --no-cache-dir --force-reinstall --upgrade

# ollama 설치 (리눅스 전용)
sudo apt update
sudo apt install -y pciutils lshw

echo "[*] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# ollama 서버 실행 (백그라운드)
echo "[*] Starting Ollama daemon..."
nohup ollama serve > ollama.log 2>&1 &

# 모델 디렉토리 생성 및 다운로드
echo "[*] Downloading EXAONE model..."
mkdir -p models
wget https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B-GGUF/resolve/main/EXAONE-4.0-32B-Q6_K.gguf \
     -O ./models/EXAONE-4.0-32B-Q6_K.gguf

# Modelfile 생성
echo "[*] Creating Modelfile..."
cat <<EOF > Modelfile
FROM ./models/EXAONE-4.0-32B-Q6_K.gguf
EOF

# Ollama 모델 등록
echo "[*] Registering model to Ollama..."
ollama create exaone-custom -f Modelfile

echo "[*] 끝~~"

echo "[*] Running main.py..."
python main.py