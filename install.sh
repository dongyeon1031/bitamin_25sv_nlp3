#!/bin/bash

# pip 및 requirements 설치
pip install --upgrade pip
pip install -r requirements.txt

# llama-cpp-python 설치 제거
# CMAKE_ARGS="-DLLAMA_CUBLAS=OFF -DGGML_CUDA=ON" \
# pip install llama-cpp-python[cuda12]==0.3.16 --no-cache-dir --force-reinstall --upgrade

# ollama 설치 (리눅스 전용)
echo "[*] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# ollama 서버 실행 (백그라운드)
echo "[*] Starting Ollama daemon..."
ollama serve &

# 모델 다운로드 (예: DavidAU의 Qwen3-128k 30B 모델)
echo "[*] Pulling model..."
ollama pull davidau/qwen3-128k-30b-a3b-neo-max-ultra:Q6_K

echo "[*] All done !"