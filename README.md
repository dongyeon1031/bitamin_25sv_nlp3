# 실행 가이드

## 🖥️ 사전 요구사항
- Python 3.10 이상
- pip 최신 버전
- 가상환경 권장 (venv 또는 conda)
- NVIDIA GPU + CUDA 설치됨

---

## 1. 실행환경 구성

### 리눅스 환경 (CUDA + llama-cpp-python)

```bash
chmod +x install.sh
./install.sh
```

### 윈도우 환경 (CUDA 설정 포함)

```bash
pip install -r requirements.txt
CMAKE_ARGS="-DLLAMA_CUBLAS=OFF -DGGML_CUDA=ON" pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
```
---