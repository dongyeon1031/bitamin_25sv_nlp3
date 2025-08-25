#!/bin/bash

pip install -r requirements.txt

# llama-cpp-python은 별도로 CMake 옵션 강제해서 설치
CMAKE_ARGS="-DLLAMA_CUBLAS=OFF -DGGML_CUDA=ON" \
pip install llama-cpp-python[cuda12]==0.3.16 --no-cache-dir --force-reinstall --upgrade