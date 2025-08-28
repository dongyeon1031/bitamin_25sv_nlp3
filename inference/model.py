import os
from llama_cpp import Llama

def load_model_and_tokenizer(model_path: str):
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,        # 토큰 길이 설정
        n_threads=int(os.getenv("LLM_THREADS", "6")),       # CPU 코어 수에 따라 조절
        n_batch=int(os.getenv("LLM_BATCH", "128")),        # 배치 사이즈 (GPU 허용선에서 증가)
        n_gpu_layers=int(os.getenv("LLM_GPU_LAYERS", "-1")),    # GPU 사용
        f16_kv=True        # float16 key/value 캐시
    )
    return llm