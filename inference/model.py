from llama_cpp import Llama

def load_model_and_tokenizer(model_path: str):
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,        # 토큰 길이 설정
        n_threads=4,       # CPU 코어 수에 따라 조절
        n_batch=32,        # 배치 사이즈
        n_gpu_layers=-1,    # GPU 사용
        f16_kv=True        # float16 key/value 캐시
    )
    return llm