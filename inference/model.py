import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        token=True,
        trust_remote_code=True,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        pad_token_id=tokenizer.eos_token_id
    )
    return pipe