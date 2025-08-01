## 윈도우 환경에서 실행 안되는 경우 확인해주세요!
- 현재 코드에선 bitsandbytes 패키지를 통해 모델을 4비트 양자화해서 로드하는 로직이 포함돼있습니다.
- bitsandbytes는 기본적으로 Linux/Ubuntu 전용이기 때문에 해당 부분 코멘트처리 해주시면 됩니다!
- 깃에 푸쉬할땐 다시 코멘트 풀어서 해주세요.

### code
- inference.model.py 에서
```python
  model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,  # 이 부분 코멘트처리 해주세요!
        torch_dtype=torch.float16
    )
```
