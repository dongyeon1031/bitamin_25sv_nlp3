from inference.runner_rag import run_inference_with_rag

from inference.runner import run_inference
from data.loader import load_test_data, load_sample_submission
from inference.model import load_model_and_tokenizer
from configs.model_config import MODEL_NAME
from output.save import save_submission

if __name__ == "__main__":
    '''
        여기 나중에 따로 빼기..
    '''
    import os
    from huggingface_hub import snapshot_download

    model_dir = "./models/claude3-gemma"
    if not os.path.exists(model_dir):
        snapshot_download(
            repo_id="reedmayhew/claude-3.7-sonnet-reasoning-gemma3-12B",
            local_dir=model_dir,
            local_dir_use_symlinks=False  # 하드카피로 저장
        )
    

    model_name = MODEL_NAME

    test = load_test_data()
    submission = load_sample_submission()
    pipe = load_model_and_tokenizer(model_name)
    preds = run_inference(pipe, test)


    preds = run_inference_with_rag(pipe, test)
    submission['Answer'] = preds
    save_submission(submission)