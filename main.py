from inference.runner import run_inference
from data.loader import load_test_data, load_sample_submission
from inference.model import load_model_and_tokenizer
from configs.model_config import MODEL_NAME
from output.save import save_submission
from inference.download import download_model

if __name__ == "__main__":
    download_model()

    model_name = MODEL_NAME

    test = load_test_data()
    submission = load_sample_submission()
    pipe = load_model_and_tokenizer(model_name)
    preds = run_inference(pipe, test)
    submission['Answer'] = preds
    save_submission(submission)
