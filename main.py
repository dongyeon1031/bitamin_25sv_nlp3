from inference.runner import run_inference
from data.loader import load_test_data, load_sample_submission
from output.save import save_submission

if __name__ == "__main__":

    test = load_test_data()
    submission = load_sample_submission()
    preds = run_inference(test)
    submission['Answer'] = preds
    save_submission(submission)
