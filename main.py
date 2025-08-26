from data.loader import load_test_data, load_sample_submission
from inference.model import load_model_and_tokenizer
from configs.model_config import MODEL_NAME
from output.save import save_submission
from inference.runner_rag import run_inference_mixed   

if __name__ == "__main__":
    import os
   
    model_name = MODEL_NAME
    pipe = load_model_and_tokenizer(model_name)

    test = load_test_data()
    submission = load_sample_submission()


    preds = run_inference_mixed(
        pipe,
        test,
        score_threshold=0.03,   
        use_reranker=True,     
        top_k_retrieve=30       
    )

    submission["Answer"] = preds
    save_submission(submission, "./submission_RAG.csv")
