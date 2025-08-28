from data.loader import load_test_data, load_sample_submission
from output.save import save_submission
from inference.runner_rag import run_inference_ensemble   

if __name__ == "__main__":
    import os

    test = load_test_data()
    submission = load_sample_submission()

    # 간단한 Ensemble Reranker를 사용한 추론
    preds = run_inference_ensemble(
        test,
        score_threshold=0.03,   
        use_ensemble=True,     
        top_k_retrieve=30       
    )

    submission["Answer"] = preds
    save_submission(submission, "./submission_Simple_Ensemble.csv")
