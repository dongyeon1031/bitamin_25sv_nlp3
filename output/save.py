def save_submission(df, output_path: str = "./baseline_submission.csv"):
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
