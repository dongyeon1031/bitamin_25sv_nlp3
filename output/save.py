def save_submission(df, output_path: str = "./submission.csv"):
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
