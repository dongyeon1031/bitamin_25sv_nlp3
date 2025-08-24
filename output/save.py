def save_submission(df, output_path: str = "./submission.csv"):
    # chars_to_remove = ['*', '-', ':']
    # df_cleaned = df.applymap(lambda x: ''.join(c for c in str(x) if c not in chars_to_remove))
    # df_cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")