from constants import TEST_CSV_PATH, SAMPLE_SUBMISSION_PATH
import pandas as pd

def load_test_data() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV_PATH)

def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_SUBMISSION_PATH)