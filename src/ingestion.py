# 1. src/ingestion.py
import os
import pandas as pd
from validation import validate_dataset

# Final raw data path
RAW_DATA_PATH = os.path.join("data", "raw", "Telco-Customer-Churn.csv")

def load_data():
    """
    Loads the raw dataset produced by ingestion.
    """
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Raw data not found at {RAW_DATA_PATH}. Run ingestion.py first."
        )
    return pd.read_csv(RAW_DATA_PATH)


def ingest_data():
    """
    Ingests the cleaned dataset, saves it into data/raw,
    and immediately validates it.
    """
    # Source (already cleaned in notebook)
    source_path = os.path.join("data", "processed", "churn_clean.csv")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found at: {source_path}")

    df = pd.read_csv(source_path)

    # Ensure target directory exists
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)

    # Save as raw dataset for pipeline
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Data ingestion completed. Saved to: {RAW_DATA_PATH}")

    # Validate immediately
    validate_dataset(RAW_DATA_PATH)


if __name__ == "__main__":
    ingest_data()


# python "src\ingestion.py"