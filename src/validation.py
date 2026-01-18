# scr/validation

# Do NOT RUN VALIDATION CODE

import os
import pandas as pd

REQUIRED_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]

def validate_dataset(file_path: str) -> bool:
    """
    Validates the dataset for existence, schema, and basic integrity.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    df = pd.read_csv(file_path)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["Churn"].isna().any():
        raise ValueError("Target column 'Churn' contains null values.")

    print("Data validation passed.")
    return True


if __name__ == "__main__":
    path = os.path.join("data", "raw", "churn.csv")
    validate_dataset(path)
