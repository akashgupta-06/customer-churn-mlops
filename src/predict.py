# 6. src/predict.py

import os
import joblib
import pandas as pd
from typing import Dict, Any


ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")


class PredictionError(Exception):
    """Custom exception for prediction failures."""
    pass


def load_artifact():
    if not os.path.exists(MODEL_PATH):
        raise PredictionError(
            f"Model artifact not found at {MODEL_PATH}. Train the model first."
        )

    artifact = joblib.load(MODEL_PATH)

    model = artifact.get("model")
    preprocessor = artifact.get("preprocessor")

    if model is None or preprocessor is None:
        raise PredictionError("Invalid model artifact format.")

    return model, preprocessor


def _apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate all derived features exactly as done during training.
    This guarantees schema parity between training and inference.
    """

    # AvgCharges = TotalCharges / tenure
    if "AvgCharges" not in df.columns:
        if "TotalCharges" in df.columns and "tenure" in df.columns:
            df["AvgCharges"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
        else:
            raise PredictionError(
                "Cannot compute AvgCharges: missing TotalCharges or tenure"
            )

    # Add any future engineered features here
    # e.g. df["is_monthly_contract"] = (df["Contract"] == "Month-to-month").astype(int)

    return df


def predict_single(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict churn for a single customer.

    Args:
        customer_data (dict): Raw customer features (same schema as training data, excluding 'Churn').

    Returns:
        dict: Prediction result with label and probability.
    """
    model, preprocessor = load_artifact()

    # Convert to DataFrame
    df = pd.DataFrame([customer_data])

    # Apply deterministic feature engineering
    df = _apply_feature_engineering(df)

    # Transform using trained preprocessor
    X_processed = preprocessor.transform(df)

    # Predict
    prob = model.predict_proba(X_processed)[0, 1]
    label = int(prob >= 0.5)

    return {
        "churn_probability": float(round(prob, 4)),
        "churn_label": label,
        "risk_level": "High" if label == 1 else "Low",
    }


if __name__ == "__main__":
    # Example usage (must match training schema)
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 430.0,
    }

    result = predict_single(sample_customer)
    print("Prediction Result")
    print(result)



# python 'src/predict.py'