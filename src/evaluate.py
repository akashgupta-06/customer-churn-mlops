# 5. src/evaluate.py

import os
import joblib
from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from ingestion import load_data
from preprocessing import prepare_datasets


ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")


class EvaluationError(Exception):
    """Custom exception for evaluation failures."""
    pass


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def evaluate_model() -> Dict[str, float]:
    if not os.path.exists(MODEL_PATH):
        raise EvaluationError(
            f"Model artifact not found at {MODEL_PATH}. Train the model first."
        )

    artifact = joblib.load(MODEL_PATH)
    model = artifact.get("model")
    preprocessor = artifact.get("preprocessor")

    if model is None or preprocessor is None:
        raise EvaluationError("Invalid model artifact format.")

    # Load fresh data and reproduce split
    df = load_data()
    X_train, X_test, y_train, y_test, _ = prepare_datasets(
        df, handle_imbalance=False
    )

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)

    print("Evaluation Metrics")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")

    print("\nClassification Report")
    print("-" * 40)
    print(classification_report(y_test, y_pred))

    return metrics


if __name__ == "__main__":
    evaluate_model()


# python 'src/evaluate.py'