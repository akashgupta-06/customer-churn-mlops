# src/train.py
# Logistic model is best to use 

import os
import joblib
import argparse
from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from ingestion import load_data
from preprocessing import prepare_datasets


ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")


class TrainingError(Exception):
    """Custom exception for training failures."""
    pass


def evaluate(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def train_model(model_name: str = "logistic_regression", run_name: str = None):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # Load and prepare data
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = prepare_datasets(df)

    # Select model
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
        params = {"model": "LogisticRegression", "max_iter": 1000}
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
        params = {
            "model": "RandomForest",
            "n_estimators": 300,
            "max_depth": None,
        }
    else:
        raise TrainingError(f"Unsupported model: {model_name}")

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_pred, y_prob)

    # Persist model + preprocessor together
    artifact = {
        "model": model,
        "preprocessor": preprocessor,
    }
    joblib.dump(artifact, MODEL_PATH)

    # Optional MLflow logging
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("customer-churn-mlops")
        with mlflow.start_run(run_name=run_name or model_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "random_forest"],
        help="Model to train",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (e.g., logreg_v1, rf_v1)",
    )

    args = parser.parse_args()

    metrics = train_model(model_name=args.model, run_name=args.run_name)

    print("Training completed.")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        

# To run Models
'''
'python src\train.py --model logistic_regression --run-name logreg_v1'  --> logistic model
'python src\train.py --model random_forest --run-name rf_v1'              --> random forest model

# Change the run-name to control version of model
'''

