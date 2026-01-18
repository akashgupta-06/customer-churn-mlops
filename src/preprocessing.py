# 2. src/preprocessing.py

import pandas as pd
from typing import Tuple
from ingestion import load_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


class PreprocessingError(Exception):
    """Custom exception for preprocessing failures."""
    pass


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer based on dataframe schema.

    Args:
        df (pd.DataFrame): Cleaned input dataframe including target.

    Returns:
        ColumnTransformer: Fitted preprocessing pipeline blueprint.
    """
    if "Churn" not in df.columns:
        raise PreprocessingError("Target column 'Churn' not found in dataset.")

    X = df.drop(columns=["Churn"])

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    if not num_cols and not cat_cols:
        raise PreprocessingError("No usable feature columns detected.")

    numeric_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop"
    )

    return preprocessor


def prepare_datasets(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    handle_imbalance: bool = True
) -> Tuple:
    """
    Prepare model-ready datasets:
    - Split features/target
    - Stratified train-test split
    - Fit and apply preprocessing
    - Optionally handle class imbalance using SMOTE

    Args:
        df (pd.DataFrame): Cleaned dataset.
        test_size (float): Proportion for test split.
        random_state (int): Random seed.
        handle_imbalance (bool): Whether to apply SMOTE on training data.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    if "Churn" not in df.columns:
        raise PreprocessingError("Target column 'Churn' not found in dataset.")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Ensure binary numeric target
    if y.dtype == "object":
        y = y.map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(df)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Optional imbalance handling
    if handle_imbalance:
        if not IMBLEARN_AVAILABLE:
            raise PreprocessingError(
                "imblearn is not installed, but handle_imbalance=True."
            )
        smote = SMOTE(random_state=random_state)
        X_train_processed, y_train = smote.fit_resample(
            X_train_processed, y_train
        )

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Local test hook
    from ingestion import load_data

    df = load_data()
    Xtr, Xte, ytr, yte, prep = prepare_datasets(df)

    print("Preprocessing successful.")
    print("Train shape:", Xtr.shape)
    print("Test shape:", Xte.shape)
    print("Class distribution (train):")
    print(ytr.value_counts())

# python 'src\preprocessing.py'