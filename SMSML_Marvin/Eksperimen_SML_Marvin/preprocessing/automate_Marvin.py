from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Column groups (kept in one place so the notebook and this file stay in sync)
# --------------------------------------------------------------------------- #
TARGET = "Churn"
ID_COL = "customerID"
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
# Binary Yes/No style columns (SeniorCitizen is already 0/1)
BINARY_MAP_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
# Nominal columns that get one-hot encoded
NOMINAL_COLS = [
    "gender", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaymentMethod",
]


def load_data(input_path: str) -> pd.DataFrame:
    """Step 1 - Data loading (mirrors notebook section 3)."""
    df = pd.read_csv(input_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2 - Missing values, duplicates, dtype fixes (notebook section 5)."""
    df = df.copy()

    # Drop the identifier column - it carries no predictive signal.
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # `TotalCharges` arrives as text because new customers (tenure == 0) have a
    # blank value. Coerce to numeric and impute the resulting NaNs with 0,
    # which is the correct value for a customer that has not been billed yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Remove exact duplicate rows.
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3 - Encoding categorical data (notebook section 5)."""
    df = df.copy()

    # Target -> 0/1
    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    # Binary Yes/No -> 1/0
    for col in BINARY_MAP_COLS:
        df[col] = (df[col] == "Yes").astype(int)

    # Nominal -> one-hot (drop_first avoids the dummy-variable trap)
    df = pd.get_dummies(df, columns=NOMINAL_COLS, drop_first=True)

    # Cast any boolean dummy columns to int for a clean, uniform matrix.
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def scale_and_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale only the continuous numeric columns; fit on train, apply to test
    # to avoid data leakage.
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    return X_train, X_test, y_train, y_test, scaler


def preprocess_data(
    input_path: str,
    output_dir: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    
    df = load_data(input_path)
    df = clean_data(df)
    df = encode_features(df)
    X_train, X_test, y_train, y_test, scaler = scale_and_split(
        df, test_size=test_size, random_state=random_state
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Full processed frame (features + target) - handy single artefact.
        full = df.copy()
        full.to_csv(os.path.join(output_dir, "telco_churn_clean.csv"), index=False)

        # Ready-to-train splits.
        train = X_train.copy()
        train[TARGET] = y_train.values
        test = X_test.copy()
        test[TARGET] = y_test.values
        train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

        # Persist the fitted scaler so serving uses identical transforms.
        joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))

        print(f"[automate_Marvin] Wrote processed data to: {output_dir}")
        print(f"  train.csv : {train.shape}")
        print(f"  test.csv  : {test.shape}")
        print(f"  features  : {X_train.shape[1]}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(X_train.columns),
    }


def _default_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.join(here, "..", "telco_churn_raw", "telco_churn_raw.csv")
    default_out = os.path.join(here, "telco_churn_preprocessing")
    return default_in, default_out


if __name__ == "__main__":
    d_in, d_out = _default_paths()
    parser = argparse.ArgumentParser(description="Automated Telco Churn preprocessing")
    parser.add_argument("--input", default=d_in, help="Path to raw CSV")
    parser.add_argument("--output", default=d_out, help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    preprocess_data(
        input_path=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print("[automate_Marvin] Preprocessing finished successfully.")
