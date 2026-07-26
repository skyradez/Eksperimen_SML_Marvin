"""
modelling.py  -  Kriteria 2 (Basic / 2 pts)
===========================================
Melatih model klasifikasi churn (Scikit-Learn) dengan MLflow **autolog**,
disimpan pada MLflow Tracking UI lokal (http://127.0.0.1:5000).

Menjalankan
-----------
1. Jalankan tracking server (opsional, bisa juga file store lokal ./mlruns):
       mlflow ui --host 127.0.0.1 --port 5000
2. Latih model:
       python modelling.py
"""

import os
import argparse

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "telco_churn_preprocessing")


def load_split(data_dir=DATA_DIR):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    X_train, y_train = train.drop(columns=["Churn"]), train["Churn"]
    X_test, y_test = test.drop(columns=["Churn"]), test["Churn"]
    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", default="telco_churn_basic")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=12)
    args = parser.parse_args()

    # Gunakan tracking server lokal jika tersedia; jika tidak, fallback ke ./mlruns
    try:
        mlflow.set_tracking_uri(args.tracking_uri)
    except Exception:
        mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment)

    # ---- Basic requirement: autolog dari MLflow ----
    mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test = load_split(args.data_dir)

    with mlflow.start_run(run_name="rf_baseline_autolog"):
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        # autolog otomatis mencatat params, metrics (training) & model.
        # Evaluasi eksplisit agar test metrics juga tercatat.
        acc = model.score(X_test, y_test)
        print(f"Test accuracy: {acc:.4f}")

    print("Selesai. Buka MLflow UI di", args.tracking_uri)


if __name__ == "__main__":
    main()
