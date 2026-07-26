"""
modelling.py  -  Kriteria 3 (Workflow CI via MLflow Project)
============================================================
Entry point untuk MLflow Project. Melatih model churn, mencatat metrik &
artefak dengan manual logging, lalu menyimpan model sehingga dapat dikemas
menjadi Docker image (`mlflow build-docker`) pada tahap Advance.

Dijalankan otomatis oleh GitHub Actions melalui `mlflow run MLProject`.
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, ConfusionMatrixDisplay, roc_curve,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_split(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return (train.drop(columns=["Churn"]), test.drop(columns=["Churn"]),
            train["Churn"], test["Churn"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(HERE, "telco_churn_preprocessing"))
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--max_depth", type=int, default=12)
    ap.add_argument("--min_samples_split", type=int, default=2)
    args = ap.parse_args()

    X_train, X_test, y_train, y_test = load_split(args.data_dir)

    # MLflow Project berjalan di dalam run aktif; buat run eksplisit agar aman
    # dijalankan langsung maupun via `mlflow run`.
    active = mlflow.active_run()
    ctx = mlflow.start_run(nested=bool(active)) if active else mlflow.start_run(run_name="ci_retrain")
    with ctx:
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)

        model = RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            min_samples_split=args.min_samples_split, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "log_loss": log_loss(y_test, y_proba),
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        print("Metrics:", {k: round(v, 4) for k, v in metrics.items()})

        # Artefak: confusion matrix + roc curve
        os.makedirs(os.path.join(HERE, "artifacts"), exist_ok=True)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(ax=ax, cmap="Blues", colorbar=False)
        cmp = os.path.join(HERE, "artifacts", "confusion_matrix.png")
        fig.tight_layout(); fig.savefig(cmp, dpi=120); plt.close(fig)
        mlflow.log_artifact(cmp, artifact_path="plots")

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}"); ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.legend(); ax.set_title("ROC Curve")
        rp = os.path.join(HERE, "artifacts", "roc_curve.png")
        fig.tight_layout(); fig.savefig(rp, dpi=120); plt.close(fig)
        mlflow.log_artifact(rp, artifact_path="plots")

        with open(os.path.join(HERE, "artifacts", "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(os.path.join(HERE, "artifacts", "metrics.json"))

        # Log + register model (dipakai oleh `mlflow build-docker`)
        mlflow.sklearn.log_model(
            model, artifact_path="model",
            registered_model_name=os.environ.get("MODEL_NAME", None),
        )
        run_id = mlflow.active_run().info.run_id
        # Simpan run_id agar step CI berikutnya dapat membangun docker image
        with open(os.path.join(HERE, "run_id.txt"), "w") as f:
            f.write(run_id)
        print("RUN_ID:", run_id)
        print("Model logged at runs:/%s/model" % run_id)


if __name__ == "__main__":
    main()
