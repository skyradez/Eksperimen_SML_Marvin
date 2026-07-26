"""
modelling_tuning.py  -  Kriteria 2 (Skilled 3 pts / Advance 4 pts)
==================================================================
- Hyperparameter tuning (GridSearchCV).
- **Manual logging** (bukan autolog) dengan metrik yang setara autolog
  (accuracy, precision, recall, f1, roc_auc) + log_loss.
- Advance: >= 2 artefak tambahan di luar cakupan autolog
  (confusion matrix, ROC curve, classification report, feature importance,
  metric summary) dan tracking online ke **DagsHub**.

Mode tracking
-------------
* Lokal  : set MLFLOW_TRACKING_URI=http://127.0.0.1:5000  (default)
* DagsHub: set env DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME (+ DAGSHUB_TOKEN)
           lalu jalankan dengan --use-dagshub.

Menjalankan
-----------
    python modelling_tuning.py                 # lokal
    python modelling_tuning.py --use-dagshub   # online DagsHub
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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, classification_report,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "telco_churn_preprocessing")
ART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_artifacts_tmp")


def load_split(data_dir=DATA_DIR):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    X_train, y_train = train.drop(columns=["Churn"]), train["Churn"]
    X_test, y_test = test.drop(columns=["Churn"]), test["Churn"]
    return X_train, X_test, y_train, y_test


def setup_tracking(use_dagshub: bool, experiment: str):
    if use_dagshub:
        import dagshub

        dagshub.init(
            repo_owner="marvinluckianto5",
            repo_name="Eksperimen_SML_Marvin",
            mlflow=True
        )

        print("[tracking] DagsHub enabled")

    else:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        print("[tracking] Local MLflow")

    mlflow.set_experiment(experiment)


def build_plots(y_test, y_pred, y_proba, model, feature_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    # 1) Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    p = os.path.join(out_dir, "confusion_matrix.png"); fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    paths["confusion_matrix"] = p

    # 2) ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend()
    p = os.path.join(out_dir, "roc_curve.png"); fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    paths["roc_curve"] = p

    # 3) Feature importance (top 15)
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.iloc[::-1].plot.barh(ax=ax, color="teal")
    ax.set_title("Top 15 Feature Importances")
    p = os.path.join(out_dir, "feature_importance.png"); fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    paths["feature_importance"] = p

    # 4) Classification report (text)
    rep = classification_report(y_test, y_pred, target_names=["No", "Yes"])
    p = os.path.join(out_dir, "classification_report.txt")
    with open(p, "w") as f:
        f.write(rep)
    paths["classification_report"] = p

    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--use-dagshub", action="store_true")
    parser.add_argument("--experiment", default="telco_churn_tuning")
    args = parser.parse_args()

    setup_tracking(args.use_dagshub, args.experiment)

    X_train, X_test, y_train, y_test = load_split(args.data_dir)

    # ---- Hyperparameter tuning ----
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, None],
        "min_samples_split": [2, 5],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0,
    )

    with mlflow.start_run(run_name="rf_gridsearch_manual_logging"):
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

        # ---- Manual logging: params ----
        for k, v in grid.best_params_.items():
            mlflow.log_param(k, v)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("tuning", "GridSearchCV")

        # Predictions
        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]

        # ---- Manual logging: metrics (setara autolog) ----
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "log_loss": log_loss(y_test, y_proba),
            "best_cv_f1": grid.best_score_,
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        print("Metrics:", {k: round(v, 4) for k, v in metrics.items()})

        # ---- Log the model ----
        mlflow.sklearn.log_model(best, artifact_path="model")

        # ---- Advance: >=2 artefak tambahan di luar autolog ----
        plot_paths = build_plots(y_test, y_pred, y_proba, best,
                                 list(X_train.columns), ART_DIR)
        for name, p in plot_paths.items():
            mlflow.log_artifact(p, artifact_path="extra_artifacts")

        # metric summary JSON (artefak tambahan ke-5)
        summary_path = os.path.join(ART_DIR, "metric_summary.json")
        with open(summary_path, "w") as f:
            json.dump({"best_params": grid.best_params_, "metrics": metrics}, f, indent=2)
        mlflow.log_artifact(summary_path, artifact_path="extra_artifacts")

        print("Logged model + extra artifacts:", list(plot_paths.keys()) + ["metric_summary.json"])

    print("Selesai. Best params:", grid.best_params_)


if __name__ == "__main__":
    main()
