"""
Metrik yang diekspor (>= 10):
  1. http_requests_total                 (Counter)   - total request per endpoint/method/status
  2. http_request_duration_seconds       (Histogram) - latензi request
  3. model_predictions_total             (Counter)   - jumlah prediksi per kelas (churn/no-churn)
  4. model_prediction_errors_total       (Counter)   - jumlah error inferensi
  5. model_inference_duration_seconds    (Histogram) - durasi inferensi model
  6. model_prediction_confidence         (Gauge)     - confidence prediksi terakhir
  7. model_churn_ratio                   (Gauge)     - proporsi prediksi churn (running)
  8. app_active_requests                 (Gauge)     - request yang sedang diproses
  9. app_uptime_seconds                  (Gauge)     - uptime aplikasi
 10. system_cpu_usage_percent            (Gauge)     - penggunaan CPU host
 11. system_memory_usage_percent         (Gauge)     - penggunaan memori host
 12. system_disk_usage_percent           (Gauge)     - penggunaan disk host
"""

import os
import time
import threading

import numpy as np
import pandas as pd
import psutil
import joblib
from flask import Flask, request, jsonify, Response
from prometheus_client import (
    Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(HERE, "model.joblib"))
DATA_DIR = os.environ.get(
    "DATA_DIR",
    os.path.join(HERE, "..", "Membangun_model", "telco_churn_preprocessing"),
)

app = Flask(__name__)

# --------------------------------------------------------------------------- #
# Prometheus metrics (>= 10 metrik berbeda)
# --------------------------------------------------------------------------- #
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency (s)",
    ["endpoint"],
)
PREDICTIONS = Counter(
    "model_predictions_total", "Total predictions by class", ["predicted_class"],
)
PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total", "Total prediction errors",
)
INFERENCE_LATENCY = Histogram(
    "model_inference_duration_seconds", "Model inference duration (s)",
)
PREDICTION_CONFIDENCE = Gauge(
    "model_prediction_confidence", "Confidence (max proba) of last prediction",
)
CHURN_RATIO = Gauge(
    "model_churn_ratio", "Running ratio of churn predictions",
)
ACTIVE_REQUESTS = Gauge(
    "app_active_requests", "In-flight requests",
)
UPTIME = Gauge("app_uptime_seconds", "Application uptime (s)")
CPU_USAGE = Gauge("system_cpu_usage_percent", "System CPU usage percent")
MEM_USAGE = Gauge("system_memory_usage_percent", "System memory usage percent")
DISK_USAGE = Gauge("system_disk_usage_percent", "System disk usage percent")

_START = time.time()
_churn = {"churn": 0, "total": 0}


# --------------------------------------------------------------------------- #
# Model loading (fallback: latih cepat dari data preprocessing bila belum ada)
# --------------------------------------------------------------------------- #
def _load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    from sklearn.ensemble import RandomForestClassifier
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    X, y = train.drop(columns=["Churn"]), train["Churn"]
    model = RandomForestClassifier(n_estimators=200, max_depth=12,
                                   random_state=42, n_jobs=-1)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


MODEL = _load_or_train_model()
FEATURES = list(pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
                .drop(columns=["Churn"]).columns)


def _system_metrics_loop():
    """Refresh host metrics in the background."""
    while True:
        CPU_USAGE.set(psutil.cpu_percent(interval=None))
        MEM_USAGE.set(psutil.virtual_memory().percent)
        DISK_USAGE.set(psutil.disk_usage("/").percent)
        UPTIME.set(time.time() - _START)
        time.sleep(5)


@app.route("/health")
def health():
    REQUEST_COUNT.labels("GET", "/health", "200").inc()
    return jsonify(status="ok", uptime=round(time.time() - _START, 1))


@app.route("/metrics")
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    MEM_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage("/").percent)
    UPTIME.set(time.time() - _START)
    REQUEST_COUNT.labels("GET", "/metrics", "200").inc()
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/predict", methods=["POST"])
def predict():
    ACTIVE_REQUESTS.inc()
    t0 = time.time()
    try:
        payload = request.get_json(force=True)
        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
        df = pd.DataFrame(rows)
        # Selaraskan kolom dengan urutan fitur model
        for col in FEATURES:
            if col not in df.columns:
                df[col] = 0
        df = df[FEATURES]

        with INFERENCE_LATENCY.time():
            proba = MODEL.predict_proba(df)
            preds = MODEL.predict(df)

        for p, pr in zip(preds, proba):
            cls = "churn" if int(p) == 1 else "no_churn"
            PREDICTIONS.labels(cls).inc()
            PREDICTION_CONFIDENCE.set(float(np.max(pr)))
            _churn["total"] += 1
            _churn["churn"] += int(p == 1)
        if _churn["total"]:
            CHURN_RATIO.set(_churn["churn"] / _churn["total"])

        REQUEST_COUNT.labels("POST", "/predict", "200").inc()
        REQUEST_LATENCY.labels("/predict").observe(time.time() - t0)
        return jsonify(predictions=[int(p) for p in preds],
                       probabilities=[float(pr[1]) for pr in proba])
    except Exception as e:  # noqa
        PREDICTION_ERRORS.inc()
        REQUEST_COUNT.labels("POST", "/predict", "500").inc()
        REQUEST_LATENCY.labels("/predict").observe(time.time() - t0)
        return jsonify(error=str(e)), 500
    finally:
        ACTIVE_REQUESTS.dec()


if __name__ == "__main__":
    threading.Thread(target=_system_metrics_loop, daemon=True).start()
    port = int(os.environ.get("PORT", "8000"))
    print(f"Serving Telco Churn model on http://127.0.0.1:{port}")
    print(f"Prometheus metrics at http://127.0.0.1:{port}/metrics")
    app.run(host="0.0.0.0", port=port)
