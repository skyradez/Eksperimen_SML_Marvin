import os
import time
import argparse
import random

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "Membangun_model", "telco_churn_preprocessing")


def sample_rows(n):
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).drop(columns=["Churn"])
    idx = [random.randint(0, len(test) - 1) for _ in range(n)]
    return test.iloc[idx].to_dict(orient="records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/predict")
    ap.add_argument("--n", type=int, default=100, help="jumlah request")
    ap.add_argument("--delay", type=float, default=0.2, help="jeda antar request (s)")
    args = ap.parse_args()

    rows = sample_rows(args.n)
    ok = err = 0
    for i, row in enumerate(rows, 1):
        try:
            r = requests.post(args.url, json={"data": [row]}, timeout=10)
            if r.status_code == 200:
                ok += 1
                out = r.json()
                if i <= 5 or i % 25 == 0:
                    print(f"[{i}] pred={out['predictions']} "
                          f"proba={round(out['probabilities'][0], 3)}")
            else:
                err += 1
                print(f"[{i}] HTTP {r.status_code}: {r.text[:120]}")
        except Exception as e:  # noqa
            err += 1
            print(f"[{i}] ERROR: {e}")
        time.sleep(args.delay)

    print(f"\nSelesai. Sukses={ok} Gagal={err} Total={args.n}")
    print("Cek metrik di http://127.0.0.1:8000/metrics dan dashboard Grafana.")


if __name__ == "__main__":
    main()
