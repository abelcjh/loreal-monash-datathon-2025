# src/data_ingest.py
import os
import json
import time
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/synthetic_trends.csv")

def generate_synthetic_data(n_trends=8, days=30, seed=42):
    np.random.seed(seed)
    rows = []
    base_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
    for t in range(n_trends):
        name = f"#trend{t+1}"
        start = np.random.randint(0, days//2)
        peak = start + np.random.randint(3,10)
        amplitude = np.random.randint(50, 500)
        for d in range(days):
            date = base_date + pd.Timedelta(days=d)
            # build a bell-like curve around peak with noise
            value = amplitude * np.exp(-0.5 * ((d - peak)/3)**2) 
            value += np.random.randn() * amplitude * 0.05
            rows.append({"hashtag": name, "date": date.strftime("%Y-%m-%d"), "count": max(0, int(value))})
    df = pd.DataFrame(rows)
    df.to_csv(DATA_PATH, index=False)
    return df

@app.route("/ingest/generate", methods=["POST"])
def generate():
    df = generate_synthetic_data()
    return jsonify({"status":"ok", "rows": len(df)})

@app.route("/ingest/mock", methods=["POST"])
def ingest_mock():
    # This endpoint simulates ingestion triggered by n8n
    payload = request.get_json(silent=True) or {}
    # For demo: just ensure synthetic data exists
    if not os.path.exists(DATA_PATH):
        generate_synthetic_data()
    return jsonify({"status":"ingested","payload": payload})

@app.route("/data/latest", methods=["GET"])
def latest_data():
    if not os.path.exists(DATA_PATH):
        generate_synthetic_data()
    df = pd.read_csv(DATA_PATH)
    return jsonify(df.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(port=8000, debug=True)
