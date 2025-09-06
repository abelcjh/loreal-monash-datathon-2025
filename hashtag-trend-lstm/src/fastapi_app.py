import pandas as pd
import uvicorn
from fastapi import FastAPI, UploadFile, File
from typing import List
import io

from src.preprocessing import preprocess_csv
from src.features import compute_features
from src.inference_lstm import load_model, predict

app = FastAPI(title="Hashtag Trend Forecasting API")

# Load model at startup
model = load_model()

@app.post("/predict/")
async def predict_from_csv(file: UploadFile = File(...), freq: str = "D", seq_len: int = 10):
    """
    Upload a CSV of videos, preprocess, compute features,
    and return hashtag trend predictions.
    """
    # Read uploaded file into dataframe
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # Preprocess + feature engineering
    df = preprocess_csv(io.BytesIO(contents))
    features = compute_features(df, freq=freq)

    # Run inference
    preds = predict(features, model, seq_len=seq_len)

    # Map predictions back to hashtags + timestamps
    results = []
    for i, row in features.iloc[seq_len:].iterrows():
        results.append({
            "timestamp": row["publishedAt"].isoformat() if hasattr(row["publishedAt"], "isoformat") else str(row["publishedAt"]),
            "hashtag": row["hashtag"],
            "pred_mentions_diff": float(preds[i - seq_len, 0]),
            "pred_views_diff": float(preds[i - seq_len, 1]),
            "pred_likes_diff": float(preds[i - seq_len, 2]),
            "pred_favourites_diff": float(preds[i - seq_len, 3]),
            "pred_comments_diff": float(preds[i - seq_len, 4]),
        })

    return {"predictions": results}


if __name__ == "__main__":
    uvicorn.run("src.fastapi_app:app", host="0.0.0.0", port=8000, reload=True)