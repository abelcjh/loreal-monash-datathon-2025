# src/inference.py
import os
import numpy as np
import torch
from train_classifier import MLP
import pandas as pd
from features import compute_timeseries_features
from embeddings import TextEmbedder

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")

def load_model():
    scaler_mean = None
    # in our simple script we saved scaler mean only - in production save scaler object
    model = MLP(6)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR,"trend_mlp.pth"), map_location='cpu'))
    model.eval()
    return model

def predict_lifecycle(csv_path="../data/synthetic_trends.csv"):
    df = pd.read_csv(csv_path)
    feats_df = compute_timeseries_features(df)
    feat_cols = ['slope','volatility','max_count','mean_recent','age_days','recent_vs_overall']
    X = feats_df[feat_cols].fillna(0).values
    # naive scaling
    X = (X - X.mean(axis=0)) / (X.std(axis=0)+1e-6)
    model = load_model()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).numpy()
        preds = logits.argmax(axis=1)
    feats_df['pred'] = preds
    label_map = {0:"Emerging",1:"Peak",2:"Decay"}
    feats_df['label'] = feats_df['pred'].map(label_map)
    # Add embeddings for clustering / similarity
    embedder = TextEmbedder()
    tags = feats_df['hashtag'].tolist()
    embs = embedder.embed_texts(tags)
    feats_df['embedding'] = list(embs)
    return feats_df

if __name__ == "__main__":
    print(predict_lifecycle().to_dict(orient='records'))
