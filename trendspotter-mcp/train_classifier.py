# src/train_classifier.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from features import compute_timeseries_features
import os

MODEL_OUT = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(MODEL_OUT, exist_ok=True)

class TrendDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )
    def forward(self,x): return self.net(x)

def heuristic_label(row):
    # label: 0=Emerging (fast positive slope), 1=Peak (high counts, low slope), 2=Decay (negative slope)
    if row['slope'] > 5 and row['recent_vs_overall'] > 1.2:
        return 0
    if row['slope'] < -1 and row['recent_vs_overall'] < 0.9:
        return 2
    return 1

def train_on_csv(csv_path, epochs=50):
    df = pd.read_csv(csv_path)
    feats = compute_timeseries_features(df)
    feats = feats.fillna(0)
    feats['label'] = feats.apply(heuristic_label, axis=1)
    X = feats[['slope','volatility','max_count','mean_recent','age_days','recent_vs_overall']].values
    y = feats['label'].values
    # normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    np.save(os.path.join(MODEL_OUT, "scaler.npy"), scaler.mean_)
    # dataset
    ds = TrendDataset(Xs, y)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    model = MLP(Xs.shape[1])
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        if (ep+1) % 10 == 0:
            print(f"Epoch {ep+1}/{epochs} loss={total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), os.path.join(MODEL_OUT,"trend_mlp.pth"))
    print("Saved model to models/trend_mlp.pth")
    return model, scaler

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv)>1 else "../data/synthetic_trends.csv"
    train_on_csv(csv_path)
