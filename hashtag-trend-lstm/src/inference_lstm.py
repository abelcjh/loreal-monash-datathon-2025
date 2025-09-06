import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union

from src.lstm_multitarget import get_model

MODEL_PATH = Path("models/lstm_multitarget.pth")

def load_model(model_path: Union[str, Path] = MODEL_PATH, input_size: int = 5) -> torch.nn.Module:
    """Load trained LSTM model from disk."""
    model = get_model(input_size=input_size)
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

@torch.no_grad()
def predict(df: pd.DataFrame, model: torch.nn.Module, seq_len: int = 10) -> np.ndarray:
    """
    Run LSTM inference on feature dataframe.

    Args:
        df (pd.DataFrame): Must contain columns:
            ['mentions_diff','viewCount_diff','likeCount_diff','favouriteCount_diff','commentCount_diff']
        model: Loaded LSTM model
        seq_len (int): Number of timesteps per sequence

    Returns:
        np.ndarray: Predictions shape (num_samples, 5)
    """
    required = ["mentions_diff","viewCount_diff","likeCount_diff","favouriteCount_diff","commentCount_diff"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required feature: {col}")

    values = df[required].values.astype(np.float32)

    X = []
    for i in range(len(values) - seq_len):
        X.append(values[i:i+seq_len])
    if not X:
        return np.empty((0, 5))
    X = torch.tensor(np.stack(X))

    preds = model(X).cpu().numpy()
    return preds

if __name__ == "__main__":
    # Quick dry run
    sample = pd.DataFrame({
        "mentions_diff": [1,2,3,4,5,6,7,8,9,10,11],
        "viewCount_diff": [10]*11,
        "likeCount_diff": [2]*11,
        "favouriteCount_diff": [0]*11,
        "commentCount_diff": [1]*11
    })

    model = load_model()
    predictions = predict(sample, model, seq_len=5)
    print("Predictions shape:", predictions.shape)
    print(predictions[:3])