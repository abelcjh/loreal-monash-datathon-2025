import sys
import torch
import pandas as pd
import yaml
from src.lstm_multitarget import LSTMForecast

def load_params():
    """Load config from params.yaml"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def main(model_path, features_path, output_path):
    # Load params
    params = load_params()
    lstm_params = params["lstm"]

    # Load data
    df = pd.read_csv(features_path)
    X = df.drop(columns=["hashtags"]).values  # features
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (batch, seq=1, features)

    # Define model
    model = LSTMForecast(
        input_size=lstm_params["input_size"],
        hidden_size=lstm_params["hidden_size"],
        num_layers=lstm_params["num_layers"],
        dropout=lstm_params["dropout"],
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    # Save predictions
    pred_df = pd.DataFrame(predictions, columns=["pred_views", "pred_likes", "pred_favourites", "pred_comments", "pred_mentions"])
    pred_df.to_csv(output_path, index=False)

    print(f"✅ Predictions saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python scripts/predict_lstm.py <model_path> <features_path> <output_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    features_path = sys.argv[2]
    output_path = sys.argv[3]
    main(model_path, features_path, output_path)