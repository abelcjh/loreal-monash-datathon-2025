"""
Utility functions for the Hashtag Trend Forecasting project.
Includes logging, data loaders, reproducibility helpers, and model I/O.
"""

import os
import random
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# ------------------------------
# Model Input/Output Preparation
# ------------------------------
# features and targets from preprocessed dataset
FEATURE_COLS = ["mentions", "viewCount", "likeCount", "commentCount"]
TARGET_COLS= ["mentions", "viewCount", "likeCount", "commentCount"]

def make_sequences(df: pd.DataFrame, seq_len: int = 24):
    """
    Convert hashtag × hour dataframe into sliding window sequences.

    Args:
        df (pd.DataFrame): DataFrame with columns ['hashtag', 'PublishedAt'] + FEATURE_COLS
        seq_len (int): number of time steps in each sequence

    Returns:
        torch.Tensor: X (features), y (targets)
    """
    X_list  = []
    y_list = []

    # Group by hashtag
    for tag, group in df.groupby("hashtag"):  # type: ignore
        group = group.sort_values("publishedAt")  # ensure time ordering

        features = group[FEATURE_COLS].values
        targets = group[TARGET_COLS].values

        for i in range(len(group) - seq_len):
            X_list.append(features[i:i + seq_len]) # type: ignore
            y_list.append(targets[i + seq_len])  # predict next timestep # type: ignore

    # Convert to NumPy arrays first (faster)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Convert to PyTorch tensors
    return torch.tensor(X), torch.tensor(y)

def get_dataloaders(train_csv: str, val_csv: str, batch_size: int = 32, seq_len: int = 24):
    """
    Build PyTorch dataloaders from processed hashtag × hour CSVs.

    Args:
        train_csv (str): path to train split
        val_csv (str): path to validation split
        batch_size (int): batch size for DataLoader
        seq_len (int): sequence length for LSTM

    Returns:
        train_loader, val_loader
    """
    # Load splits
    train_df = pd.read_csv(train_csv) # type: ignore
    val_df = pd.read_csv(val_csv) # type: ignore

    # Make sequences
    X_train, y_train = make_sequences(train_df, seq_len)
    X_val, y_val = make_sequences(val_df, seq_len)

    # Wrap as TensorDataset
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader

# ------------------------------
# Logging Setup
# ------------------------------
def get_logger(name: str = "hashtag_trend", level: int = logging.INFO):
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:  # prevent duplicate handlers in interactive runs
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ------------------------------
# Reproducibility
# ------------------------------
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU seed for PyTorch operations # type: ignore
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------
# Model Save/Load
# ------------------------------
def save_model(model: torch.nn.Module, path: str):
    """Save PyTorch model state_dict."""
    try:
        # Validate path to prevent path traversal
        normalized_path = os.path.normpath(path)
        if not normalized_path.startswith(os.getcwd()):
            raise ValueError("Path must be within current working directory")
        
        os.makedirs(os.path.dirname(normalized_path), exist_ok=True)
        torch.save(model.state_dict(), normalized_path)
    except (FileNotFoundError, PermissionError, OSError) as e:
        raise RuntimeError(f"Failed to save model to {path}: {e}")


def load_model(model_class: torch.nn.Module, path: str, device: str = "cpu") -> torch.nn.Module:
    """
    Load PyTorch model state_dict into a fresh instance.

    Args:
        model_class: callable that instantiates the model (no args or with defaults).
        path (str): path to saved .pth file.
        device (str): cpu or cuda.
    """
    try:
        # Validate path to prevent path traversal
        normalized_path = os.path.normpath(path)
        if not normalized_path.startswith(os.getcwd()):
            raise ValueError("Path must be within current working directory")
        
        if not os.path.exists(normalized_path):
            raise FileNotFoundError(f"Model file not found: {normalized_path}")
        
        model = model_class()
        with torch.inference_mode():
            model.load_state_dict(torch.load(normalized_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except (FileNotFoundError, RuntimeError, KeyError) as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}")