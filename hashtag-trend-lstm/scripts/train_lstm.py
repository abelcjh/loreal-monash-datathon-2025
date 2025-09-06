import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import get_dataloaders
from src.lstm_multitarget import LSTMForecast
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_lstm(
    train_csv: str = "data/splits/train.csv",
    val_csv: str = "data/splits/val.csv",
    test_csv: str = "data/splits/test.csv",
    num_epochs: int = 20,
    batch_size: int = 32,
    seq_len: int = 5,
    lr: float = 1e-3,
    hidden_size: int = 64,
    num_layers: int = 2,
):
    # --- Data ---
    train_loader, val_loader = get_dataloaders(train_csv, val_csv, batch_size, seq_len)

    # --- Model ---
    model = LSTMForecast(
        input_size = 4,  # viewCount, likeCount, commentCount, mentions
        hidden_size = hidden_size,
        num_layers = num_layers,
        dropout = 0.2
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    # --- Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step() # type: ignore

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                preds = model(X_val)
                loss = criterion(preds, y_val)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # save checkpoint if best so far
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), "models/best_lstm.pt")
            best_val_loss = val_loss

    print("✅ Training complete. Best val_loss:", best_val_loss)

    # --- Final Test Evaluation ---
    if test_csv:
        print("\n📊 Running final evaluation on test set...")
        test_df = pd.read_csv(test_csv) # type: ignore
        from src.utils import make_sequences
        X_test, y_test = make_sequences(test_df, seq_len)
        test_ds = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model.load_state_dict(torch.load("models/best_lstm.pt"))  # reload best
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_t, y_t in test_loader:
                X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
                preds = model(X_t)
                loss = criterion(preds, y_t)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f"📉 Final test_loss={test_loss:.4f}")

if __name__ == "__main__":
    train_lstm()