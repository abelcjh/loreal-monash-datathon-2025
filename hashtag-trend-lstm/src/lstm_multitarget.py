import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    """
    Multi-target LSTM for forecasting engagement metric changes.
    Predicts future rate-of-change for:
    [mentions, viewCount, likeCount, commentCount]
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMForecast, self).__init__() # type: ignore
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 4)  # 4 outputs for metrics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape (batch, seq_len, input_size)
        Returns:
            Tensor: shape (batch, 5) → predicted next-step diffs
        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take last time step
        out = self.fc(out)
        return out


def get_model(input_size: int = 4, hidden_size: int = 64) -> LSTMForecast:
    """Helper to instantiate model"""
    return LSTMForecast(input_size=input_size, hidden_size=hidden_size)


if __name__ == "__main__":
    # Quick test run
    model = get_model()
    X = torch.randn(8, 10, 4)  # batch=8, seq_len=10, features=4
    y = model(X)
    print("Output shape:", y.shape)  # (8, 4)