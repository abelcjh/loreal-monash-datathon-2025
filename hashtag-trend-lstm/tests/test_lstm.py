import torch
from src.lstm_multitarget import LSTMForecast

def test_lstm_forward():
    # Define model with dummy dimensions
    input_size = 5   # number of input features
    hidden_size = 16
    output_size = 5  # predicting 5 metrics
    model = LSTMForecast(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=2)
    
    # Dummy batch: batch_size=2, seq_len=4, features=5
    X = torch.randn(2, 4, input_size)
    out = model(X)
    
    # Output should be (batch_size, output_size)
    assert out.shape == (2, output_size)
    
    # Output should be float tensor
    assert out.dtype == torch.float32