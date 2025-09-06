import pandas as pd
from src.inference_lstm import predict, load_model

def test_predict(tmp_path):
    # Load model (untrained but valid for forward pass)
    model = load_model()
    
    # Create dummy input data
    df = pd.DataFrame({
        'viewCount': [1, 2, 3, 4],
        'likeCount': [1, 1, 1, 1],
        'favouriteCount': [0, 0, 0, 0],
        'commentCount': [0, 0, 0, 0]
    })
    
    # Run prediction
    preds = predict(df, model)
    
    # Shape: rows=4, outputs=5 (mentions_diff, views, likes, favourites, comments)
    assert preds.shape[0] == 4
    assert preds.shape[1] == 5