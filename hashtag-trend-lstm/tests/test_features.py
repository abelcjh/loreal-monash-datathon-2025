import pandas as pd
from src.features import compute_features

def test_compute_features():
    df = pd.DataFrame({
        'publishedAt': pd.date_range("2023-01-01", periods=2, freq='D'),
        'hashtags': [['#AI'], ['#AI']],
        'viewCount': [10, 20],
        'likeCount': [1, 2],
        'favouriteCount': [0, 0],
        'commentCount': [0, 1]
    })
    features = compute_features(df)
    
    # Basic checks
    assert 'hashtags' in features.columns
    assert 'viewCount' in features.columns
    
    # Ensure rate-of-change features exist
    assert any("diff" in col for col in features.columns)