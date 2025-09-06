import pandas as pd
from src.preprocessing import extract_hashtags, preprocess_csv

def test_extract_hashtags():
    text = "Check out #AI and #MachineLearning"
    tags = extract_hashtags(text)
    assert "#AI" in tags and "#MachineLearning" in tags

def test_preprocess_csv(tmp_path):
    file = tmp_path / "sample.csv"
    df = pd.DataFrame({
        'title': ["Video about #AI"],
        'publishedAt': ["2023-01-01"],
        'viewCount': [10],
        'likeCount': [1],
        'favouriteCount': [0],
        'commentCount': [0]
    })
    df.to_csv(file, index=False)
    df_out = preprocess_csv(str(file), str(tmp_path / "out.csv"))
    assert 'hashtags' in df_out.columns