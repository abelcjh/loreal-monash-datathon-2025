import pandas as pd

def compute_features(df: pd.DataFrame, freq: str = "H") -> pd.DataFrame:
    """
    Compute time-series features for hashtags:
    - Aggregate engagement metrics per hashtag and time bucket (daily/hourly)
    - Compute rate of change (diff) for each metric
    - Return structured features for ML

    Args:
        df (pd.DataFrame): Preprocessed DataFrame with ['publishedAt','hashtags','viewCount','likeCount','commentCount']
        freq (str): Resampling frequency, e.g., 'D' (daily), 'H' (hourly)

    Returns:
        pd.DataFrame: Aggregated features with diffs
    """
    if "publishedAt" not in df.columns or "hashtags" not in df.columns:
        raise ValueError("DataFrame must contain 'publishedAt' and 'hashtags'")

    expanded = df.explode('hashtags').copy()
    expanded['hashtag'] = expanded['hashtags'].str.lower()
    expanded['mentions'] = 1
    
    cols = ['hashtag', 'publishedAt', 'viewCount', 'likeCount', 'commentCount', 'mentions']
    expanded = expanded[cols].fillna(0) # type: ignore
    expanded = expanded.set_index("publishedAt")

    grouped = []
    for tag, g in expanded.groupby("hashtag"): # type: ignore
        g = g.resample(freq).sum().fillna(0) # type: ignore

        # Add rate of change features
        for col in ["mentions", "viewCount", "likeCount", "commentCount"]:
            g[f"{col}_diff"] = g[col].diff().fillna(0) # type: ignore

        g["hashtag"] = tag
        grouped.append(g.reset_index()) # type: ignore

    features = pd.concat(grouped, ignore_index=True) # type: ignore

    return features

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Compute hashtag features from preprocessed CSV")
    parser.add_argument("input_csv", help="Path to preprocessed CSV file")
    parser.add_argument("--output_csv", help="Where to save features", default=None)
    parser.add_argument("--freq", help="Resampling frequency (D=day, H=hour)", default="D")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input file not found: {args.input_csv}")

    try:
        df = pd.read_csv(args.input_csv, parse_dates=["publishedAt"]) # type: ignore
        features = compute_features(df, freq=args.freq)

        if args.output_csv:
            os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
            features.to_csv(args.output_csv, index=False)

        print(f"Computed features for {features['hashtag'].nunique()} hashtags.")
    except Exception as e:
        print(f"Error processing features: {e}")
        raise