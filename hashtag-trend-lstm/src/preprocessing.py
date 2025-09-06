# src/preprocessing.py
import re
import pandas as pd
from typing import List

HASHTAG_PATTERN = re.compile(r"#\w+")

def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from a video title."""
    return HASHTAG_PATTERN.findall(text)

def preprocess_csv(input_path: str, output_path: str = '') -> pd.DataFrame:
    """
    Preprocess raw CSV file:
    - Parse dates
    - Convert numeric fields
    - Extract hashtags
    - Drop rows without hashtags
    """
    df = pd.read_csv(input_path) # type: ignore

    # Ensure timestamp parsing
    if "publishedAt" in df.columns:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")

    # Ensure numeric conversion
    for col in ["viewCount", "likeCount", "commentCount"]:  # Removed favouriteCount
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int) # type: ignore

    # Extract hashtags
    df["hashtags"] = df["title"].apply(extract_hashtags) # type: ignore

    # Remove rows without hashtags
    df = df[df["hashtags"].map(len) > 0]

    # Save if requested
    if output_path:
        df.to_csv(output_path, index=False)

    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess raw video CSV file")
    parser.add_argument("input_csv", help="Path to raw CSV file")
    parser.add_argument("--output_csv", help="Where to save processed CSV", default=None)
    args = parser.parse_args()

    processed = preprocess_csv(args.input_csv, args.output_csv)
    print(f"Processed {len(processed)} rows with hashtags.")