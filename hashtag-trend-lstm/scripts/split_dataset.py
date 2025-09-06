import pandas as pd
import argparse
from pathlib import Path

def split_dataset(input_csv: str, output_dir: str,
                  train_ratio: float = 0.7, val_ratio: float = 0.15,
                  test_ratio: float = 0.10, live_ratio: float = 0.05):
    """
    Split dataset into train/val/test/live sets by chronological order.
    Ensures no overlap, and drops 'favouriteCount' if present.
    """
    df = pd.read_csv(input_csv)  # type: ignore

    # Drop favouriteCount if exists
    if "favouriteCount" in df.columns:
        df = df.drop(columns=["favouriteCount"])

    # Ensure time ordering
    if "publishedAt" in df.columns:
        df = df.sort_values("publishedAt")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = train_end + int(n * val_ratio)
    test_end  = val_end + int(n * test_ratio)
    live_end  = test_end + int(n * live_ratio)

    # Defensive: cap at n in case of rounding issues
    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:test_end]
    live_df  = df.iloc[test_end:live_end]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_path / "train.csv", index=False)
    val_df.to_csv(out_path / "val.csv", index=False)
    test_df.to_csv(out_path / "test.csv", index=False)
    live_df.to_csv(out_path / "live.csv", index=False)

    print(f"✅ Split complete: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test, {len(live_df)} live samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test/live")
    parser.add_argument("input_csv", help="Path to cleaned CSV file")
    parser.add_argument("output_dir", default="data/splits", help="Where to save outputs")
    args = parser.parse_args()

    split_dataset(args.input_csv, args.output_dir)