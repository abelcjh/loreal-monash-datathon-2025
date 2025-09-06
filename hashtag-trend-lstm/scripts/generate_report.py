import sys
import pandas as pd
import yaml
from src.trend_report import generate_trend_report

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def main(predictions_path, output_path):
    # Load params
    params = load_params()
    report_params = params["report"]

    # Load predictions
    df = pd.read_csv(predictions_path)

    reports = []
    for _, row in df.iterrows():
        metrics = {
            "mentions": [row["pred_mentions"]],
            "views": [row["pred_views"]],
            "likes": [row["pred_likes"]],
            "favourites": [row["pred_favourites"]],
            "comments": [row["pred_comments"]],
        }
        report = generate_trend_report(
            hashtag=row.get("hashtags", "#unknown"),
            metrics=metrics,
            emerging_threshold=report_params["emerging_threshold"],
            decaying_threshold=report_params["decaying_threshold"],
        )
        reports.append(report)

    # Save report
    with open(output_path, "w") as f:
        f.write("\n\n".join(reports))

    print(f"✅ Trend report saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/generate_report.py <predictions_path> <output_path>")
        sys.exit(1)

    predictions_path = sys.argv[1]
    output_path = sys.argv[2]
    main(predictions_path, output_path)