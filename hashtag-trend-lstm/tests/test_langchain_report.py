import pandas as pd
import yaml
from src.langchain_report import generate_langchain_report

def test_generate_report_respects_thresholds(tmp_path, monkeypatch):
    # Load thresholds from params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    emerging_th = params["report"]["emerging_threshold"]

    # Create a sample prediction CSV with ROC > emerging_th
    df = pd.DataFrame([
        {
            "hashtags": "#ai",
            "mentions": 100,
            "pred_mentions": 120,  # +20% growth
            "pred_views": 1000,
            "pred_likes": 100,
            "pred_favourites": 10,
            "pred_comments": 5,
        }
    ])
    preds_path = tmp_path / "preds.csv"
    df.to_csv(preds_path, index=False)

    # Monkeypatch LLM to avoid calling real API
    monkeypatch.setattr(
        "src.langchain_report.OpenAI",
        lambda *args, **kwargs: (lambda prompt: f"[MOCKED LLM RESPONSE] {prompt}")
    )

    # Generate report
    report = generate_langchain_report(str(preds_path))

    # Assert it contains "Emerging" since ROC > emerging_th
    assert "Emerging" in report
    assert "#ai" in report
    assert "[MOCKED LLM RESPONSE]" in report