import yaml
import pandas as pd
from langchain_community.llms import OpenAI

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def generate_langchain_report(predictions_path: str) -> str:
    """
    Generate a natural language summary of hashtag trends using LangChain + LLM.
    Thresholds come from params.yaml for consistency.
    """
    # Load params
    params = load_params()
    emerging_th = params["report"]["emerging_threshold"]
    decaying_th = params["report"]["decaying_threshold"]

    # Load predictions
    df = pd.read_csv(predictions_path)

    summaries = []
    for _, row in df.iterrows():
        hashtag = row.get("hashtags", "#unknown")
        mentions = row.get("pred_mentions", 0)
        prev_mentions = row.get("mentions", mentions)  # fallback if no prev data

        # Compute rate of change
        if prev_mentions > 0:
            roc = (mentions - prev_mentions) / prev_mentions
        else:
            roc = 0

        if roc >= emerging_th:
            trend = "Emerging"
        elif roc <= decaying_th:
            trend = "Decaying"
        else:
            trend = "Peaking"

        summaries.append(
            f"Hashtag {hashtag}: {trend} trend (mentions={mentions}, ROC={roc:.2%})"
        )

    # Use LangChain LLM to refine into readable report
    llm = OpenAI(temperature=0)
    prompt = (
        "Summarize the following hashtag trend insights into a clear report:\n\n"
        + "\n".join(summaries)
    )
    refined_report = llm(prompt)

    return refined_report