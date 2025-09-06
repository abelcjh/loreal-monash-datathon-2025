def generate_trend_report(hashtag, metrics, emerging_threshold=0.05, decaying_threshold=-0.05):
    """
    Classify hashtag trend based on predicted rate of change.
    """
    mentions = metrics["mentions"]
    if len(mentions) < 2:
        return f"Hashtag {hashtag}: insufficient data"

    # Rate of change = (latest - previous) / previous
    roc = (mentions[-1] - mentions[-2]) / max(1, mentions[-2])

    if roc >= emerging_threshold:
        trend = "Emerging"
    elif roc <= decaying_threshold:
        trend = "Decaying"
    else:
        trend = "Peaking"

    return f"Hashtag {hashtag}: {trend} trend (ROC={roc:.2%})"