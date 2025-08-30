import os
import json
import asyncio
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Load API keys from .env
from dotenv import load_dotenv
load_dotenv()

TWITTER_BEARER = os.getenv("TWITTER_BEARER")

app = FastAPI()

# Sentiment & emotion models
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

class TweetQuery(BaseModel):
    query: str
    max_results: int = 10
    analyze: bool = True

@app.post("/tools/search_tweets")
def search_tweets(q: TweetQuery):
    headers = {"Authorization": f"Bearer {TWITTER_BEARER}"}
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {"query": q.query, "max_results": min(q.max_results, 50), "tweet.fields": "text,created_at"}

    resp = requests.get(url, headers=headers, params=params)
    tweets = resp.json().get("data", [])

    results = []
    for t in tweets:
        analysis = {}
        if q.analyze:
            analysis["sentiment"] = sentiment_pipeline(t["text"])[0]
            analysis["emotion"] = emotion_pipeline(t["text"])[0]
        results.append({
            "id": t["id"],
            "text": t["text"],
            "created_at": t["created_at"],
            **analysis
        })
    return {"query": q.query, "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
