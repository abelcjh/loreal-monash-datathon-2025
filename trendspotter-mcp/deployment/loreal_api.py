import os
import time
import json
import hmac
import hashlib
import logging
import pathlib
import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import requests
import tweepy
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --------- CONFIG ---------
OUTPUT_DIR = pathlib.Path("trendspotter_out")
OUTPUT_DIR.mkdir(exist_ok=True)
DEFAULT_MAX_ITEMS = 300

# X (Twitter)
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")

# Meta Graph (Facebook + Instagram)
META_TOKEN = os.getenv("META_LONG_LIVED_TOKEN")
FB_PAGE_ID = os.getenv("FB_PAGE_ID")
IG_BUSINESS_USER_ID = os.getenv("IG_BUSINESS_USER_ID")
META_BASE = "https://graph.facebook.com/v18.0"

# TikTok Business (optional) or Apify fallback
TIKTOK_ACCESS_TOKEN = os.getenv("TIKTOK_BUSINESS_ACCESS_TOKEN")
TIKTOK_APP_ID = os.getenv("TIKTOK_BUSINESS_APP_ID")
APIFY_TOKEN = os.getenv("APIFY_TOKEN")
APIFY_TIKTOK_ACTOR = os.getenv("APIFY_TIKTOK_ACTOR", "apify/tiktok-scraper")


# --------- UNIFIED SCHEMA ---------
@dataclass
class SocialReview:
    platform: str                  # "instagram" | "facebook" | "x" | "tiktok"
    source_type: str               # "post" | "comment" | "rating" | "video"
    id: str                        # platform object id
    parent_id: Optional[str]       # e.g., comment’s post id
    author: Optional[str]
    author_id: Optional[str]
    text: Optional[str]
    rating: Optional[float]        # for FB ratings if available
    like_count: Optional[int]
    comment_count: Optional[int]
    share_count: Optional[int]
    view_count: Optional[int]
    url: Optional[str]
    created_at: Optional[str]      # ISO8601
    extra: Dict[str, Any]


# --------- HELPERS ---------
def iso(dt_str_or_ts) -> Optional[str]:
    if not dt_str_or_ts:
        return None
    if isinstance(dt_str_or_ts, (int, float)):
        return dt.datetime.utcfromtimestamp(dt_str_or_ts).isoformat() + "Z"
    try:
        # pass-through if already ISO-ish
        return dt.datetime.fromisoformat(dt_str_or_ts.replace("Z", "+00:00")).isoformat() + "Z"
    except Exception:
        return None


def save_records(name: str, records: List[SocialReview]) -> None:
    if not records:
        logging.warning(f"No records to save for {name}")
        return
    df = pd.DataFrame([asdict(r) for r in records])
    csv_path = OUTPUT_DIR / f"{name}.csv"
    parquet_path = OUTPUT_DIR / f"{name}.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    logging.info(f"Saved {len(df)} rows → {csv_path} & {parquet_path}")


# --------- X (Twitter) ---------
class XClient:
    def __init__(self, bearer_token: str):
        if not bearer_token:
            raise ValueError("X_BEARER_TOKEN not set")
        self.client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

    @retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(5))
    def search(self, query: str, max_items: int = DEFAULT_MAX_ITEMS, start_time: Optional[str] = None,
               end_time: Optional[str] = None) -> List[SocialReview]:
        logging.info(f"[X] Query: {query}")
        # exclude retweets/replies for cleaner “reviews”
        q = f"({query}) -is:retweet lang:en"
        fields = ["created_at", "public_metrics", "lang"]
        expansions = ["author_id"]
        users_fields = ["username", "name"]

        paginator = tweepy.Paginator(
            self.client.search_recent_tweets,
            query=q,
            tweet_fields=fields,
            expansions=expansions,
            user_fields=users_fields,
            start_time=start_time,
            end_time=end_time,
            max_results=100
        )

        id_to_user = {}
        out = []
        for resp in paginator:
            if resp.includes and "users" in resp.includes:
                for u in resp.includes["users"]:
                    id_to_user[u.id] = u

            for t in resp.data or []:
                metrics = t.data.get("public_metrics", {})
                user = id_to_user.get(t.author_id)
                url = f"https://x.com/{user.username}/status/{t.id}" if user else None

                out.append(SocialReview(
                    platform="x",
                    source_type="post",
                    id=str(t.id),
                    parent_id=None,
                    author=user.username if user else None,
                    author_id=str(t.author_id) if t.author_id else None,
                    text=t.text,
                    rating=None,
                    like_count=metrics.get("like_count"),
                    comment_count=metrics.get("reply_count"),
                    share_count=metrics.get("retweet_count"),
                    view_count=metrics.get("impression_count"),
                    url=url,
                    created_at=iso(str(t.created_at)),
                    extra={"lang": t.data.get("lang")}
                ))
                if len(out) >= max_items:
                    return out
        return out


# --------- FACEBOOK (Page posts/comments + ratings if permitted) ---------
class FacebookClient:
    def __init__(self, token: str, page_id: str):
        if not token or not page_id:
            raise ValueError("META_LONG_LIVED_TOKEN or FB_PAGE_ID missing")
        self.token = token
        self.page_id = page_id

    def _get(self, path, params=None):
        params = params or {}
        params["access_token"] = self.token
        r = requests.get(f"{META_BASE}/{path}", params=params, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Facebook API error {r.status_code}: {r.text[:300]}")
        return r.json()

    @retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(5))
    def page_posts_and_comments(self, max_items=DEFAULT_MAX_ITEMS) -> List[SocialReview]:
        fields = "id,message,created_time,permalink_url,shares,comments.summary(true),likes.summary(true)"
        out = []
        feed = self._get(f"{self.page_id}/feed", params={"fields": fields, "limit": 100})
        while True:
            for p in feed.get("data", []):
                like_count = p.get("likes", {}).get("summary", {}).get("total_count")
                comment_count = p.get("comments", {}).get("summary", {}).get("total_count")
                share_count = p.get("shares", {}).get("count")
                out.append(SocialReview(
                    platform="facebook",
                    source_type="post",
                    id=p["id"],
                    parent_id=None,
                    author=None,
                    author_id=None,
                    text=p.get("message"),
                    rating=None,
                    like_count=like_count,
                    comment_count=comment_count,
                    share_count=share_count,
                    view_count=None,
                    url=p.get("permalink_url"),
                    created_at=iso(p.get("created_time")),
                    extra={}
                ))
                # Fetch top-level comments
                comments = self._get(f"{p['id']}/comments", params={
                    "filter": "toplevel",
                    "summary": "true",
                    "limit": 100,
                    "fields": "id,from,message,created_time,like_count"
                })
                for c in comments.get("data", []):
                    out.append(SocialReview(
                        platform="facebook",
                        source_type="comment",
                        id=c["id"],
                        parent_id=p["id"],
                        author=(c.get("from") or {}).get("name"),
                        author_id=(c.get("from") or {}).get("id"),
                        text=c.get("message"),
                        rating=None,
                        like_count=c.get("like_count"),
                        comment_count=None,
                        share_count=None,
                        view_count=None,
                        url=None,
                        created_at=iso(c.get("created_time")),
                        extra={}
                    ))
                if len(out) >= max_items:
                    return out

            paging = feed.get("paging", {}).get("next")
            if not paging:
                break
            feed = requests.get(paging, timeout=30).json()
        return out

    # Optional: Page ratings (requires extra permissions; may be limited/region-dependent)
    def page_ratings(self, max_items=200) -> List[SocialReview]:
        try:
            ratings = self._get(f"{self.page_id}/ratings", params={
                "fields": "review_text,rating,created_time,recommendation_type,reviewer",
                "limit": 100
            })
        except Exception as e:
            logging.warning(f"[Facebook] Ratings not accessible: {e}")
            return []
        out = []
        while True:
            for r in ratings.get("data", []):
                out.append(SocialReview(
                    platform="facebook",
                    source_type="rating",
                    id=f"rating:{r.get('reviewer',{}).get('id','unknown')}:{r.get('created_time')}",
                    parent_id=self.page_id,
                    author=(r.get("reviewer") or {}).get("name"),
                    author_id=(r.get("reviewer") or {}).get("id"),
                    text=r.get("review_text"),
                    rating=float(r.get("rating")) if r.get("rating") is not None else None,
                    like_count=None,
                    comment_count=None,
                    share_count=None,
                    view_count=None,
                    url=None,
                    created_at=iso(r.get("created_time")),
                    extra={"recommendation_type": r.get("recommendation_type")}
                ))
                if len(out) >= max_items:
                    return out
            paging = ratings.get("paging", {}).get("next")
            if not paging:
                break
            ratings = requests.get(paging, timeout=30).json()
        return out


# --------- INSTAGRAM (Business) via Graph API: hashtag → media → comments ---------
class InstagramClient:
    def __init__(self, token: str, ig_user_id: str):
        if not token or not ig_user_id:
            raise ValueError("META_LONG_LIVED_TOKEN or IG_BUSINESS_USER_ID missing")
        self.token = token
        self.ig_user_id = ig_user_id

    def _get(self, path, params=None):
        params = params or {}
        params["access_token"] = self.token
        r = requests.get(f"{META_BASE}/{path}", params=params, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Instagram API error {r.status_code}: {r.text[:300]}")
        return r.json()

    def hashtag_id(self, hashtag: str) -> Optional[str]:
        res = self._get("ig_hashtag_search", params={"user_id": self.ig_user_id, "q": hashtag.strip("#")})
        data = res.get("data", [])
        return data[0]["id"] if data else None

    @retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(5))
    def hashtag_recent_media(self, hashtag: str, max_items=DEFAULT_MAX_ITEMS) -> List[SocialReview]:
        hid = self.hashtag_id(hashtag)
        if not hid:
            logging.warning(f"[Instagram] Hashtag not found: {hashtag}")
            return []
        params = {
            "user_id": self.ig_user_id,
            "fields": "id,caption,media_type,media_url,permalink,timestamp,like_count,comments_count,owner",
            "limit": 50
        }
        out = []
        resp = self._get(f"{hid}/recent_media", params=params)
        while True:
            for m in resp.get("data", []):
                out.append(SocialReview(
                    platform="instagram",
                    source_type="post",
                    id=m["id"],
                    parent_id=None,
                    author=(m.get("owner") or {}).get("id"),
                    author_id=(m.get("owner") or {}).get("id"),
                    text=m.get("caption"),
                    rating=None,
                    like_count=m.get("like_count"),
                    comment_count=m.get("comments_count"),
                    share_count=None,
                    view_count=None,
                    url=m.get("permalink"),
                    created_at=iso(m.get("timestamp")),
                    extra={"media_type": m.get("media_type"), "media_url": m.get("media_url")}
                ))
                # Fetch comments
                try:
                    comments = self._get(f"{m['id']}/comments", params={
                        "fields": "id,text,timestamp,like_count,username",
                        "limit": 100
                    })
                    for c in comments.get("data", []):
                        out.append(SocialReview(
                            platform="instagram",
                            source_type="comment",
                            id=c["id"],
                            parent_id=m["id"],
                            author=c.get("username"),
                            author_id=None,
                            text=c.get("text"),
                            rating=None,
                            like_count=c.get("like_count"),
                            comment_count=None,
                            share_count=None,
                            view_count=None,
                            url=None,
                            created_at=iso(c.get("timestamp")),
                            extra={}
                        ))
                except Exception as e:
                    logging.warning(f"[Instagram] comments error for {m['id']}: {e}")

                if len(out) >= max_items:
                    return out

            paging = resp.get("paging", {}).get("next")
            if not paging:
                break
            resp = requests.get(paging, timeout=30).json()
        return out


# --------- TIKTOK: Business API (if you have it) OR Apify fallback ---------
class TikTokClient:
    def __init__(self, access_token: Optional[str] = None, app_id: Optional[str] = None,
                 apify_token: Optional[str] = None, apify_actor: Optional[str] = None):
        self.access_token = access_token
        self.app_id = app_id
        self.apify_token = apify_token
        self.apify_actor = apify_actor

    # ---- Option A: TikTok Business API stub (requires onboarding; endpoints vary by scope)
    # This function is illustrative; replace with your approved endpoints.
    def business_search_hashtag(self, hashtag: str, max_items=DEFAULT_MAX_ITEMS) -> List[SocialReview]:
        if not (self.access_token and self.app_id):
            logging.info("[TikTok] Business API creds not set; skipping Business path.")
            return []
        # TODO: implement with your approved endpoints/scopes
        logging.info("[TikTok] Business API integration pending — returning [].")
        return []

    # ---- Option B: Apify fallback (serverless scraper over HTTP API)
    @retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(5))
    def apify_hashtag(self, hashtag: str, max_items=DEFAULT_MAX_ITEMS) -> List[SocialReview]:
        if not (self.apify_token and self.apify_actor):
            logging.info("[TikTok] APIFY_TOKEN or actor not set; skipping Apify path.")
            return []

        # Start an actor run
        start_url = f"https://api.apify.com/v2/acts/{self.apify_actor}/runs?token={self.apify_token}"
        payload = {
            "input": {
                "hashtags": [hashtag.strip("#")],
                "resultsLimit": min(max_items, 500),
                "downloadSubtitles": False
            }
        }
        start = requests.post(start_url, json=payload, timeout=60).json()
        run_id = start.get("data", {}).get("id")
        if not run_id:
            raise RuntimeError(f"Apify start error: {start}")

        # Poll until finished
        while True:
            run = requests.get(
                f"https://api.apify.com/v2/actor-runs/{run_id}",
                params={"token": self.apify_token},
                timeout=30
            ).json()
            status = run.get("data", {}).get("status")
            if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                if status != "SUCCEEDED":
                    raise RuntimeError(f"Apify run status={status}")
                break
            time.sleep(5)

        # Fetch dataset items
        dataset_id = run.get("data", {}).get("defaultDatasetId")
        ds = requests.get(
            f"https://api.apify.com/v2/datasets/{dataset_id}/items",
            params={"token": self.apify_token, "clean": "true"},
            timeout=60
        ).json()

        out = []
        for item in ds:
            stats = item.get("stats") or {}
            out.append(SocialReview(
                platform="tiktok",
                source_type="video",
                id=str(item.get("id") or item.get("videoId")),
                parent_id=None,
                author=item.get("authorUniqueId") or item.get("authorName"),
                author_id=str(item.get("authorId")) if item.get("authorId") else None,
                text=item.get("desc"),
                rating=None,
                like_count=stats.get("diggCount"),
                comment_count=stats.get("commentCount"),
                share_count=stats.get("shareCount"),
                view_count=stats.get("playCount"),
                url=item.get("url") or item.get("shareUrl"),
                created_at=iso(item.get("createTime")),
                extra={"music": item.get("musicMeta", {}).get("musicName")}
            ))
            if len(out) >= max_items:
                break
        return out


# --------- ORCHESTRATION ---------
def collect_all(
    hashtags: List[str],
    max_items_per_source: int = 300,
    include_facebook_ratings: bool = False
) -> pd.DataFrame:
    records: List[SocialReview] = []

    # X
    if X_BEARER_TOKEN:
        xcli = XClient(X_BEARER_TOKEN)
        for tag in hashtags:
            try:
                recs = xcli.search(query=tag, max_items=max_items_per_source)
                records.extend(recs)
            except Exception as e:
                logging.error(f"[X] error for {tag}: {e}")

    # Facebook
    if META_TOKEN and FB_PAGE_ID:
        fb = FacebookClient(META_TOKEN, FB_PAGE_ID)
        try:
            records.extend(fb.page_posts_and_comments(max_items=max_items_per_source))
        except Exception as e:
            logging.error(f"[Facebook] posts/comments error: {e}")
        if include_facebook_ratings:
            try:
                records.extend(fb.page_ratings(max_items=max_items_per_source))
            except Exception as e:
                logging.error(f"[Facebook] ratings error: {e}")

    # Instagram
    if META_TOKEN and IG_BUSINESS_USER_ID:
        ig = InstagramClient(META_TOKEN, IG_BUSINESS_USER_ID)
        for tag in hashtags:
            try:
                records.extend(ig.hashtag_recent_media(hashtag=tag, max_items=max_items_per_source))
            except Exception as e:
                logging.error(f"[Instagram] error for {tag}: {e}")

    # TikTok
    tiktok = TikTokClient(
        access_token=TIKTOK_ACCESS_TOKEN,
        app_id=TIKTOK_APP_ID,
        apify_token=APIFY_TOKEN,
        apify_actor=APIFY_TIKTOK_ACTOR
    )
    for tag in hashtags:
        try:
            # prefer Business API if configured, else Apify
            recs = tiktok.business_search_hashtag(tag, max_items=max_items_per_source)
            if not recs:
                recs = tiktok.apify_hashtag(tag, max_items=max_items_per_source)
            records.extend(recs)
        except Exception as e:
            logging.error(f"[TikTok] error for {tag}: {e}")

    # Save unified output
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"social_reviews_{ts}"
    save_records(name, records)

    return pd.DataFrame([asdict(r) for r in records])


if __name__ == "__main__":
    # Example run
    hashtags = ["#veganlipstick", "#sulfatefree", "#crueltyfree", "#kbeauty"]
    df = collect_all(hashtags, max_items_per_source=200, include_facebook_ratings=False)
    print(df.head())
    print(f"Total rows: {len(df)}")
