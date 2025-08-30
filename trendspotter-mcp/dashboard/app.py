import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
import os

st.set_page_config(page_title="TrendSpotter Dashboard", layout="wide")

DB_URI = os.getenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/trends")

@st.cache_data(ttl=60)
def load_data():
    conn = psycopg2.connect(DB_URI)
    df = pd.read_sql("SELECT * FROM trends ORDER BY timestamp DESC LIMIT 500", conn)
    conn.close()
    return df

st.title("📊 TrendSpotter Dashboard")
df = load_data()

if df.empty:
    st.warning("No data yet. Wait for n8n workflow to populate database.")
else:
    hashtags = df["hashtag"].unique()
    choice = st.selectbox("Select Hashtag", hashtags)

    subset = df[df["hashtag"] == choice]

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(subset, x="timestamp", y="volume", title="Tweet Volume Over Time")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sentiment_counts = subset[["positive", "negative", "neutral"]].sum().reset_index()
        sentiment_counts.columns = ["sentiment", "count"]
        fig2 = px.pie(sentiment_counts, names="sentiment", values="count", title="Sentiment Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Emotion Breakdown (Latest Batch)")
    latest = subset.iloc[0]
    emotions = pd.DataFrame(latest["emotions"].items(), columns=["emotion", "count"])
    fig3 = px.bar(emotions, x="emotion", y="count")
    st.plotly_chart(fig3, use_container_width=True)
