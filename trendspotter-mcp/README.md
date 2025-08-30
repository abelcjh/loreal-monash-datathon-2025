mcp_servers/

Each server wraps platform APIs (Twitter, TikTok, IG) into a standard MCP interface.

Example: twitter_server.py → listens on localhost, serves get_trends, get_hashtags.

pipeline/

LangChain agent loads MCP tools and orchestrates calls.

Hugging Face (transformers) for embeddings, sentiment.

PyTorch/TensorFlow for audio classification.

Decay forecasting → time-series model (Prophet or PyTorch LSTM).

workflows/

n8n JSONs you can directly import:

Run every hour → call MCP servers → push to pipeline → store in DB.

Alert Slack when a new “rising” trend is found.

dashboard/

Streamlit or Gradio dashboard:

Charts: Trend growth rate, audience segments.

Cards: “Hot trends”, “Stable”, “Decaying”.

Filter by category: Beauty, Fitness, Lifestyle.

deployment/

Dockerfile for MCP + LangChain pipeline.

docker-compose.yml runs:

pipeline (Python service)

mcp_servers

dashboard

optional n8n container