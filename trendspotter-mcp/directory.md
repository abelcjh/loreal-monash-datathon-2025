trendspotter-mcp/
│── README.md
│── requirements.txt
│── .env.example
│
├── data/                          # Sample datasets, mock trend data
│   ├── sample_tweets.json
│   ├── sample_tiktok_audio.wav
│   └── ...
│
├── mcp_servers/                   # MCP connectors for platforms
│   ├── twitter_server.py
│   ├── tiktok_server.py
│   ├── instagram_server.py
│   └── reddit_server.py
│
├── pipeline/                      # Core LangChain AI pipeline
│   ├── __init__.py
│   ├── agent.py                   # LangChain agent with MCP tools
│   ├── text_analysis.py           # Hugging Face sentiment, keywords
│   ├── audio_analysis.py          # PyTorch/TensorFlow audio models
│   ├── segmentation.py            # Audience (Gen Z vs Millennials)
│   └── decay_model.py             # Trend decay forecasting
│
├── workflows/                     # n8n workflow JSONs
│   ├── trend_scan.json
│   ├── alert_to_slack.json
│   └── store_results.json
│
├── dashboard/                     # Streamlit/Gradio dashboard
│   ├── app.py
│   ├── components/
│   │   ├── charts.py
│   │   └── trend_cards.py
│   └── assets/
│       └── logo.png
│
├── notebooks/                     # Jupyter notebooks for experiments
│   ├── huggingface_experiments.ipynb
│   ├── pytorch_audio_model.ipynb
│   └── decay_forecast.ipynb
│
└── deployment/                    # Docker & deployment configs
    ├── Dockerfile
    ├── docker-compose.yml
    └── n8n_docker.yml
