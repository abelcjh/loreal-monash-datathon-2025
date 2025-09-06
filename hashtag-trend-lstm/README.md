📈 Hashtag Trend Forecasting (PyTorch LSTM + Hugging Face + LangChain)

This repo predicts hashtag trends (emerging, peaking, decaying) from YouTube video metadata.
It combines preprocessing, feature engineering, embeddings, LSTM forecasting, and AI-powered reporting.

## ⚙️ Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/hashtag-trend-lstm.git
cd hashtag-trend-lstm
pip install -r requirements.txt
```

## 🗂️ Data
🧹 Preprocessing
Raw → Cleaned

```bash
python src/preprocessing.py data/raw/videos.csv data/processed/cleaned.csv
```

Removes rows without hashtags
Normalizes numeric fields
Extracts hashtags


Cleaned → Hashtag Time Series

```bash
python src/features.py data/processed/cleaned.csv --output_csv data/processed/hashtag_timeseries.csv --freq h
```
Expands per-video rows into hashtag × hour format
Aggregates mentions, views, likes, comments per hashtag per hour
Computes rate-of-change (*_diff) features

### ✂️ Splitting the Dataset
Split into train / val / test / live sets:

bash
python scripts/split_dataset.py data/processed/hashtag_timeseries.csv data/splits

train.csv – model training
val.csv – hyperparameter tuning
test.csv – final offline evaluation
live.csv – held-out for live demo

### 🤖 Training the LSTM
Train the multi-target LSTM model:

``` bash
python -m scripts.train_lstm --train_csv data/splits/train.csv --val_csv data/splits/val.csv --epochs 10 --batch_size 32 --seq_len 24
```
Model checkpoints are saved under models/.

### 🚀 Demo / Deployment
For hackathon demos:

Use the deploy.csv split as live unseen data.

Load the trained model and run inference on this split.

### 📂 Project Structure
bash
Copy code
├── data/
│   ├── raw/                # Original CSVs
│   ├── processed/          # cleaned.csv, hashtag_timeseries.csv
│   └── splits/             # train.csv, val.csv, test.csv, deploy.csv
├── models/                 # Saved LSTM models
├── scripts/
│   └── train_lstm.py       # Training entry point
├── src/
│   ├── preprocessing.py    # Raw → Cleaned
│   ├── features.py         # Cleaned → Hashtag time series
│   ├── split_dataset.py    # Chronological splits
│   ├── utils.py            # Dataloaders, helpers
│   └── lstm_multitarget.py # Model definition
└── README.md

### ✅ End-to-end workflow
Raw CSV → cleaned.csv → hashtag_timeseries.csv → splits → LSTM training → deploy demo.