# Stock Market Sentiment Analyzer — StockSentinel 📈

**An end-to-end AI-powered platform that predicts stock price direction by fusing real-time financial news sentiment with time-series technical indicators.**

> Resume-worthy · Production-ready · Placement-showcase quality

---

## 🗂️ Project Structure

```
stock-sentiment-analyzer/
│
├── app.py                  # Streamlit dashboard (main entry point)
├── data.py                 # News scraping + yfinance stock data
├── sentiment.py            # VADER + TextBlob NLP pipeline
├── features.py             # Feature engineering (RSI, MACD, lags…)
├── model.py                # ML training: Logistic Regression, RF, LSTM
├── visualize.py            # Plotly chart builders
├── alerts.py               # Real-time sentiment alert system
├── utils.py                # Shared helpers, logging, text cleaning
├── generate_sample_data.py # Offline sample dataset generator
├── requirements.txt
└── README.md
```

---

## ⚙️ Quick Setup

### 1 — Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **TensorFlow note:** TF 2.15 requires Python 3.8–3.11.  
> If you skip TF, the LSTM model is gracefully skipped and the best of LR/RF is used.

### 3 — Generate offline sample data (no internet needed)

```bash
python generate_sample_data.py
```

### 4 — Launch the dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🚀 How to Use

| Step | Action |
|------|--------|
| 1 | Enter a stock ticker in the sidebar (`AAPL`, `TSLA`, `MSFT`, `RELIANCE.NS`, etc.) |
| 2 | Choose a historical period (1 month → 2 years) |
| 3 | Click **🚀 Fetch & Analyze** — scrapes live news + downloads OHLCV data |
| 4 | Click **🧠 Train Models** — trains LR, RF, and LSTM and compares them |
| 5 | View prediction, charts, and alerts across the 5 dashboard tabs |

---

## 📊 Features

### Data Sources
| Source | Data |
|--------|------|
| Yahoo Finance RSS | Real-time financial headlines |
| Economic Times | Indian & global market news |
| Moneycontrol | Market news & analysis |
| yfinance | Historical OHLCV prices |

### NLP Pipeline
- **VADER** with finance-domain lexicon injection (rally, crash, upgrade…)
- **TextBlob** polarity + subjectivity
- **Composite score** = 0.7 × VADER + 0.3 × TextBlob
- Labels: `Positive` (≥ 0.05) · `Neutral` (−0.05 to 0.05) · `Negative` (≤ −0.05)

### Technical Indicators
`SMA-5/10/20` · `EMA-12/26` · `RSI-14` · `MACD` · `Bollinger Bands` · `Daily Return` · `Volatility (σ)` · `Volume Ratio`

### ML Models

| Model | Type | Notes |
|-------|------|-------|
| Logistic Regression | Baseline | Interpretable, fast |
| Random Forest | Ensemble | Feature importance, robust |
| LSTM | Deep Learning | Captures sequential price patterns |

- Evaluation: Accuracy, Precision, Recall, F1 (TimeSeriesSplit)
- Best model auto-selected by F1 score
- Saved with `pickle` / Keras `.keras` format

### Alert System
- Fires on `|composite score| ≥ 0.6`
- Scheduled checks every 30 minutes
- Color-coded 🟢 Bullish / 🔴 Bearish cards in sidebar

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `data/news_data.csv` | Scraped headlines with sentiment scores |
| `data/stock_data.csv` | OHLCV price history |
| `data/merged_dataset.csv` | Stock + daily aggregated sentiment |
| `data/features.csv` | Full feature matrix (ML-ready) |
| `models/logistic_regression.pkl` | Trained LR model |
| `models/random_forest.pkl` | Trained RF model |
| `models/lstm_model.keras` | Trained LSTM weights |
| `models/scaler.pkl` | Feature scaler (StandardScaler) |
| `models/model_meta.pkl` | Best model name + metadata |
| `logs/YYYY-MM-DD.log` | Daily application logs |

---

## 🛠️ Running Individual Modules

```bash
# Test news scraping
python data.py AAPL

# Test sentiment scoring
python sentiment.py

# Build features from existing CSVs
python features.py

# Train all models
python model.py

# Generate synthetic sample dataset
python generate_sample_data.py
```

---

## 🌐 Supported Tickers

Any Yahoo Finance ticker works:
- **US:** `AAPL`, `TSLA`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`
- **India:** `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`
- **Crypto:** `BTC-USD`, `ETH-USD`
- **ETFs:** `SPY`, `QQQ`

---

## 🔧 Tech Stack

| Layer | Tech |
|-------|------|
| Dashboard | Streamlit |
| Scraping | BeautifulSoup4, Requests |
| NLP | VADER, TextBlob, NLTK |
| Stock Data | yfinance |
| ML | scikit-learn, TensorFlow/Keras |
| Visualization | Plotly, Matplotlib, Seaborn |
| Utilities | pandas, numpy, schedule |

---

## 💡 Resume Talking Points

- "Designed a multi-source NLP pipeline combining VADER and TextBlob with a custom finance-domain lexicon achieving more accurate bullish/bearish classification than out-of-box tools."
- "Engineered 27 features including RSI, MACD, Bollinger Bands, and sentiment lag variables; trained and compared Logistic Regression, Random Forest, and LSTM models using time-series cross-validation."
- "Built an end-to-end Streamlit dashboard with real-time news scraping, live ML inference, and a configurable alert system for strong sentiment signals."

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is **not financial advice**. Do not use it to make real investment decisions.

---

*Built with ❤️ — StockSentinel v1.0*
