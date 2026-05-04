# sentiment.py - NLP sentiment scoring using VADER + TextBlob

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from utils import get_logger, clean_text, NEWS_CSV, ensure_nltk_resources

ensure_nltk_resources()
log = get_logger("sentiment")

# extra finance terms that VADER doesn't know about natively
FINANCE_LEXICON: dict[str, float] = {
    "rally": 2.5, "surge": 2.5, "soar": 2.5,
    "bullish": 2.5, "beat": 2.0, "outperform": 2.0,
    "upgrade": 2.0, "breakout": 2.0, "strong": 1.5,
    "profit": 1.5, "growth": 1.5, "gain": 1.5,
    "positive": 1.5, "buy": 1.0, "acquire": 1.0,
    "crash": -3.0, "plunge": -2.5, "tank": -2.5,
    "bearish": -2.5, "miss": -2.0, "downgrade": -2.0,
    "loss": -1.5, "decline": -1.5, "fall": -1.5,
    "sell": -1.0, "layoff": -1.5, "lawsuit": -1.5,
    "bankruptcy": -3.0, "fraud": -2.5, "recall": -2.0,
}

_vader = SentimentIntensityAnalyzer()
_vader.lexicon.update(FINANCE_LEXICON)


def vader_scores(text: str) -> dict:
    if not text or not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    return _vader.polarity_scores(text)


def textblob_scores(text: str) -> dict:
    if not text or not text.strip():
        return {"polarity": 0.0, "subjectivity": 0.0}
    blob = TextBlob(text)
    return {"polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity}


def composite_score(text: str) -> float:
    """Weighted blend: 70% VADER (finance-tuned) + 30% TextBlob."""
    v = vader_scores(text)["compound"]
    t = textblob_scores(text)["polarity"]
    return round(0.7 * v + 0.3 * t, 4)


def classify_sentiment(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"


def add_sentiment(df: pd.DataFrame, text_col: str = "headline") -> pd.DataFrame:
    """Score every row and append compound/polarity/subjectivity/label columns."""
    if df.empty:
        return df

    log.info("Computing sentiment for %d rows ...", len(df))

    # use pre-cleaned text if available
    src = df["clean_headline"].fillna(df[text_col]) if "clean_headline" in df.columns else df[text_col]

    vader_rows = src.apply(vader_scores)
    tb_rows = src.apply(textblob_scores)

    df["compound"] = vader_rows.apply(lambda x: x["compound"])
    df["vader_pos"] = vader_rows.apply(lambda x: x["pos"])
    df["vader_neg"] = vader_rows.apply(lambda x: x["neg"])
    df["vader_neu"] = vader_rows.apply(lambda x: x["neu"])
    df["polarity"] = tb_rows.apply(lambda x: x["polarity"])
    df["subjectivity"] = tb_rows.apply(lambda x: x["subjectivity"])
    df["composite"] = src.apply(composite_score)
    df["sentiment_label"] = df["composite"].apply(classify_sentiment)

    pos = (df["sentiment_label"] == "Positive").sum()
    neg = (df["sentiment_label"] == "Negative").sum()
    neu = (df["sentiment_label"] == "Neutral").sum()
    log.info("Distribution: %d positive, %d neutral, %d negative", pos, neu, neg)

    return df


def generate_alerts(df: pd.DataFrame, threshold: float = 0.6) -> list[dict]:
    """Flag any articles with extreme sentiment above threshold."""
    alerts = []
    if "composite" not in df.columns:
        return alerts

    strong = df[df["composite"].abs() >= threshold].copy()
    for _, row in strong.iterrows():
        direction = "🟢 BULLISH" if row["composite"] > 0 else "🔴 BEARISH"
        alerts.append({
            "direction": direction,
            "headline": row.get("headline", ""),
            "date": row.get("date", ""),
            "score": row["composite"],
            "label": row.get("sentiment_label", ""),
        })
    return alerts


def load_and_score_news(csv_path=None, save: bool = True) -> pd.DataFrame:
    """Quick helper: read CSV, score it, optionally save back."""
    path = csv_path or NEWS_CSV
    df = pd.read_csv(path)
    df = add_sentiment(df)
    if save:
        df.to_csv(path, index=False)
        log.info("Scored news saved to %s", path)
    return df


if __name__ == "__main__":
    samples = [
        "Apple surges to record high after strong quarterly earnings beat",
        "Tesla crashes 15% amid fears of rising interest rates and slowing EV demand",
        "Markets remain flat as investors await Federal Reserve decision",
    ]
    for s in samples:
        sc = composite_score(s)
        print(f"{classify_sentiment(sc):10s} ({sc:+.4f})  ->  {s[:70]}")
