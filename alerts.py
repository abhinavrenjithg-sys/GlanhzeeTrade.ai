# alerts.py - watches for strong sentiment signals and maintains an alert log

import time
import schedule
import pandas as pd
from datetime import datetime
from pathlib import Path

from utils import get_logger, NEWS_CSV
from sentiment import generate_alerts, add_sentiment, load_and_score_news
from data import collect_news

log = get_logger("alerts")

ALERT_THRESHOLD = 0.6
_ALERT_HISTORY: list[dict] = []  # in-memory, resets on app restart


def check_alerts(ticker: str) -> list[dict]:
    """Pull fresh news, score it, return any strong sentiment hits (deduped)."""
    global _ALERT_HISTORY
    seen = {a["headline"] for a in _ALERT_HISTORY}

    try:
        news_df = collect_news(ticker, save=False)
        if news_df.empty:
            return []
        news_df = add_sentiment(news_df)
        alerts = generate_alerts(news_df, threshold=ALERT_THRESHOLD)
    except Exception as exc:
        log.error("Alert check failed: %s", exc)
        return []

    new_alerts = [a for a in alerts if a["headline"] not in seen]
    for a in new_alerts:
        a["checked_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _ALERT_HISTORY.append(a)
        log.info("ALERT [%s] %s  score=%.3f", a["direction"], a["headline"][:60], a["score"])

    return new_alerts


def get_all_alerts() -> list[dict]:
    return list(reversed(_ALERT_HISTORY))


def start_scheduler(ticker: str, interval_minutes: int = 30):
    """Blocking scheduler loop - run this in a background thread."""
    log.info("Alert scheduler starting, checking every %d min for %s", interval_minutes, ticker)
    schedule.every(interval_minutes).minutes.do(check_alerts, ticker=ticker)
    while True:
        schedule.run_pending()
        time.sleep(60)
