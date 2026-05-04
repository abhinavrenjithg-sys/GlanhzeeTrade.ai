# data.py - news scraping, stock data from yfinance, and merging

import time
import os
from dotenv import load_dotenv
load_dotenv()
import random
import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

from utils import get_logger, clean_text, normalise_date, NEWS_CSV, STOCK_CSV, MERGED_CSV

log = get_logger("data")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# polite crawl delay so we don't get rate-limited
MIN_DELAY = 1.0
MAX_DELAY = 2.5


def _polite_get(url, timeout=10):
    """GET with retries and random delay between requests."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            return resp
        except requests.RequestException as exc:
            log.warning("Attempt %d failed for %s: %s", attempt + 1, url, exc)
            time.sleep(2 ** attempt)
    return None


def scrape_yahoo_rss(ticker, max_articles=50):
    """Pulls headlines from Yahoo Finance RSS for a ticker."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    resp = _polite_get(url)
    if resp is None:
        log.error("Couldn't reach Yahoo RSS for %s", ticker)
        return pd.DataFrame()

    soup = BeautifulSoup(resp.content, "xml")
    items = soup.find_all("item")
    records = []

    for item in items:
        title = item.find("title")
        pubdate = item.find("pubDate")
        desc = item.find("description")
        link = item.find("link")

        raw_date = pubdate.text.strip() if pubdate else ""
        try:
            parsed = datetime.strptime(raw_date, "%a, %d %b %Y %H:%M:%S %z")
            clean_date = parsed.strftime("%Y-%m-%d")
        except ValueError:
            clean_date = normalise_date(raw_date) or ""

        records.append({
            "headline": title.text.strip() if title else "",
            "date": clean_date,
            "source": "Yahoo Finance",
            "content": desc.text.strip() if desc else "",
            "url": link.text.strip() if link else "",
            "ticker": ticker.upper(),
        })

    df = pd.DataFrame(records)
    log.info("Yahoo RSS: %d articles for %s", len(df), ticker)
    return df


def scrape_economic_times(query, max_pages=3):
    """Grabs market news headlines from Economic Times."""
    base = "https://economictimes.indiatimes.com/markets/stocks/news"
    resp = _polite_get(base)
    if resp is None:
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    articles = soup.find_all("div", class_="eachStory")
    records = []

    for art in articles:
        h3 = art.find("h3")
        time_tag = art.find("time")
        p = art.find("p")
        a = art.find("a", href=True)

        headline = h3.text.strip() if h3 else ""
        content = p.text.strip() if p else ""
        raw_date = time_tag.get("data-time", "") if time_tag else ""

        try:
            clean_date = datetime.fromtimestamp(int(raw_date) / 1000).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            clean_date = normalise_date(raw_date) or datetime.today().strftime("%Y-%m-%d")

        href = "https://economictimes.indiatimes.com" + a["href"] if a else ""

        records.append({
            "headline": headline,
            "date": clean_date,
            "source": "Economic Times",
            "content": content,
            "url": href,
            "ticker": query.upper(),
        })

    df = pd.DataFrame(records)
    log.info("Economic Times: %d articles", len(df))
    return df


def scrape_moneycontrol(ticker):
    """Scrapes latest market news from Moneycontrol."""
    url = "https://www.moneycontrol.com/news/business/markets/"
    resp = _polite_get(url)
    if resp is None:
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    articles = soup.find_all("li", class_="clearfix")
    records = []

    for art in articles:
        h2 = art.find("h2")
        span = art.find("span")
        p = art.find("p")
        a = art.find("a", href=True)

        headline = h2.text.strip() if h2 else ""
        content = p.text.strip() if p else ""
        raw_date = span.text.strip() if span else ""
        clean_date = normalise_date(raw_date) or datetime.today().strftime("%Y-%m-%d")

        if not headline:
            continue

        records.append({
            "headline": headline,
            "date": clean_date,
            "source": "Moneycontrol",
            "content": content,
            "url": a["href"] if a else "",
            "ticker": ticker.upper(),
        })

    df = pd.DataFrame(records)
    log.info("Moneycontrol: %d articles", len(df))
    return df


def scrape_newsapi(ticker):
    """Fetches from NewsAPI if NEWSAPI_KEY is set in environment."""
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        return pd.DataFrame()

    log.info("Fetching from NewsAPI for %s...", ticker)
    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&apiKey={api_key}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("NewsAPI failed: %s", exc)
        return pd.DataFrame()

    if data.get("status") != "ok":
        log.warning("NewsAPI error: %s", data.get("message"))
        return pd.DataFrame()

    articles = data.get("articles", [])
    records = []

    for art in articles:
        headline = art.get("title")
        content = art.get("description")
        raw_date = art.get("publishedAt")

        if not headline or headline == "[Removed]":
            continue

        try:
            clean_date = datetime.strptime(raw_date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
        except Exception:
            clean_date = normalise_date(raw_date) or datetime.today().strftime("%Y-%m-%d")

        records.append({
            "headline": headline,
            "date": clean_date,
            "source": art.get("source", {}).get("name", "NewsAPI"),
            "content": content or "",
            "url": art.get("url", ""),
            "ticker": ticker.upper(),
        })

    df = pd.DataFrame(records)
    log.info("NewsAPI: %d articles", len(df))
    return df


def collect_news(ticker, save=True):
    """Aggregates news from all sources, deduplicates, cleans text."""
    log.info("Collecting news for %s...", ticker)
    frames = [
        scrape_yahoo_rss(ticker),
        scrape_economic_times(ticker),
        scrape_moneycontrol(ticker),
        scrape_newsapi(ticker),
    ]

    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    if df.empty:
        log.warning("No news found for %s", ticker)
        return df

    df = df[df["headline"].str.len() > 10]
    df = df.drop_duplicates(subset=["headline"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    df["clean_headline"] = df["headline"].apply(clean_text)
    df["clean_content"] = df["content"].apply(clean_text)

    if save:
        df.to_csv(NEWS_CSV, index=False)
        log.info("News saved -> %s (%d rows)", NEWS_CSV, len(df))

    return df


def fetch_stock_data(ticker, start=None, end=None, period="1y", save=True):
    """Downloads OHLCV from Yahoo Finance via yfinance."""
    log.info("Fetching stock data for %s...", ticker)

    try:
        tkr = yf.Ticker(ticker)
        if start and end:
            df = tkr.history(start=start, end=end)
        else:
            df = tkr.history(period=period)
    except Exception as exc:
        log.error("yfinance error: %s", exc)
        return pd.DataFrame()

    if df.empty:
        log.warning("No stock data for %s", ticker)
        return df

    df = df.reset_index()
    df.rename(columns={"Date": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    cols = ["date", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]]
    df.columns = [c.lower() for c in df.columns]
    df["ticker"] = ticker.upper()

    if save:
        df.to_csv(STOCK_CSV, index=False)
        log.info("Stock data saved -> %s (%d rows)", STOCK_CSV, len(df))

    return df


def merge_datasets(news_df=None, stock_df=None, save=True):
    """Left-joins stock data with daily aggregated sentiment scores."""
    if news_df is None:
        news_df = pd.read_csv(NEWS_CSV)
    if stock_df is None:
        stock_df = pd.read_csv(STOCK_CSV)

    if news_df.empty or stock_df.empty:
        log.warning("Can't merge, one dataframe is empty")
        return pd.DataFrame()

    if "compound" in news_df.columns:
        daily_sent = (
            news_df.groupby("date")["compound"]
            .mean().reset_index()
            .rename(columns={"compound": "avg_sentiment"})
        )
    else:
        daily_sent = news_df[["date"]].drop_duplicates()
        daily_sent["avg_sentiment"] = 0.0

    merged = stock_df.merge(daily_sent, on="date", how="left")
    merged["avg_sentiment"] = merged["avg_sentiment"].fillna(0.0)

    if save:
        merged.to_csv(MERGED_CSV, index=False)
        log.info("Merged data saved -> %s (%d rows)", MERGED_CSV, len(merged))

    return merged


# screener ticker groups
TICKER_GROUPS = {
    "Tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"],
    "Energy": ["XOM", "CVX", "SHEL", "COP"],
    "Finance": ["JPM", "BAC", "V", "MA"],
    "Automotive": ["TSLA", "TM", "F", "GM"]
}


def fetch_screener_data():
    """Grabs 5-day data for all ticker groups and calculates daily change."""
    all_tickers = []
    for group in TICKER_GROUPS.values():
        all_tickers.extend(group)

    log.info("Screener: fetching %d tickers...", len(all_tickers))
    results = []

    for ticker in all_tickers:
        try:
            tkr = yf.Ticker(ticker)
            df = tkr.history(period="5d")
            df = df.dropna()

            if len(df) >= 2:
                close = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2])
                pct_change = ((close - prev_close) / prev_close) * 100
            else:
                continue

            cat = "Other"
            for k, v in TICKER_GROUPS.items():
                if ticker in v:
                    cat = k
                    break

            status = "Profit 🟢" if pct_change > 0 else ("Loss 🔴" if pct_change < 0 else "Flat ⚪")
            results.append({
                "Ticker": ticker,
                "Category": cat,
                "Price": close,
                "Change %": pct_change,
                "Status": status
            })
        except Exception as e:
            log.debug("Screener skip %s: %s", ticker, e)

    return pd.DataFrame(results).sort_values("Change %", ascending=False) if results else pd.DataFrame()


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    news = collect_news(ticker)
    stock = fetch_stock_data(ticker)
    merged = merge_datasets(news, stock)
    print(merged.tail())
