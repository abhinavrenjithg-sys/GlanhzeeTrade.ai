# utils.py - shared helpers and config

import os
import logging
import re
import string
from datetime import datetime, timedelta
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

for _dir in (DATA_DIR, MODEL_DIR, LOGS_DIR):
    _dir.mkdir(exist_ok=True)

NEWS_CSV = DATA_DIR / "news_data.csv"
STOCK_CSV = DATA_DIR / "stock_data.csv"
MERGED_CSV = DATA_DIR / "merged_dataset.csv"


def get_logger(name: str) -> logging.Logger:
    """Sets up a logger with console + daily file output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_file = LOGS_DIR / f"{datetime.now():%Y-%m-%d}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def ensure_nltk_resources():
    """Downloads any missing NLTK packages quietly."""
    needed = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("sentiment/vader_lexicon", "vader_lexicon"),
    ]
    for path, pkg in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_stemmer = PorterStemmer()
_STOPWORDS: set[str] | None = None

def _get_stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        ensure_nltk_resources()
        _STOPWORDS = set(stopwords.words("english"))
        # keep finance-relevant negators
        _STOPWORDS -= {"no", "not", "nor", "up", "down"}
    return _STOPWORDS


def clean_text(text: str, stem: bool = False) -> str:
    """Lowercase, strip urls/html, remove punctuation and stopwords."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    sw = _get_stopwords()
    tokens = [t for t in tokens if t not in sw and len(t) > 1]

    if stem:
        tokens = [_stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


def last_n_days(n: int) -> tuple[str, str]:
    """Returns (start, end) date strings for the past n days."""
    end = datetime.today()
    start = end - timedelta(days=n)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def normalise_date(raw: str) -> str | None:
    """Tries common date formats, returns YYYY-MM-DD or None."""
    formats = [
        "%B %d, %Y", "%b %d, %Y", "%d %B %Y",
        "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
    ]
    raw = raw.strip()
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None
