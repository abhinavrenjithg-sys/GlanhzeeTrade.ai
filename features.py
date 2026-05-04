# features.py - builds the feature matrix for ML training

import numpy as np
import pandas as pd
from utils import get_logger, MERGED_CSV

log = get_logger("features")


# -- technical indicators --

def add_sma(df, col="close", windows=(5, 10, 20, 50)):
    for w in windows:
        df[f"sma_{w}"] = df[col].rolling(window=w).mean()
    return df

def add_ema(df, col="close", spans=(12, 26)):
    for s in spans:
        df[f"ema_{s}"] = df[col].ewm(span=s, adjust=False).mean()
    return df

def add_rsi(df, col="close", period=14):
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def add_macd(df, col="close"):
    ema12 = df[col].ewm(span=12, adjust=False).mean()
    ema26 = df[col].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df

def add_bollinger(df, col="close", window=20):
    sma = df[col].rolling(window=window).mean()
    std = df[col].rolling(window=window).std()
    df["bb_upper"] = sma + 2 * std
    df["bb_lower"] = sma - 2 * std
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_pct"] = (df[col] - df["bb_lower"]) / (df["bb_width"] + 1e-9)
    return df

def add_volatility(df, col="close", window=10):
    log_ret = np.log(df[col] / df[col].shift(1))
    df["volatility"] = log_ret.rolling(window=window).std()
    return df

def add_price_change(df, col="close"):
    df["daily_return"] = df[col].pct_change()
    df["price_change"] = df[col].diff()
    return df


# -- sentiment-derived features --

def add_sentiment_lags(df, col="avg_sentiment", lags=(1, 2, 3)):
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

def add_sentiment_rolling(df, col="avg_sentiment", windows=(3, 5, 7)):
    for w in windows:
        df[f"{col}_roll{w}"] = df[col].rolling(window=w).mean()
    return df

def add_volume_features(df):
    if "volume" not in df.columns:
        return df
    df["volume_sma5"] = df["volume"].rolling(5).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma5"] + 1)
    df["volume_change"] = df["volume"].pct_change()
    return df


# -- target --

def add_target(df, col="close", lookahead=1):
    """1 if price goes up tomorrow, 0 otherwise."""
    df["target"] = (df[col].shift(-lookahead) > df[col]).astype(int)
    return df


# columns we'll actually feed into the models
FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "daily_return", "price_change",
    "sma_5", "sma_10", "sma_20", "sma_50",
    "ema_12", "ema_26",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "bb_pct",
    "volatility",
    "volume_sma5", "volume_ratio", "volume_change",
    "avg_sentiment",
    "avg_sentiment_lag1", "avg_sentiment_lag2", "avg_sentiment_lag3",
    "avg_sentiment_roll3", "avg_sentiment_roll5", "avg_sentiment_roll7",
]


def build_features(df=None, save=True):
    """
    Runs the full feature pipeline. Returns (feature_df, feature_col_names).
    NaN rows from rolling calcs get dropped, as does the last row (no target).
    """
    if df is None:
        df = pd.read_csv(MERGED_CSV, parse_dates=["date"])
    else:
        df = df.copy()

    df = df.sort_values("date").reset_index(drop=True)

    if "avg_sentiment" not in df.columns:
        df["avg_sentiment"] = 0.0

    log.info("Building features on %d rows", len(df))

    df = add_price_change(df)
    df = add_sma(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_volatility(df)
    df = add_volume_features(df)
    df = add_sentiment_lags(df)
    df = add_sentiment_rolling(df)
    df = add_target(df)

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    df_clean = df[feat_cols + ["target", "date"]].dropna()
    df_clean = df_clean.iloc[:-1]  # last row has no future target

    log.info("Feature matrix ready: %d rows x %d features", len(df_clean), len(feat_cols))

    if save:
        out = MERGED_CSV.parent / "features.csv"
        df_clean.to_csv(out, index=False)
        log.info("Features saved to %s", out)

    return df_clean, feat_cols


if __name__ == "__main__":
    feat_df, cols = build_features()
    print(feat_df.tail())
    print("\nFeature columns:", cols)
    print("Target distribution:\n", feat_df["target"].value_counts())
