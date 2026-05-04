# visualize.py - Plotly chart builders for the dashboard

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils import get_logger

log = get_logger("visualize")

BULL_COLOR = "#00C896"
BEAR_COLOR = "#FF4B6A"
NEUT_COLOR = "#A0AEC0"
ACCENT = "#7B61FF"
BG_COLOR = "#050505"
PAPER_BG = "rgba(0,0,0,0)"


def _dark_layout(fig, title="", height=400):
    """Applies the dark terminal-style theme to any plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=16, family="Inter")),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=BG_COLOR,
        font=dict(color="#8B9BB4", family="Inter, sans-serif"),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)",
            showspikes=True, spikecolor="#7B61FF", spikethickness=1, spikedash="dot", spikemode="across"
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)",
            showspikes=True, spikecolor="#7B61FF", spikethickness=1, spikedash="dot", spikemode="across"
        ),
        hovermode="x unified"
    )
    return fig


def candlestick_chart(stock_df, ticker):
    """OHLC candlestick with volume subplot and optional SMA overlays."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=[0.72, 0.28],
    )

    fig.add_trace(go.Candlestick(
        x=stock_df["date"],
        open=stock_df["open"], high=stock_df["high"],
        low=stock_df["low"], close=stock_df["close"],
        name="OHLC",
        increasing_line_color=BULL_COLOR,
        decreasing_line_color=BEAR_COLOR,
    ), row=1, col=1)

    if "sma_20" in stock_df.columns:
        fig.add_trace(go.Scatter(
            x=stock_df["date"], y=stock_df["sma_20"],
            name="SMA-20", line=dict(color=ACCENT, width=1.5, dash="dot"),
        ), row=1, col=1)

    if "sma_50" in stock_df.columns:
        fig.add_trace(go.Scatter(
            x=stock_df["date"], y=stock_df["sma_50"],
            name="SMA-50", line=dict(color="#F59E0B", width=1.5, dash="dash"),
            hoverinfo="y+name"
        ), row=1, col=1)

    if "volume" in stock_df.columns:
        colors = [BULL_COLOR if c >= o else BEAR_COLOR
                  for o, c in zip(stock_df["open"], stock_df["close"])]
        fig.add_trace(go.Bar(
            x=stock_df["date"], y=stock_df["volume"],
            marker_color=colors, opacity=0.8, name="Volume", hoverinfo="y+name"
        ), row=2, col=1)

    # crosshair and slider setup
    fig.update_xaxes(rangeslider_visible=False, showspikes=True, spikemode="across", spikethickness=1, spikecolor="#64748B", row=1, col=1)
    fig.update_yaxes(showspikes=True, spikemode="across", spikethickness=1, spikecolor="#64748B", row=1, col=1)
    fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.05, row=2, col=1)
    fig.update_layout(hovermode="x unified")

    return _dark_layout(fig, f"{ticker} Candlestick & Volume", height=550)


def sentiment_trend_chart(news_df):
    """Daily sentiment bars with a 7-day rolling average line."""
    if "date" not in news_df.columns or "composite" not in news_df.columns:
        return go.Figure()

    daily = (
        news_df.groupby("date")["composite"]
        .mean().reset_index().sort_values("date")
    )
    daily["rolling7"] = daily["composite"].rolling(7).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily["date"], y=daily["composite"],
        name="Daily Sentiment",
        marker_color=[BULL_COLOR if v >= 0 else BEAR_COLOR for v in daily["composite"]],
        opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["rolling7"],
        name="7-day Avg", line=dict(color=ACCENT, width=2), mode="lines",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=NEUT_COLOR, opacity=0.4)

    return _dark_layout(fig, "Sentiment Trend Over Time")


def sentiment_pie(news_df):
    if "sentiment_label" not in news_df.columns:
        return go.Figure()

    counts = news_df["sentiment_label"].value_counts()
    color_map = {"Positive": BULL_COLOR, "Negative": BEAR_COLOR, "Neutral": NEUT_COLOR}

    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        marker_colors=[color_map.get(l, ACCENT) for l in counts.index],
        hole=0.55, textfont_size=13,
    ))
    fig = _dark_layout(fig, "Sentiment Distribution", height=320)
    fig.update_traces(textposition="outside", textinfo="percent+label")
    return fig


def correlation_heatmap(merged_df):
    num_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    keep = [c for c in num_cols if c != "ticker"][:18]
    corr = merged_df[keep].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}", textfont_size=9,
    ))
    return _dark_layout(fig, "Feature Correlation Matrix", height=480)


def model_comparison_chart(results):
    """Grouped bars comparing accuracy/precision/recall/F1 across models."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    models = [k for k in results if results[k].get("accuracy") is not None]
    palette = [BULL_COLOR, ACCENT, "#FFD166", BEAR_COLOR]

    fig = go.Figure()
    for m, color in zip(metrics, palette):
        vals = [results[model].get(m, 0) for model in models]
        fig.add_trace(go.Bar(
            name=m.capitalize(), x=models, y=vals,
            marker_color=color, text=[f"{v:.2%}" for v in vals], textposition="outside",
        ))
    fig.update_layout(barmode="group")
    return _dark_layout(fig, "Model Performance Comparison", height=400)


def sentiment_vs_price(merged_df, ticker):
    """Dual-axis overlay: price on left, sentiment on right."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=merged_df["date"], y=merged_df["close"],
        name="Close Price", line=dict(color=BULL_COLOR, width=2),
    ), secondary_y=False)

    if "avg_sentiment" in merged_df.columns:
        fig.add_trace(go.Scatter(
            x=merged_df["date"], y=merged_df["avg_sentiment"],
            name="Avg Sentiment", line=dict(color=ACCENT, width=1.5, dash="dot"),
        ), secondary_y=True)

    fig.update_yaxes(title_text="Price (USD)", secondary_y=False, showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=True, showgrid=False)
    fig.update_layout(hovermode="x unified")
    return _dark_layout(fig, f"{ticker} - Price vs Sentiment", height=450)


def future_prediction_chart(past_df, future_dates, future_prices, ticker):
    """Historical close + AI forecast with confidence band."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=past_df["date"].tail(60), y=past_df["close"].tail(60),
        name="Historical Close", line=dict(color="#94A3B8", width=2),
        fill="tozeroy", fillcolor="rgba(148, 163, 184, 0.1)"
    ))

    # Convert future_dates to list of strings to avoid type issues
    if hasattr(future_dates, 'strftime'):
        fd_list = future_dates.strftime("%Y-%m-%d").tolist()
    else:
        fd_list = [str(d)[:10] for d in future_dates]

    fig.add_trace(go.Scatter(
        x=fd_list, y=future_prices,
        name="AI Forecast (5 Days)", line=dict(color=ACCENT, width=3, dash="dot"),
        mode="lines+markers", marker=dict(size=8, color=ACCENT)
    ))

    upper = [p * 1.02 for p in future_prices]
    lower = [p * 0.98 for p in future_prices]

    fig.add_trace(go.Scatter(
        x=fd_list + fd_list[::-1],
        y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(123, 97, 255, 0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Interval", showlegend=True
    ))

    fig.update_layout(hovermode="x unified")
    return _dark_layout(fig, f"{ticker} AI Price Forecast (Next 5 Days)", height=400)


def rsi_macd_chart(stock_df):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08, subplot_titles=("RSI (14)", "MACD")
    )

    if "rsi" in stock_df.columns:
        fig.add_trace(go.Scatter(
            x=stock_df["date"], y=stock_df["rsi"],
            line=dict(color=ACCENT, width=1.5), name="RSI",
        ), row=1, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=BEAR_COLOR, opacity=0.5, row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=BULL_COLOR, opacity=0.5, row=1, col=1)

    if "macd" in stock_df.columns:
        fig.add_trace(go.Scatter(
            x=stock_df["date"], y=stock_df["macd"],
            line=dict(color=BULL_COLOR, width=1.5), name="MACD",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=stock_df["date"], y=stock_df["macd_signal"],
            line=dict(color=BEAR_COLOR, width=1.5, dash="dot"), name="Signal",
        ), row=2, col=1)
        if "macd_hist" in stock_df.columns:
            fig.add_trace(go.Bar(
                x=stock_df["date"], y=stock_df["macd_hist"],
                marker_color=[BULL_COLOR if v >= 0 else BEAR_COLOR for v in stock_df["macd_hist"]],
                name="Histogram", opacity=0.5,
            ), row=2, col=1)

    return _dark_layout(fig, "", height=440)
