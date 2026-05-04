# app.py - main Streamlit dashboard for GlanhzeeTrade.ai

import os
import base64
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

from utils import get_logger, MODEL_DIR, DATA_DIR, ensure_nltk_resources
from data import collect_news, fetch_stock_data, merge_datasets, fetch_screener_data, TICKER_GROUPS
from sentiment import add_sentiment, generate_alerts
from features import build_features, FEATURE_COLS
from model import train_all, load_best_model, predict, META_PATH
from visualize import (
    candlestick_chart, sentiment_trend_chart, sentiment_pie,
    correlation_heatmap, model_comparison_chart, sentiment_vs_price,
    rsi_macd_chart, future_prediction_chart
)
from alerts import check_alerts, get_all_alerts, start_scheduler
from components import (
    render_header, render_metric_card, render_ai_insight,
    render_news_card, render_risk_gauge, render_landing,
    render_prediction_signal, render_section_header
)

log = get_logger("app")
ensure_nltk_resources()

# page setup
st.set_page_config(
    page_title="GlanhzeeTrade.ai",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css(path):
    with open(path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('style.css')

# load logo
LOGO_PATH = os.path.join(os.path.dirname(__file__), "IMAGE",
                         "A_minimalist_luxury_logo_for_202605021735 (1).jpeg")
LOGO_B64 = ""
if os.path.exists(LOGO_PATH):
    with open(LOGO_PATH, "rb") as f:
        LOGO_B64 = base64.b64encode(f.read()).decode()


# landing page
if st.session_state.get("page", "landing") == "landing":
    load_css('landing.css')
    render_landing()
    col_c = st.columns([1, 2, 1])
    with col_c[1]:
        if st.button("🚀 Initialize Trading Terminal", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
    st.stop()


# session state defaults
def _init_state():
    defaults = {
        "news_df": None, "stock_df": None, "merged_df": None,
        "feat_df": None, "feature_cols": None, "train_results": None,
        "ticker": "AAPL", "alert_thread": None, "page": "landing",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# screener widget (reused in dialog + tab)
def render_screener(key_suffix=""):
    if "screener_df" not in st.session_state:
        with st.spinner("Fetching live market data..."):
            st.session_state["screener_df"] = fetch_screener_data()

    scr_df = st.session_state.get("screener_df")

    if st.button("🔄 Refresh Market Data", key=f"refresh_{key_suffix}"):
        with st.spinner("Refreshing..."):
            st.session_state["screener_df"] = fetch_screener_data()
            st.rerun()

    if scr_df is not None and not scr_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            categories = ["All"] + list(TICKER_GROUPS.keys())
            sel_cat = st.selectbox("Filter by Sector", categories, key=f"cat_{key_suffix}")
        with c2:
            st.markdown("<div style='margin-bottom:2px; font-size:13px; font-weight:500; color:#A0AEC0;'>Performance</div>", unsafe_allow_html=True)
            perf_filter = st.radio("Performance", ["All", "Profit Only", "Loss Only"],
                                   horizontal=True, label_visibility="collapsed", key=f"perf_{key_suffix}")

        filtered_df = scr_df.copy()
        if sel_cat != "All":
            filtered_df = filtered_df[filtered_df["Category"] == sel_cat]
        if perf_filter == "Profit Only":
            filtered_df = filtered_df[filtered_df["Change %"] > 0]
        elif perf_filter == "Loss Only":
            filtered_df = filtered_df[filtered_df["Change %"] < 0]

        display_df = filtered_df.copy()
        display_df["Price"] = display_df["Price"].apply(lambda x: f"${x:,.2f}")
        display_df["Change %"] = display_df["Change %"].apply(lambda x: f"{x:+.2f}%")

        st.dataframe(display_df, use_container_width=True, hide_index=True, height=350)

        if len(filtered_df) > 0:
            render_section_header("Top Movers (Selected Filter)")
            top = filtered_df.iloc[0]
            bottom = filtered_df.iloc[-1]
            c3, c4 = st.columns(2)

            raw_top_pct = float(top["Change %"])
            raw_bottom_pct = float(bottom["Change %"])

            if raw_top_pct >= 0:
                c3.metric(f"🚀 Top Gainer: {top['Ticker']}", f"${top['Price']:.2f}", f"{top['Change %']:+.2f}%")
            if len(filtered_df) > 1 and raw_bottom_pct < 0:
                c4.metric(f"📉 Top Loser: {bottom['Ticker']}", f"${bottom['Price']:.2f}", f"{bottom['Change %']:+.2f}%")

            render_section_header("🔍 Deep Analysis")
            st.caption("Select a stock from the list to run full analysis on the main dashboard.")

            col_sel, col_btn = st.columns([2, 1])
            with col_sel:
                selected_ticker = st.selectbox("Select Asset", display_df["Ticker"].tolist(),
                                                label_visibility="collapsed", key=f"sel_{key_suffix}")
            with col_btn:
                if st.button(f"Analyze {selected_ticker} ➔", use_container_width=True, key=f"btn_{key_suffix}"):
                    st.session_state.ticker_input = selected_ticker
                    st.session_state.news_df = None
                    st.session_state.stock_df = None
                    st.rerun()
    else:
        st.warning("Could not load screener data. Yahoo Finance may be rate-limiting.")


@st.dialog("🌐 Live Market Screener", width="large")
def market_screener_dialog():
    render_screener(key_suffix="dialog")


# sidebar
with st.sidebar:
    if LOGO_B64:
        st.markdown(
            f'<div style="text-align:center;"><img src="data:image/jpeg;base64,{LOGO_B64}" '
            f'style="width:80%;border-radius:12px;margin-bottom:20px;box-shadow:0 4px 15px rgba(0,0,0,0.5);"></div>',
            unsafe_allow_html=True
        )
    st.markdown("## 📈 GlanhzeeTrade.ai")
    st.caption("Market Sentiment Analyzer")
    st.divider()

    if "ticker_input" not in st.session_state:
        st.session_state.ticker_input = "AAPL"

    ticker = st.text_input("Stock Ticker", key="ticker_input",
                           placeholder="e.g. AAPL, TSLA, MSFT").upper().strip()
    if not ticker:
        ticker = "AAPL"

    period_map = {
        "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo",
        "1 Year": "1y", "2 Years": "2y", "5 Years": "5y",
        "10 Years": "10y", "Max": "max", "Custom Range": "custom"
    }
    period_label = st.selectbox("Historical Period", list(period_map.keys()), index=3)
    period = period_map[period_label]

    start_date, end_date = None, None
    if period == "custom":
        import datetime
        c_sd, c_ed = st.columns(2)
        with c_sd:
            start_date = st.date_input("Start", datetime.date(2020, 1, 1))
        with c_ed:
            end_date = st.date_input("End", datetime.date.today())
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

    st.divider()
    run_btn = st.button("🚀 Fetch & Analyze", use_container_width=True)
    train_btn = st.button("🧠 Train Models", use_container_width=True)

    st.divider()
    st.markdown("### 🔔 Live Alerts")
    alert_ph = st.empty()

    st.divider()
    st.caption("Data: Yahoo Finance · Economic Times · Moneycontrol\nModels: LR · RF · LSTM")
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🌐 Quick View Market", use_container_width=True):
        market_screener_dialog()


# data fetch pipeline
if run_btn or st.session_state["news_df"] is None:
    active_ticker = ticker

    with st.spinner(f"Fetching news for **{active_ticker}**..."):
        try:
            news_df = collect_news(active_ticker, save=True)
            if not news_df.empty:
                news_df = add_sentiment(news_df)
        except Exception as e:
            log.warning("News collection issue: %s", e)
            news_df = pd.DataFrame()

    with st.spinner("Downloading stock data..."):
        try:
            if period == "custom":
                stock_df = fetch_stock_data(active_ticker, start=start_date, end=end_date, save=True)
            else:
                stock_df = fetch_stock_data(active_ticker, period=period, save=True)
        except Exception as e:
            log.error("Stock data error: %s", e)
            stock_df = pd.DataFrame()

    if not stock_df.empty and not news_df.empty:
        merged_df = merge_datasets(news_df, stock_df, save=True)
        feat_df, feature_cols = build_features(merged_df, save=True)
    elif not stock_df.empty:
        merged_df = stock_df.copy()
        merged_df["avg_sentiment"] = 0.0
        feat_df, feature_cols = build_features(merged_df, save=True)
    else:
        merged_df = pd.DataFrame()
        feat_df = pd.DataFrame()
        feature_cols = []

    st.session_state.update({
        "news_df": news_df, "stock_df": stock_df, "merged_df": merged_df,
        "feat_df": feat_df, "feature_cols": feature_cols, "ticker": active_ticker,
    })

    if stock_df.empty:
        st.error(f"Could not fetch stock data for '{active_ticker}'. Check the ticker symbol.")
    else:
        st.success(f"Data ready - {len(stock_df)} trading days | {len(news_df)} news articles")


# model training
if train_btn:
    feat_df = st.session_state.get("feat_df")
    feature_cols = st.session_state.get("feature_cols")

    if feat_df is None or feat_df.empty:
        st.error("Please fetch data first before training.")
    else:
        with st.spinner("Training LR, Random Forest & LSTM..."):
            try:
                results = train_all(feat_df, feature_cols)
                st.session_state["train_results"] = results
                st.success("All models trained and saved!")
            except Exception as e:
                st.error(f"Training error: {e}")


# sidebar alerts
all_alerts = get_all_alerts()
with alert_ph.container():
    if all_alerts:
        for a in all_alerts[:5]:
            css = "alert-bull" if "BULL" in a["direction"] else "alert-bear"
            alert_ph.markdown(
                f'<div class="alert-card {css}">'
                f'<b>{a["direction"]}</b> ({a["score"]:+.2f})<br>'
                f'{a["headline"][:80]}...'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        alert_ph.caption("No strong signals detected yet.")


# helpers
def get(key):
    return st.session_state.get(key)

def has_data():
    return get("stock_df") is not None and not get("stock_df").empty


# tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Overview", "📊 Technical", "💬 Sentiment",
    "🤖 Prediction", "🗄️ Data", "🌐 Market Screener", "🧠 Gemini AI"
])


# -- overview tab --
with tab1:
    ticker_label = get("ticker") or ticker
    render_header(ticker_label, is_online=has_data())

    if has_data():
        stock_df = get("stock_df")
        news_df = get("news_df") if get("news_df") is not None else pd.DataFrame()
        merged_df = get("merged_df") if get("merged_df") is not None else pd.DataFrame()

        latest = stock_df.iloc[-1]
        prev = stock_df.iloc[-2] if len(stock_df) > 1 else latest
        chg = latest["close"] - prev["close"]
        chg_pct = (chg / prev["close"]) * 100 if prev["close"] else 0
        direction_class = "delta-up" if chg >= 0 else "delta-down"
        direction_sym = "▲" if chg >= 0 else "▼"

        avg_sent = news_df["composite"].mean() if "composite" in news_df.columns and not news_df.empty else 0.0
        sent_lbl = "Positive" if avg_sent >= 0.05 else ("Negative" if avg_sent <= -0.05 else "Neutral")
        vol_avg = stock_df["volume"].mean() if "volume" in stock_df.columns else 0

        # risk score based on 30-day volatility
        volatility = stock_df['close'].pct_change().rolling(30).std().iloc[-1] * 100 if len(stock_df) >= 30 else 50
        risk_score = min(max(int(volatility * 10), 10), 95) if not np.isnan(volatility) else 50
        risk_label = "High Risk" if risk_score > 70 else "Medium Risk" if risk_score > 40 else "Low Risk"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card("CURRENT PRICE", f"${latest['close']:.2f}",
                               f"{direction_sym} {abs(chg):.2f} ({chg_pct:+.2f}%)", direction_class, "📈")
        with c2:
            render_metric_card("TRADING VOLUME", f"{latest['volume']/1e6:.2f}M",
                               f"Avg: {vol_avg/1e6:.1f}M", "delta-flat", "📊")
        with c3:
            sent_class = "delta-up" if avg_sent >= 0.05 else "delta-down" if avg_sent <= -0.05 else "delta-flat"
            render_metric_card("MARKET SENTIMENT", f"{avg_sent:+.2f}", sent_lbl, sent_class, "🧠")
        with c4:
            render_risk_gauge(risk_score, risk_label)

        st.markdown("<br>", unsafe_allow_html=True)

        if not merged_df.empty:
            render_section_header("Technical & Sentiment Action")
            st.plotly_chart(candlestick_chart(merged_df, ticker_label), use_container_width=True)

        render_section_header("AI Forward Projections")
        col_ai, col_pred = st.columns([1, 1])

        with col_ai:
            insight_text = f"The engine detects a {'bullish' if chg > 0 else 'bearish'} momentum forming. "
            insight_text += f"Market sentiment is currently {sent_lbl.lower()}, aligning with overall price action. "
            if not np.isnan(volatility):
                insight_text += f"Volatility is at {volatility:.2f}%."
            render_ai_insight(insight_text, confidence=0.85)

            if st.button("Run Deep Analysis ➔", key="btn_deep"):
                st.toast("Analysis Triggered")
            if st.button("Start Trading Simulation ➔", key="btn_sim"):
                st.toast("Simulation Sandbox Starting...")

        with col_pred:
            try:
                last_price = latest["close"]
                trend = (chg_pct / 100) if abs(chg_pct) < 5 else 0.01
                future_dates = pd.date_range(start=latest["date"], periods=6)[1:]
                future_prices = [last_price * (1 + trend * i + np.random.normal(0, 0.005)) for i in range(1, 6)]
                st.plotly_chart(future_prediction_chart(merged_df, list(future_dates), future_prices, ticker_label),
                                use_container_width=True)
            except Exception:
                st.warning("Train the models to unlock future forecasting.")

        if not news_df.empty and "headline" in news_df.columns:
            render_section_header("Live Market Wire")
            cols_to_show = ["date", "headline", "source", "sentiment_label", "composite"]
            show_cols = [c for c in cols_to_show if c in news_df.columns]

            n_col1, n_col2 = st.columns(2)
            news_items = news_df[show_cols].head(8).to_dict('records')

            for i, row in enumerate(news_items):
                col = n_col1 if i % 2 == 0 else n_col2
                with col:
                    render_news_card(row["headline"], str(row["date"])[:10],
                                     row.get("source", "NewsWire"), row.get("composite", 0))

    else:
        if get("ticker") and get("stock_df") is not None and get("stock_df").empty:
            st.warning(f"No stock data found for '{get('ticker')}'. Use a valid ticker like AAPL, MSFT, etc.")
        else:
            st.info("Enter a ticker and click **Fetch & Analyze** to get started.")


# -- technical analysis tab --
with tab2:
    render_section_header("Candlestick & Volume")

    if has_data():
        stock_df = get("stock_df")
        feat_df = get("feat_df") if get("feat_df") is not None else pd.DataFrame()
        ticker_lbl = get("ticker") or ticker

        if not feat_df.empty and "sma_20" in feat_df.columns:
            merged_chart = stock_df.merge(
                feat_df[["date", "sma_20", "rsi", "macd", "macd_signal", "macd_hist"]],
                on="date", how="left",
            )
        else:
            merged_chart = stock_df.copy()

        st.plotly_chart(candlestick_chart(merged_chart, ticker_lbl), use_container_width=True)

        render_section_header("RSI & MACD")
        st.plotly_chart(rsi_macd_chart(merged_chart), use_container_width=True)

        render_section_header("Statistics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("52-week High", f"${stock_df['high'].max():.2f}")
            st.metric("52-week Low", f"${stock_df['low'].min():.2f}")
        with c2:
            st.metric("Avg Daily Volume", f"{stock_df['volume'].mean()/1e6:.2f}M")
            st.metric("Avg Daily Return", f"{stock_df['close'].pct_change().mean()*100:.3f}%")
        with c3:
            if len(stock_df) >= 30:
                st.metric("Volatility (30d)", f"{stock_df['close'].pct_change().rolling(30).std().iloc[-1]*100:.2f}%")
            else:
                st.metric("Volatility (30d)", "N/A")
            if "rsi" in feat_df.columns and not feat_df.empty:
                st.metric("Current RSI", f"{feat_df['rsi'].iloc[-1]:.1f}")
    else:
        if get("ticker") and get("stock_df") is not None and get("stock_df").empty:
            st.warning("No data available. Enter a valid ticker symbol.")
        else:
            st.info("Fetch data first to see technical analysis.")


# -- sentiment tab --
with tab3:
    news_df = get("news_df") if get("news_df") is not None else pd.DataFrame()

    if not news_df.empty and "composite" in news_df.columns:
        render_section_header("Sentiment Trend")
        st.plotly_chart(sentiment_trend_chart(news_df), use_container_width=True)

        col_pie, col_stats = st.columns([1, 1])
        with col_pie:
            render_section_header("Distribution")
            st.plotly_chart(sentiment_pie(news_df), use_container_width=True)
        with col_stats:
            render_section_header("Sentiment Stats")
            st.metric("Mean Score", f"{news_df['composite'].mean():+.4f}")
            st.metric("Max Bullish", f"{news_df['composite'].max():+.4f}")
            st.metric("Max Bearish", f"{news_df['composite'].min():+.4f}")
            st.metric("Std Dev", f"{news_df['composite'].std():.4f}")

        render_section_header("Top Bullish Headlines")
        top_bull = news_df.nlargest(5, "composite")[["date", "headline", "source", "composite"]]
        st.dataframe(top_bull, use_container_width=True, hide_index=True)

        render_section_header("Top Bearish Headlines")
        top_bear = news_df.nsmallest(5, "composite")[["date", "headline", "source", "composite"]]
        st.dataframe(top_bear, use_container_width=True, hide_index=True)

        alerts = generate_alerts(news_df)
        if alerts:
            render_section_header("🔔 Strong Signals")
            for a in alerts[:10]:
                css = "alert-bull" if "BULL" in a["direction"] else "alert-bear"
                st.markdown(
                    f'<div class="alert-card {css}">'
                    f'<b>{a["direction"]}</b> | Score: {a["score"]:+.4f} | {a["date"]}<br>'
                    f'{a["headline"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Fetch data to see sentiment analysis.")


# -- prediction tab --
with tab4:
    feat_df = get("feat_df") if get("feat_df") is not None else pd.DataFrame()
    results = get("train_results")
    ticker_lbl = get("ticker") or ticker

    if results:
        render_section_header("Model Performance")
        st.caption("Comparing Logistic Regression, Random Forest, and LSTM models.")
        st.plotly_chart(model_comparison_chart(results), use_container_width=True)

        rows = []
        for name, r in results.items():
            if r.get("accuracy") is not None:
                rows.append({
                    "Model": name,
                    "Accuracy": f"{r['accuracy']:.2%}",
                    "Precision": f"{r['precision']:.2%}",
                    "Recall": f"{r['recall']:.2%}",
                    "F1 Score": f"{r['f1']:.2%}",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        render_section_header("🔮 Predict Next Move")
        if not feat_df.empty and META_PATH.exists():
            try:
                model, scaler, meta = load_best_model()
                feat_cols_saved = meta["feature_cols"]
                latest_row = feat_df[feat_cols_saved].iloc[-1].values
                pred, prob = predict(latest_row, model, scaler)
                render_prediction_signal(ticker_lbl, pred, prob, meta["best_model"])
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.info("Train models first to see predictions.")
    else:
        st.info("Click **🧠 Train Models** in the sidebar to train and compare models.")

    merged_df = get("merged_df") if get("merged_df") is not None else pd.DataFrame()
    if not merged_df.empty:
        render_section_header("Feature Correlations")
        st.plotly_chart(correlation_heatmap(feat_df if not feat_df.empty else merged_df),
                        use_container_width=True)


# -- data explorer tab --
with tab5:
    render_section_header("📰 News Dataset")
    news_df = get("news_df") if get("news_df") is not None else pd.DataFrame()
    if not news_df.empty:
        st.caption(f"{len(news_df)} articles collected")
        st.dataframe(news_df, use_container_width=True, height=300)
        csv_news = news_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download News CSV", csv_news, f"{ticker}_news.csv", "text/csv")
    else:
        st.info("No news data loaded yet.")

    render_section_header("📊 Stock Dataset")
    stock_df = get("stock_df") if get("stock_df") is not None else pd.DataFrame()
    if not stock_df.empty:
        st.caption(f"{len(stock_df)} trading days")
        st.dataframe(stock_df, use_container_width=True, height=300)
        csv_stock = stock_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Stock CSV", csv_stock, f"{ticker}_stock.csv", "text/csv")
    else:
        st.info("No stock data loaded yet.")

    render_section_header("🔗 Merged Feature Dataset")
    feat_df = get("feat_df") if get("feat_df") is not None else pd.DataFrame()
    if not feat_df.empty:
        st.caption(f"{len(feat_df)} rows x {feat_df.shape[1]} columns")
        st.dataframe(feat_df, use_container_width=True, height=300)
        csv_feat = feat_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Features CSV", csv_feat, f"{ticker}_features.csv", "text/csv")
    else:
        st.info("No feature data generated yet.")


# -- market screener tab --
with tab6:
    render_section_header("🌐 Market Screener (Live Performance)")
    st.caption("View top gainers and losers across Tech, Crypto, Energy, Finance, and Automotive sectors.")
    render_screener(key_suffix="tab")


# -- gemini AI tab --
with tab7:
    render_section_header("🧠 Gemini AI Chat Assistant")
    st.caption("Ask Gemini about market trends, technical analysis, or trading strategies.")

    if "gemini_messages" not in st.session_state:
        st.session_state.gemini_messages = []

    for message in st.session_state.gemini_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask Gemini AI..."):
        st.session_state.gemini_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                ticker_lbl = get("ticker") or ticker
                context = ""
                if has_data():
                    latest_price = get("stock_df").iloc[-1]["close"]
                    context = f"Context: analyzing '{ticker_lbl}', latest close ${latest_price:.2f}. "

                response = model.generate_content(context + prompt)
                st.markdown(response.text)
                st.session_state.gemini_messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Gemini AI error: {e}")


# footer
st.divider()
st.caption("GlanhzeeTrade.ai v2.0 · Built with Streamlit, VADER, yfinance, Scikit-learn & TensorFlow")
