# model.py - trains LR, Random Forest, and LSTM, picks the best one

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from sklearn.pipeline import Pipeline

from utils import get_logger, MODEL_DIR, MERGED_CSV
from features import build_features, FEATURE_COLS

log = get_logger("model")

LR_PATH = MODEL_DIR / "logistic_regression.pkl"
RF_PATH = MODEL_DIR / "random_forest.pkl"
LSTM_PATH = MODEL_DIR / "lstm_model.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
META_PATH = MODEL_DIR / "model_meta.pkl"


def prepare_data(feat_df, feature_cols):
    """80/20 temporal split (no shuffling - this is time series data)."""
    X = feat_df[feature_cols].values
    y = feat_df["target"].values
    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    log.info("Scaler saved -> %s", SCALER_PATH)
    return X_train_s, X_test_s, scaler


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    log.info("%s -> Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f", name, acc, prec, rec, f1)
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def train_logistic(X_train, y_train, X_test, y_test):
    log.info("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    with open(LR_PATH, "wb") as f:
        pickle.dump(lr, f)
    return {**evaluate("Logistic Regression", y_test, y_pred), "model_obj": lr}


def train_random_forest(X_train, y_train, X_test, y_test, feature_cols):
    log.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        min_samples_leaf=5, random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    log.info("Top features:\n%s", importances.head(5).to_string())

    with open(RF_PATH, "wb") as f:
        pickle.dump(rf, f)
    return {**evaluate("Random Forest", y_test, y_pred), "model_obj": rf, "feature_importance": importances}


def train_lstm(X_train, y_train, X_test, y_test, seq_len=10):
    """Stacked LSTM for sequential pattern recognition. Skips if TF isn't installed."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    except ImportError:
        log.warning("TensorFlow not installed, skipping LSTM")
        return {"model": "LSTM", "accuracy": None, "precision": None,
                "recall": None, "f1": None, "model_obj": None}

    log.info("Training LSTM (seq_len=%d)...", seq_len)

    def make_sequences(X, y, length):
        xs, ys = [], []
        for i in range(len(X) - length):
            xs.append(X[i:i + length])
            ys.append(y[i + length])
        return np.array(xs), np.array(ys)

    X_tr_seq, y_tr_seq = make_sequences(X_train, y_train, seq_len)
    X_te_seq, y_te_seq = make_sequences(X_test, y_test, seq_len)

    if len(X_tr_seq) < 10:
        log.warning("Not enough data for LSTM sequences, skipping")
        return {"model": "LSTM", "accuracy": None, "precision": None,
                "recall": None, "f1": None, "model_obj": None}

    n_features = X_train.shape[1]
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    model.fit(
        X_tr_seq, y_tr_seq,
        epochs=50, batch_size=32,
        validation_split=0.1,
        callbacks=callbacks, verbose=0,
    )

    y_prob = model.predict(X_te_seq, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    model.save(LSTM_PATH)
    log.info("LSTM saved -> %s", LSTM_PATH)

    return {**evaluate("LSTM", y_te_seq, y_pred), "model_obj": model}


def train_all(feat_df=None, feature_cols=None):
    """Trains all three models, picks the best by F1, saves metadata."""
    if feat_df is None or feature_cols is None:
        feat_df, feature_cols = build_features(save=False)

    if feat_df.empty:
        log.error("Empty feature dataframe, can't train")
        return {}

    X_train, X_test, y_train, y_test = prepare_data(feat_df, feature_cols)
    X_train_s, X_test_s, scaler = scale_data(X_train, X_test)

    results = {}
    results["Logistic Regression"] = train_logistic(X_train_s, y_train, X_test_s, y_test)
    results["Random Forest"] = train_random_forest(X_train_s, y_train, X_test_s, y_test, feature_cols)
    results["LSTM"] = train_lstm(X_train_s, y_train, X_test_s, y_test)

    # pick best by F1 (better than accuracy for imbalanced classes)
    best_name = max(
        (k for k in results if results[k].get("f1") is not None),
        key=lambda k: results[k]["f1"],
    )
    log.info("Best model: %s (F1=%.4f)", best_name, results[best_name]["f1"])

    meta = {
        "best_model": best_name,
        "feature_cols": feature_cols,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "model_obj"}
                     for k, v in results.items()},
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    return results


def load_best_model():
    """Loads the best saved model along with its scaler and metadata."""
    if not META_PATH.exists():
        raise FileNotFoundError("No trained model found. Run train_all() first.")

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    best = meta["best_model"]
    log.info("Loading best model: %s", best)

    if best == "Logistic Regression":
        with open(LR_PATH, "rb") as f:
            model = pickle.load(f)
    elif best == "Random Forest":
        with open(RF_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(LSTM_PATH)
        except Exception:
            log.warning("Couldn't load LSTM, falling back to Random Forest")
            with open(RF_PATH, "rb") as f:
                model = pickle.load(f)

    return model, scaler, meta


def predict(features_row, model, scaler):
    """Single-row prediction. Returns (0 or 1, probability)."""
    row_s = scaler.transform(features_row.reshape(1, -1))

    if hasattr(model, "predict") and "keras" in str(type(model)).lower():
        row_s = row_s.reshape(1, 1, -1)
        prob = float(model.predict(row_s, verbose=0)[0][0])
    else:
        prob = float(model.predict_proba(row_s)[0][1])

    pred = int(prob >= 0.5)
    return pred, prob


if __name__ == "__main__":
    results = train_all()
    for name, res in results.items():
        if res.get("accuracy") is not None:
            print(f"{name:<22} Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}")
