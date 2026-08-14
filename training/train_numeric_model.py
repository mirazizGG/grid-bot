"""Raqamli indikatorlar (RSI, MACD, ATR, SMA, Bollinger, Momentum, ADX,
Stochastic, sessiya vaqti) asosida Buy/Sell/Hold classifier o'qitadi.

Ko'p symbol (XAUUSD, EURUSD, GBPUSD, USDJPY) va ikkita timeframe (M15, H1)
dan yig'ilgan ma'lumotdan foydalanadi (data_pipeline/multi_symbol_dataset.py
natijasi) -- bitta symbol/timeframe'ga qaraganda ancha xilma-xil va katta
o'qitish to'plami.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import mt5_context  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_DIR = os.path.join(_THIS_DIR, "..", "data_pipeline", "output", "multi")
COMBINED_LABELED = os.path.join(_THIS_DIR, "..", "data_pipeline", "output", "multi_labeled.parquet")

MODEL_OUT = os.path.join(_THIS_DIR, "..", "backend", "model", "numeric_model.joblib")
META_OUT = os.path.join(_THIS_DIR, "..", "backend", "model", "numeric_model_meta.json")

LOOKBACK_BARS = 100  # get_context() jonli holatda ham shuncha bar oladi -- train/serve mos kelishi uchun
MIN_HISTORY = 55


def build_dataset(labeled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (symbol, timeframe), group in labeled.groupby(["symbol", "timeframe"]):
        raw_path = os.path.join(MULTI_DIR, f"{symbol}_{timeframe}_raw.parquet")
        raw = pd.read_parquet(raw_path)
        highs_all = raw["high"].to_numpy(dtype=np.float64)
        lows_all = raw["low"].to_numpy(dtype=np.float64)
        closes_all = raw["close"].to_numpy(dtype=np.float64)
        times_all = raw["time"]

        for _, row in group.iterrows():
            idx = int(row["window_end_idx"])
            start = max(0, idx - LOOKBACK_BARS + 1)
            if idx - start < MIN_HISTORY:
                continue

            h = highs_all[start: idx + 1]
            l = lows_all[start: idx + 1]
            c = closes_all[start: idx + 1]
            bar_hour = pd.to_datetime(times_all.iloc[idx]).hour

            raw_ind = mt5_context.compute_raw_indicators(h, l, c, bar_hour=bar_hour)
            features = mt5_context.to_feature_vector(raw_ind)

            rows.append({**{name: val for name, val in zip(mt5_context.FEATURE_NAMES, features)},
                         "label": row["label"], "split": row["split"],
                         "symbol": symbol, "timeframe": timeframe})

    return pd.DataFrame(rows)


def main():
    labeled = pd.read_parquet(COMBINED_LABELED)

    print("Feature'lar hisoblanmoqda...")
    dataset = build_dataset(labeled)
    print(f"{len(dataset)} qator tayyor")

    X = dataset[mt5_context.FEATURE_NAMES]
    y = dataset["label"]

    train_mask = dataset["split"] == "train"
    val_mask = dataset["split"] == "val"
    test_mask = dataset["split"] == "test"

    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X[train_mask], y[train_mask])

    for name, mask in [("VAL", val_mask), ("TEST", test_mask)]:
        preds = model.predict(X[mask])
        acc = accuracy_score(y[mask], preds)
        print(f"\n=== {name} (accuracy={acc:.3f}) ===")
        print(classification_report(y[mask], preds, target_names=model.classes_))
        print(confusion_matrix(y[mask], preds, labels=model.classes_))

    print("\n=== TEST symbol/timeframe kesimida ===")
    test_df = dataset[test_mask].copy()
    test_df["pred"] = model.predict(X[test_mask])
    for (symbol, timeframe), grp in test_df.groupby(["symbol", "timeframe"]):
        acc = accuracy_score(grp["label"], grp["pred"])
        print(f"  {symbol} {timeframe}: n={len(grp)} accuracy={acc:.3f}")

    importances = sorted(zip(mt5_context.FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    print("\nFeature importance:")
    for name, imp in importances:
        print(f"  {name}: {imp:.3f}")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    with open(META_OUT, "w") as f:
        json.dump({"feature_names": mt5_context.FEATURE_NAMES, "classes": list(model.classes_)}, f, indent=2)

    print(f"\nSaqlandi -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
