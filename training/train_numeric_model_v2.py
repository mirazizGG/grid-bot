"""train_numeric_model.py bilan bir xil, lekin kattalashtirilgan datasetdan
(data_pipeline/output/multi_v2/, ~40k qator, 99999 bar/symbol -- avvalgi
7960 qatorga nisbatan 5x ko'p) o'qitadi va natijani ALOHIDA faylga saqlaydi
(numeric_model_v2.joblib) -- joriy deploy qilingan modelga tegmaydi.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mt5_context  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_DIR = os.path.join(_THIS_DIR, "..", "data_pipeline", "output", "multi_v2")
COMBINED_LABELED = os.path.join(_THIS_DIR, "..", "data_pipeline", "output", "multi_v2_labeled.parquet")

MODEL_OUT = os.path.join(_THIS_DIR, "..", "backend", "model", "numeric_model_v2.joblib")
META_OUT = os.path.join(_THIS_DIR, "..", "backend", "model", "numeric_model_v2_meta.json")


def main():
    labeled = pd.read_parquet(COMBINED_LABELED)
    print(f"Label'langan qatorlar: {len(labeled)}")

    print("Feature'lar hisoblanmoqda (bu kattaroq datasetda biroz uzoqroq davom etadi)...")
    rows = []
    LOOKBACK_BARS = 100
    MIN_HISTORY = 55
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

    dataset = pd.DataFrame(rows)
    print(f"{len(dataset)} qator tayyor")

    X = dataset[mt5_context.FEATURE_NAMES]
    y = dataset["label"]
    train_mask = dataset["split"] == "train"
    val_mask = dataset["split"] == "val"
    test_mask = dataset["split"] == "test"

    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42,
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

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    with open(META_OUT, "w") as f:
        json.dump({"feature_names": mt5_context.FEATURE_NAMES, "classes": list(model.classes_),
                    "n_train_rows": int(train_mask.sum()), "history_bars_per_symbol": 99999}, f, indent=2)
    print(f"\nSaqlandi -> {MODEL_OUT}")

    dataset.to_parquet(os.path.join(_THIS_DIR, "..", "data_pipeline", "output", "multi_v2_features.parquet"), index=False)


if __name__ == "__main__":
    main()
