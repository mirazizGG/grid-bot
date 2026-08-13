"""Raqamli indikatorlar (RSI, MACD, ATR, SMA, Bollinger, Momentum) asosida
Buy/Sell/Hold classifier o'qitadi. Rasm-CNN'dan farqli o'laroq, bu local'da
(GPU shart emas) tez ishlaydi va data_pipeline/label_generator.py allaqachon
yaratgan label'lardan (labeled_windows.parquet) foydalanadi -- yangi label
yaratish shart emas, faqat har window uchun feature vector hisoblanadi.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline"))

import mt5_context  # noqa: E402
from config import RAW_DATA_PATH, LABELED_DATA_PATH  # noqa: E402  (data_pipeline/config.py)

MODEL_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "model", "numeric_model.joblib")
META_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "model", "numeric_model_meta.json")

LOOKBACK_BARS = 100  # get_context() jonli holatda ham shuncha bar oladi -- train/serve mos kelishi uchun
MIN_HISTORY = 55  # SMA50 barqarorlashishi uchun minimal tarix


def build_dataset(raw: pd.DataFrame, labeled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    highs_all = raw["high"].to_numpy(dtype=np.float64)
    lows_all = raw["low"].to_numpy(dtype=np.float64)
    closes_all = raw["close"].to_numpy(dtype=np.float64)

    for _, row in labeled.iterrows():
        idx = int(row["window_end_idx"])
        start = max(0, idx - LOOKBACK_BARS + 1)
        if idx - start < MIN_HISTORY:
            continue

        h = highs_all[start: idx + 1]
        l = lows_all[start: idx + 1]
        c = closes_all[start: idx + 1]

        raw_ind = mt5_context.compute_raw_indicators(h, l, c)
        features = mt5_context.to_feature_vector(raw_ind)

        rows.append({**{name: val for name, val in zip(mt5_context.FEATURE_NAMES, features)},
                     "label": row["label"], "split": row["split"]})

    return pd.DataFrame(rows)


def main():
    raw = pd.read_parquet(RAW_DATA_PATH)
    labeled = pd.read_parquet(LABELED_DATA_PATH)

    print("Feature'lar hisoblanmoqda...")
    dataset = build_dataset(raw, labeled)
    print(f"{len(dataset)} qator tayyor")

    X = dataset[mt5_context.FEATURE_NAMES]
    y = dataset["label"]

    train_mask = dataset["split"] == "train"
    val_mask = dataset["split"] == "val"
    test_mask = dataset["split"] == "test"

    # Eslatma: class_weight="balanced" bilan RandomForest sinovda tasodifiy
    # taxmindan (33%) PASTROQ natija berdi (signalni buzib yuborgan).
    # Balanslamagan GradientBoosting eng yaxshi (va statistik jihatdan hali
    # ham tasodifiy chiziq atrofida) natijani berdi -- xolis eslatma:
    # bu klassik indikatorlar M15 XAUUSD uchun kuchli signal bermaydi.
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X[train_mask], y[train_mask])

    for name, mask in [("VAL", val_mask), ("TEST", test_mask)]:
        preds = model.predict(X[mask])
        print(f"\n=== {name} ===")
        print(classification_report(y[mask], preds, target_names=model.classes_))
        print(confusion_matrix(y[mask], preds, labels=model.classes_))

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
