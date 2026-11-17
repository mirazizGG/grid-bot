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
import argparse
import hashlib
from datetime import datetime, timezone
import sklearn

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

DEFAULT_OUT_DIR = os.path.join(_THIS_DIR, "model_candidate")

LOOKBACK_BARS = 100  # get_context() jonli holatda ham shuncha bar oladi -- train/serve mos kelishi uchun
MIN_HISTORY = 55


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_provenance(labeled):
    raw_hashes = {}
    for symbol, timeframe in labeled[["symbol", "timeframe"]].drop_duplicates().itertuples(index=False, name=None):
        filename = f"{symbol}_{timeframe}_raw.parquet"
        raw_hashes[filename] = file_sha256(os.path.join(MULTI_DIR, filename))
    return {"labeled_sha256": file_sha256(COMBINED_LABELED), "raw_sha256": raw_hashes}


def split_summary(labeled):
    return {name: {"rows": int(len(group)),
                   "start": pd.to_datetime(group["time"], utc=True).min().isoformat(),
                   "end": pd.to_datetime(group["time"], utc=True).max().isoformat(),
                   "label_end": pd.to_datetime(group["label_end_time"], utc=True).max().isoformat()}
            for name, group in labeled.groupby("split")}


def validate_labeled(labeled):
    required = {"symbol", "timeframe", "window_end_idx", "time", "label_end_time", "label_version", "label", "split"}
    if not required.issubset(labeled.columns) or not labeled["label_version"].eq(2).all():
        raise ValueError("Eski dataset: data_pipeline/relabel_from_raw.py orqali yangidan yarating")
    if set(labeled["split"]) != {"train", "val", "test"}:
        raise ValueError("Train, validation va test split'lari talab qilinadi")
    if not labeled["label"].isin(("Buy", "Sell", "Hold")).all():
        raise ValueError("Noma'lum label mavjud")
    indices = pd.to_numeric(labeled["window_end_idx"], errors="coerce")
    if indices.isna().any() or (indices < 0).any() or (indices % 1 != 0).any():
        raise ValueError("Window indekslari yaroqsiz")
    if labeled.duplicated(["symbol", "timeframe", "window_end_idx"]).any():
        raise ValueError("Takroriy window mavjud")
    times = pd.to_datetime(labeled["time"], utc=True, errors="coerce")
    ends = pd.to_datetime(labeled["label_end_time"], utc=True, errors="coerce")
    if times.isna().any() or ends.isna().any() or (ends <= times).any():
        raise ValueError("Label vaqtlari yaroqsiz")
    for before, after in (("train", "val"), ("val", "test")):
        if times[labeled["split"] == before].max() >= times[labeled["split"] == after].min() or \
           ends[labeled["split"] == before].max() > times[labeled["split"] == after].min():
            raise ValueError("Split'lar orasida vaqt yoki label kesishgan")


def build_dataset(labeled: pd.DataFrame) -> pd.DataFrame:
    validate_labeled(labeled)
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
            if idx >= len(raw) or pd.Timestamp(times_all.iloc[idx]).tz_localize(None) != pd.Timestamp(row["time"]).tz_localize(None):
                raise ValueError(f"Raw/label window mos emas: {symbol} {timeframe} idx={idx}")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="Candidate model papkasi; default faol backend modeliga tegmaydi")
    args = parser.parse_args()
    if os.path.realpath(args.out_dir) == os.path.realpath(os.path.join(_THIS_DIR, "..", "backend", "model")):
        parser.error("Faol backend modelini bevosita almashtirish mumkin emas; candidate papkani tanlang")
    labeled = pd.read_parquet(COMBINED_LABELED)
    validate_labeled(labeled)

    print("Feature'lar hisoblanmoqda...")
    dataset = build_dataset(labeled)
    print(f"{len(dataset)} qator tayyor")

    X = dataset[mt5_context.FEATURE_NAMES]
    y = dataset["label"]

    train_mask = dataset["split"] == "train"
    val_mask = dataset["split"] == "val"

    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X[train_mask], y[train_mask])

    preds = model.predict(X[val_mask])
    acc = accuracy_score(y[val_mask], preds)
    validation_metrics = {"accuracy": float(acc), "rows": int(val_mask.sum()),
                          "majority_class_baseline_accuracy": float(y[val_mask].value_counts(normalize=True).max()),
                          "classification_report": classification_report(
                              y[val_mask], preds, labels=model.classes_, target_names=model.classes_,
                              zero_division=0, output_dict=True),
                          "confusion_matrix": confusion_matrix(y[val_mask], preds, labels=model.classes_).tolist()}
    print(f"\n=== VAL (accuracy={acc:.3f}) ===")
    print(classification_report(y[val_mask], preds, labels=model.classes_, target_names=model.classes_, zero_division=0))
    print(confusion_matrix(y[val_mask], preds, labels=model.classes_))

    print("\n=== VAL symbol/timeframe kesimida ===")
    val_df = dataset[val_mask].copy()
    val_df["pred"] = preds
    for (symbol, timeframe), grp in val_df.groupby(["symbol", "timeframe"]):
        acc = accuracy_score(grp["label"], grp["pred"])
        print(f"  {symbol} {timeframe}: n={len(grp)} accuracy={acc:.3f}")
    print("TEST saqlab qo'yildi; parametrlar validation'da tanlangach backtest.py orqali tekshiring.")

    importances = sorted(zip(mt5_context.FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    print("\nFeature importance:")
    for name, imp in importances:
        print(f"  {name}: {imp:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    model_out = os.path.join(args.out_dir, "numeric_model.joblib")
    meta_out = os.path.join(args.out_dir, "numeric_model_meta.json")
    joblib.dump(model, model_out)
    provenance = dataset_provenance(labeled)
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump({"feature_names": mt5_context.FEATURE_NAMES, "classes": list(model.classes_),
                   "model_sha256": file_sha256(model_out), "training_split": "train",
                   "provenance_status": "verified_candidate", "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                   "sklearn_version": sklearn.__version__, "splits": split_summary(labeled),
                   "validation": validation_metrics, "test_evaluated": False,
                   "evaluation_limitations": ["Existing historical periods were previously inspected; rebuilt test is not a new independent holdout.",
                                              "Classification/backtest does not establish Gemini or final consensus quality.",
                                              "OHLC backtests exclude spread, commission and slippage."],
                   "dataset": provenance}, f, indent=2)

    print(f"\nCandidate saqlandi -> {model_out} (faol model o'zgarmadi)")


if __name__ == "__main__":
    main()
