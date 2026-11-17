"""Custom model tadqiqoti: validation'da parametrlarni ko'rish, test'da baholash.

TEST uchun validation'da oldindan tanlangan --confidence va --tp-ratio talab
qilinadi. Candidate model va dataset hash'lari tekshiriladi. Bu natija Gemini
bilan birgalikdagi signal sifatini yoki haqiqiy daromadni tasdiqlamaydi.
Spread, komissiya va slippage bu OHLC simulyatsiyasiga kiritilmagan.
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
import mt5_context  # noqa: E402
try:
    from .train_numeric_model import (DEFAULT_OUT_DIR, COMBINED_LABELED, MULTI_DIR,
                                      validate_labeled, dataset_provenance, file_sha256)
except ImportError:
    from train_numeric_model import (DEFAULT_OUT_DIR, COMBINED_LABELED, MULTI_DIR,
                                     validate_labeled, dataset_provenance, file_sha256)

LOOKBACK_BARS = 100
MIN_HISTORY = 55
LOOKAHEAD_BARS = 24
CONFIDENCE_THRESHOLDS = [0.34, 0.45, 0.55, 0.65, 0.75]
TP_RATIOS = [1.0, 1.5, 2.0, 2.5, 3.0]


def simulate_outcome(highs, lows, idx, entry, atr, signal, tp_ratio, sl_ratio=1.0):
    if signal not in ("Buy", "Sell") or not np.isfinite(atr) or atr <= 0:
        raise ValueError("Yaroqsiz signal yoki ATR")
    tp = entry + (1 if signal == "Buy" else -1) * tp_ratio * atr
    sl = entry - (1 if signal == "Buy" else -1) * sl_ratio * atr
    for j in range(idx + 1, min(idx + 1 + LOOKAHEAD_BARS, len(highs), len(lows))):
        h, l = highs[j], lows[j]
        if not np.isfinite(h) or not np.isfinite(l):
            return "incomplete"
        hit_tp = h >= tp if signal == "Buy" else l <= tp
        hit_sl = l <= sl if signal == "Buy" else h >= sl
        if hit_sl:
            return "loss"  # bitta barda ikkalasi bo'lsa konservativ
        if hit_tp:
            return "win"
    if idx + LOOKAHEAD_BARS >= min(len(highs), len(lows)):
        return "incomplete"
    return "expired"


def verify_provenance(model_path, metadata_path, labeled):
    with open(metadata_path, encoding="utf-8") as source:
        meta = json.load(source)
    if (meta.get("training_split") != "train" or
            meta.get("model_sha256") != file_sha256(model_path) or
            meta.get("dataset") != dataset_provenance(labeled)):
        raise ValueError("Model yoki dataset provenance mos emas. Yangi candidate o'qiting.")


def summarize(predictions, raw_cache, confidence, tp_ratio):
    wins = losses = expired = incomplete = 0
    expired_r = 0.0
    for p in predictions:
        if p["confidence"] < confidence:
            continue
        raw = raw_cache[(p["symbol"], p["timeframe"])]
        outcome = simulate_outcome(raw["high"], raw["low"], p["idx"], p["entry"], p["atr"], p["signal"], tp_ratio)
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        elif outcome == "expired":
            close = raw["close"][p["idx"] + LOOKAHEAD_BARS]
            if not np.isfinite(close):
                incomplete += 1
                continue
            expired += 1
            direction = 1 if p["signal"] == "Buy" else -1
            expired_r += direction * (close - p["entry"]) / p["atr"]
        else:
            incomplete += 1
    completed = wins + losses + expired
    # Expiry uses horizon close as a hypothetical exit; incomplete paths excluded.
    mean_r = (wins * tp_ratio - losses + expired_r) / completed if completed else None
    return {"wins": wins, "losses": losses, "expired": expired,
            "incomplete": incomplete, "mean_r": mean_r}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--tp-ratio", type=float)
    parser.add_argument("--model-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if (args.confidence is None) != (args.tp_ratio is None):
        parser.error("--confidence va --tp-ratio birgalikda beriladi")
    if args.split == "test" and args.confidence is None:
        parser.error("Test uchun validation'da tanlangan --confidence va --tp-ratio kerak")
    if args.confidence is not None and (not np.isfinite(args.confidence) or not 0 <= args.confidence <= 1 or
                                        not np.isfinite(args.tp_ratio) or args.tp_ratio <= 0):
        parser.error("Confidence 0..1 va TP ratio musbat bo'lishi kerak")

    labeled = pd.read_parquet(COMBINED_LABELED)
    validate_labeled(labeled)
    model_path = os.path.join(args.model_dir, "numeric_model.joblib")
    meta_path = os.path.join(args.model_dir, "numeric_model_meta.json")
    verify_provenance(model_path, meta_path, labeled)
    model = joblib.load(model_path)
    predictions = []
    raw_cache = {}
    selected = labeled[labeled["split"] == args.split]
    for (symbol, timeframe), group in selected.groupby(["symbol", "timeframe"]):
        frame = pd.read_parquet(os.path.join(MULTI_DIR, f"{symbol}_{timeframe}_raw.parquet"))
        raw = {name: frame[name].to_numpy(dtype=np.float64) for name in ("high", "low", "close")}
        raw_cache[(symbol, timeframe)] = raw
        for row in group.itertuples():
            idx = int(row.window_end_idx)
            start = max(0, idx - LOOKBACK_BARS + 1)
            if idx - start < MIN_HISTORY:
                continue
            raw_ind = mt5_context.compute_raw_indicators(
                raw["high"][start:idx + 1], raw["low"][start:idx + 1], raw["close"][start:idx + 1],
                bar_hour=pd.to_datetime(frame["time"].iloc[idx]).hour)
            features = pd.DataFrame([mt5_context.to_feature_vector(raw_ind)], columns=mt5_context.FEATURE_NAMES)
            probs = model.predict_proba(features)[0]
            class_probs = dict(zip(model.classes_, probs))
            signal = max(class_probs, key=class_probs.get)
            if signal != "Hold":
                predictions.append({"symbol": symbol, "timeframe": timeframe, "idx": idx,
                                    "signal": signal, "confidence": class_probs[signal],
                                    "entry": raw_ind["last_close"], "atr": raw_ind["atr_14"]})

    thresholds = [args.confidence] if args.confidence is not None else CONFIDENCE_THRESHOLDS
    ratios = [args.tp_ratio] if args.tp_ratio is not None else TP_RATIOS
    print(f"Split: {args.split}; Buy/Sell: {len(predictions)}")
    print("Expiry: 24-bar oxirgi close bo'yicha shartli yopish; incomplete hisobga olinmaydi.")
    print("Spread, komissiya va slippage yo'q. Gemini/final signal backtest qilinmagan.")
    print("Test natijasiga qarab parametr tanlash uni mustaqil test sifatida yaroqsiz qiladi.")
    print(f"{'conf>=':>7} {'TP:SL':>7} {'win':>5} {'loss':>5} {'expiry':>7} {'no_data':>8} {'mean_R':>9}")
    for threshold in thresholds:
        for ratio in ratios:
            result = summarize(predictions, raw_cache, threshold, ratio)
            mean = f"{result['mean_r']:+.3f}" if result["mean_r"] is not None else "-"
            print(f"{threshold:>7.2f} {ratio:>6.1f}:1 {result['wins']:>5} {result['losses']:>5} "
                  f"{result['expired']:>7} {result['incomplete']:>8} {mean:>9}")


if __name__ == "__main__":
    main()
