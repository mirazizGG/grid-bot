"""Haqiqiy (out-of-sample TEST split) tarixiy ma'lumotda backtest:
turli ishonch chegarasi va TP:SL nisbatlarida custom model qanday
win-rate va expectancy berishini hisoblaydi.

Eslatma: bu FAQAT custom model'ni backtest qiladi (Gemini'ni tarixiy
ravishda minglab so'rov bilan qayta ishga tushirish amaliy emas -- API
xarajati va vaqt talab qiladi). Haqiqiy deploy qilingan tizim (custom
model + Gemini) buni kelishgan holatlarda kuchaytiradi, shuning uchun bu
yerdagi natijalar "pastki chegara" hisoblanadi.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
import mt5_context  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_DIR = os.path.join(_THIS_DIR, "..", "data_pipeline", "output", "multi")
COMBINED_LABELED = os.path.join(_THIS_DIR, "..", "data_pipeline", "output", "multi_labeled.parquet")
MODEL_PATH = os.path.join(_THIS_DIR, "..", "backend", "model", "numeric_model.joblib")

LOOKBACK_BARS = 100
MIN_HISTORY = 55
LOOKAHEAD_BARS = 24  # data_pipeline/config.py bilan bir xil

CONFIDENCE_THRESHOLDS = [0.34, 0.45, 0.55, 0.65, 0.75]
TP_RATIOS = [1.0, 1.5, 2.0, 2.5, 3.0]  # SL = 1.0 * ATR doim


def simulate_outcome(highs, lows, idx, entry, atr, signal, tp_ratio, sl_ratio=1.0):
    if signal == "Buy":
        tp, sl = entry + tp_ratio * atr, entry - sl_ratio * atr
    else:  # Sell
        tp, sl = entry - tp_ratio * atr, entry + sl_ratio * atr

    for j in range(idx + 1, min(idx + 1 + LOOKAHEAD_BARS, len(highs))):
        h, l = highs[j], lows[j]
        if signal == "Buy":
            hit_tp, hit_sl = h >= tp, l <= sl
        else:
            hit_tp, hit_sl = l <= tp, h >= sl
        if hit_tp and hit_sl:
            return "loss"  # ikkalasi ham bitta barda -- ehtiyotkorlik uchun loss deb hisoblaymiz
        if hit_tp:
            return "win"
        if hit_sl:
            return "loss"
    return "none"  # muddat tugadi, na TP na SL urildi


def main():
    model = joblib.load(MODEL_PATH)
    labeled = pd.read_parquet(COMBINED_LABELED)
    test_rows = labeled[labeled["split"] == "test"]

    predictions = []  # (signal, confidence, idx, symbol, timeframe)
    raw_cache = {}

    for (symbol, timeframe), group in test_rows.groupby(["symbol", "timeframe"]):
        raw = pd.read_parquet(os.path.join(MULTI_DIR, f"{symbol}_{timeframe}_raw.parquet"))
        raw_cache[(symbol, timeframe)] = raw
        opens_all = raw["open"].to_numpy(dtype=np.float64)
        highs_all = raw["high"].to_numpy(dtype=np.float64)
        lows_all = raw["low"].to_numpy(dtype=np.float64)
        closes_all = raw["close"].to_numpy(dtype=np.float64)
        times_all = raw["time"]

        for _, row in group.iterrows():
            idx = int(row["window_end_idx"])
            start = max(0, idx - LOOKBACK_BARS + 1)
            if idx - start < MIN_HISTORY:
                continue
            h, l, c = highs_all[start:idx + 1], lows_all[start:idx + 1], closes_all[start:idx + 1]
            bar_hour = pd.to_datetime(times_all.iloc[idx]).hour
            raw_ind = mt5_context.compute_raw_indicators(h, l, c, bar_hour=bar_hour)
            features = [mt5_context.to_feature_vector(raw_ind)]

            probs = model.predict_proba(features)[0]
            class_probs = dict(zip(model.classes_, probs))
            signal = max(class_probs, key=class_probs.get)
            confidence = class_probs[signal]

            if signal == "Hold":
                continue

            predictions.append({
                "symbol": symbol, "timeframe": timeframe, "idx": idx,
                "signal": signal, "confidence": confidence,
                "entry": raw_ind["last_close"], "atr": raw_ind["atr_14"],
            })

    pred_df = pd.DataFrame(predictions)
    print(f"Jami Buy/Sell signal (Hold'siz): {len(pred_df)}\n")

    print(f"{'conf>=':>7} {'TP:SL':>7} {'n_trades':>9} {'win_rate':>9} {'expectancy(R)':>14}")
    print("-" * 55)
    for conf_th in CONFIDENCE_THRESHOLDS:
        subset = pred_df[pred_df["confidence"] >= conf_th]
        for tp_ratio in TP_RATIOS:
            wins = losses = 0
            for _, p in subset.iterrows():
                raw = raw_cache[(p["symbol"], p["timeframe"])]
                outcome = simulate_outcome(
                    raw["high"].to_numpy(dtype=np.float64), raw["low"].to_numpy(dtype=np.float64),
                    p["idx"], p["entry"], p["atr"], p["signal"], tp_ratio,
                )
                if outcome == "win":
                    wins += 1
                elif outcome == "loss":
                    losses += 1
            total = wins + losses
            if total == 0:
                continue
            win_rate = wins / total
            expectancy = win_rate * tp_ratio - (1 - win_rate) * 1.0
            print(f"{conf_th:>7.2f} {tp_ratio:>6.1f}:1 {total:>9} {win_rate:>8.1%} {expectancy:>+13.3f}")


if __name__ == "__main__":
    main()
