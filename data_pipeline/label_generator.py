"""Har bir sliding window uchun triple-barrier usulida Buy/Sell/Hold label va TP/SL hisoblaydi.

Har bir window oxiridagi barda (entry) ATR asosida ikkita ssenariy sinaladi:
  - Buy: narx avval (entry + TP*ATR) ga yetsa -> yutuq, (entry - SL*ATR) ga yetsa -> yutqizish
  - Sell: narx avval (entry - TP*ATR) ga yetsa -> yutuq, (entry + SL*ATR) ga yetsa -> yutqizish
Faqat bitta tomon yutgan bo'lsa o'sha label beriladi; ikkalasi ham yutsa qaysi barvaqt
sodir bo'lgani tanlanadi; hech biri yutmasa Hold.
"""
import os

import numpy as np
import pandas as pd

from config import (
    RAW_DATA_PATH,
    LABELED_DATA_PATH,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    LOOKAHEAD_BARS,
    ATR_PERIOD,
    TP_ATR_MULT,
    SL_ATR_MULT,
    TRAIN_RATIO,
    VAL_RATIO,
)


def compute_atr(df: pd.DataFrame, period: int) -> np.ndarray:
    high, low, close = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().to_numpy()
    return atr


def label_windows(df: pd.DataFrame) -> pd.DataFrame:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr = compute_atr(df, ATR_PERIOD)

    n = len(df)
    first_valid = max(WINDOW_SIZE - 1, ATR_PERIOD)
    last_valid = n - LOOKAHEAD_BARS - 1

    rows = []
    for i in range(first_valid, last_valid, WINDOW_STRIDE):
        entry = close[i]
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue

        buy_tp = entry + TP_ATR_MULT * a
        buy_sl = entry - SL_ATR_MULT * a
        sell_tp = entry - TP_ATR_MULT * a
        sell_sl = entry + SL_ATR_MULT * a

        buy_win_idx = None
        buy_lose_idx = None
        sell_win_idx = None
        sell_lose_idx = None

        for j in range(i + 1, i + 1 + LOOKAHEAD_BARS):
            h, l = high[j], low[j]
            if buy_win_idx is None and h >= buy_tp:
                buy_win_idx = j
            if buy_lose_idx is None and l <= buy_sl:
                buy_lose_idx = j
            if sell_win_idx is None and l <= sell_tp:
                sell_win_idx = j
            if sell_lose_idx is None and h >= sell_sl:
                sell_lose_idx = j
            if None not in (buy_win_idx, buy_lose_idx, sell_win_idx, sell_lose_idx):
                break

        buy_success = buy_win_idx is not None and (buy_lose_idx is None or buy_win_idx <= buy_lose_idx)
        sell_success = sell_win_idx is not None and (sell_lose_idx is None or sell_win_idx <= sell_lose_idx)

        if buy_success and not sell_success:
            label = "Buy"
        elif sell_success and not buy_success:
            label = "Sell"
        elif buy_success and sell_success:
            label = "Buy" if buy_win_idx <= sell_win_idx else "Sell"
        else:
            label = "Hold"

        rows.append(
            {
                "window_end_idx": i,
                "window_start_idx": i - WINDOW_SIZE + 1,
                "time": df["time"].iloc[i],
                "entry_price": entry,
                "atr": a,
                "label": label,
                "tp_price": TP_ATR_MULT * a,
                "sl_price": SL_ATR_MULT * a,
            }
        )

    return pd.DataFrame(rows)


def chronological_split(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    split = np.empty(n, dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "val"
    split[val_end:] = "test"
    df = df.copy()
    df["split"] = split
    return df


def main():
    df = pd.read_parquet(RAW_DATA_PATH)
    labeled = label_windows(df)
    labeled = chronological_split(labeled)

    os.makedirs(os.path.dirname(LABELED_DATA_PATH), exist_ok=True)
    labeled.to_parquet(LABELED_DATA_PATH, index=False)

    print(f"{len(labeled)} ta window label'landi -> {LABELED_DATA_PATH}")
    print(labeled["label"].value_counts())
    print(labeled.groupby("split")["label"].value_counts())


if __name__ == "__main__":
    main()
