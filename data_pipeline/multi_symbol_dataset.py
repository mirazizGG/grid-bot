"""Bir nechta symbol va timeframe uchun MT5 tarixini yuklab, triple-barrier
label yaratadi -- FAQAT raqamli model uchun (rasm generatsiya qilinmaydi,
shuning uchun tez ishlaydi). Har (symbol, timeframe) juftligi alohida
chronological split bilan label'lanadi, so'ng bittalashtirilgan parquet'ga
yoziladi -- training/train_numeric_model.py shuni o'qiydi.
"""
import os

import numpy as np
import pandas as pd

from mt5_fetch import fetch_ohlc
from config import (
    WINDOW_SIZE,
    WINDOW_STRIDE,
    LOOKAHEAD_BARS,
    ATR_PERIOD,
    TP_ATR_MULT,
    SL_ATR_MULT,
    TRAIN_RATIO,
    VAL_RATIO,
)

# Faqat XAUUSD (foydalanuvchi shu bilan savdo qiladi) -- M15 va H1, imkon
# qadar ko'proq tarix bilan.
SYMBOL_TIMEFRAMES = [
    ("XAUUSD", "M15"),
    ("XAUUSD", "H1"),
]

HISTORY_BARS = 60_000

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "multi")
COMBINED_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "multi_labeled.parquet")


def compute_atr_series(df: pd.DataFrame, period: int) -> np.ndarray:
    high, low, close = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr).rolling(window=period, min_periods=period).mean().to_numpy()


def label_windows(df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr = compute_atr_series(df, ATR_PERIOD)

    n = len(df)
    first_valid = max(WINDOW_SIZE - 1, ATR_PERIOD)
    last_valid = n - LOOKAHEAD_BARS - 1

    rows = []
    for i in range(first_valid, last_valid, WINDOW_STRIDE):
        entry = close[i]
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue

        buy_tp, buy_sl = entry + TP_ATR_MULT * a, entry - SL_ATR_MULT * a
        sell_tp, sell_sl = entry - TP_ATR_MULT * a, entry + SL_ATR_MULT * a

        buy_win_idx = buy_lose_idx = sell_win_idx = sell_lose_idx = None
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

        rows.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "window_end_idx": i,
            "time": df["time"].iloc[i],
            "label": label,
        })

    labeled = pd.DataFrame(rows)
    n_rows = len(labeled)
    train_end = int(n_rows * TRAIN_RATIO)
    val_end = train_end + int(n_rows * VAL_RATIO)
    split = np.empty(n_rows, dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "val"
    split[val_end:] = "test"
    labeled["split"] = split
    return labeled


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_labeled = []

    for symbol, timeframe in SYMBOL_TIMEFRAMES:
        print(f"--- {symbol} {timeframe} ---")
        try:
            raw = fetch_ohlc(symbol, timeframe, HISTORY_BARS)
        except Exception as exc:
            print(f"  XATO (o'tkazib yuborildi): {exc}")
            continue

        raw_path = os.path.join(OUT_DIR, f"{symbol}_{timeframe}_raw.parquet")
        raw.to_parquet(raw_path, index=False)

        labeled = label_windows(raw, symbol, timeframe)
        labeled_path = os.path.join(OUT_DIR, f"{symbol}_{timeframe}_labeled.parquet")
        labeled.to_parquet(labeled_path, index=False)

        print(f"  {len(raw)} bar, {len(labeled)} label'langan window")
        print(f"  {labeled['label'].value_counts().to_dict()}")
        all_labeled.append(labeled)

    combined = pd.concat(all_labeled, ignore_index=True)
    combined.to_parquet(COMBINED_OUT, index=False)
    print(f"\nJami: {len(combined)} qator -> {COMBINED_OUT}")
    print(combined.groupby(["symbol", "timeframe"])["label"].count())


if __name__ == "__main__":
    main()
