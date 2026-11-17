"""Bir nechta symbol va timeframe uchun MT5 tarixini yuklab, triple-barrier
label yaratadi -- FAQAT raqamli model uchun. Barcha symbol va timeframe'lar
umumiy vaqt chegaralari bilan ajratiladi, chegaradan o'tuvchi label'lar chiqariladi.
"""
import os

import numpy as np
import pandas as pd

try:
    from .mt5_fetch import fetch_ohlc
    from . import config
except ImportError:  # to'g'ridan-to'g'ri skript sifatida ishga tushirish
    from mt5_fetch import fetch_ohlc
    import config

(
    WINDOW_SIZE,
    WINDOW_STRIDE,
    LOOKAHEAD_BARS,
    ATR_PERIOD,
    TP_ATR_MULT,
    SL_ATR_MULT,
    TRAIN_RATIO,
    VAL_RATIO,
) = (config.WINDOW_SIZE, config.WINDOW_STRIDE, config.LOOKAHEAD_BARS,
     config.ATR_PERIOD, config.TP_ATR_MULT, config.SL_ATR_MULT,
     config.TRAIN_RATIO, config.VAL_RATIO)

# XAUUSD (M15) allaqachon data_pipeline/output/ohlc_raw.parquet + labeled_windows.parquet
# ichida bor (2995 qator) -- bu yerda YANA shu juftlik qo'shiladi (H1 bilan birga),
# chunki bir xil funksiyalar ishlatilishi kerak (train/serve mosligi uchun).
SYMBOL_TIMEFRAMES = [
    ("XAUUSD", "M15"),
    ("XAUUSD", "H1"),
    ("EURUSD", "M15"),
    ("EURUSD", "H1"),
    ("GBPUSD", "M15"),
    ("GBPUSD", "H1"),
    ("USDJPY", "M15"),
    ("USDJPY", "H1"),
]

HISTORY_BARS = 20_000  # ko'p symbol uchun tezroq bo'lishi uchun 60000 dan kamroq

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

        # A bar does not reveal ordering; if both levels hit, treat it as a loss.
        buy_success = buy_win_idx is not None and (buy_lose_idx is None or buy_win_idx < buy_lose_idx)
        sell_success = sell_win_idx is not None and (sell_lose_idx is None or sell_win_idx < sell_lose_idx)

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
            # The last lookahead candle can touch a barrier until it closes.
            "label_end_time": pd.Timestamp(df["time"].iloc[i + LOOKAHEAD_BARS]) + timeframe_delta(timeframe),
            "label": label,
            "label_version": 2,
        })

    return pd.DataFrame(rows)


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    unit = {"M": "min", "H": "h", "D": "d"}
    if not timeframe or timeframe[0] not in unit or not timeframe[1:].isdigit() or int(timeframe[1:]) < 1:
        raise ValueError(f"Noma'lum timeframe: {timeframe}")
    return pd.Timedelta(int(timeframe[1:]), unit=unit[timeframe[0]])


def assign_shared_splits(labeled: pd.DataFrame) -> pd.DataFrame:
    """One chronological split across instruments; purge labels crossing a boundary."""
    if labeled.empty:
        raise ValueError("Label'lar yo'q; split yaratib bo'lmadi")
    out = labeled.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out["label_end_time"] = pd.to_datetime(out["label_end_time"], utc=True)
    if out[["time", "label_end_time"]].isna().any().any():
        raise ValueError("Vaqti noma'lum label mavjud")
    times = pd.Series(out["time"].unique()).sort_values().reset_index(drop=True)
    if len(times) < 3:
        raise ValueError("Split uchun vaqtlar yetarli emas")
    val_start = times.iloc[int(len(times) * TRAIN_RATIO)]
    test_start = times.iloc[int(len(times) * (TRAIN_RATIO + VAL_RATIO))]
    out["split"] = np.where(out["time"] < val_start, "train",
                     np.where(out["time"] < test_start, "val", "test"))
    crosses = ((out["split"] == "train") & (out["label_end_time"] > val_start) |
               (out["split"] == "val") & (out["label_end_time"] > test_start))
    out = out.loc[~crosses].copy()
    if set(out["split"]) != {"train", "val", "test"}:
        raise ValueError("Purge'dan keyin barcha split'lar uchun ma'lumot yetarli emas")
    return out


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
        print(f"  {len(raw)} bar, {len(labeled)} label'langan window")
        print(f"  {labeled['label'].value_counts().to_dict()}")
        all_labeled.append(labeled)

    combined = assign_shared_splits(pd.concat(all_labeled, ignore_index=True))
    for (symbol, timeframe), group in combined.groupby(["symbol", "timeframe"]):
        group.to_parquet(os.path.join(OUT_DIR, f"{symbol}_{timeframe}_labeled.parquet"), index=False)
    combined.to_parquet(COMBINED_OUT, index=False)
    print(f"\nJami: {len(combined)} qator -> {COMBINED_OUT}")
    print(combined.groupby(["symbol", "timeframe"])["label"].count())


if __name__ == "__main__":
    main()
