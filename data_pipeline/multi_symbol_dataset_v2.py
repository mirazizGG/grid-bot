"""multi_symbol_dataset.py bilan bir xil, lekin ANCHA ko'proq tarixiy data bilan
(MT5 terminal maxbars=100000 chegarasigacha) -- "ko'proq data custom model
aniqligini oshiradimi" tajribasi uchun.

Joriy deploy qilingan datasetga (output/multi/, output/multi_labeled.parquet)
TEGMAYDI -- alohida output/multi_v2/ papkasiga yozadi, shunda joriy model
xavfsiz qoladi va ikkalasini keyin solishtirish mumkin.
"""
import os

import pandas as pd

from mt5_fetch import fetch_ohlc
from config import ATR_PERIOD, TP_ATR_MULT, SL_ATR_MULT, TRAIN_RATIO, VAL_RATIO
from multi_symbol_dataset import label_windows, SYMBOL_TIMEFRAMES

HISTORY_BARS = 99_999  # MT5 terminal maxbars=100000 chegarasi (aniq 100000 "Invalid params" beradi) -- avvalgi 20_000dan 5x ko'p

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "multi_v2")
COMBINED_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "multi_v2_labeled.parquet")


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

        print(f"  {len(raw)} bar ({raw['time'].min()} .. {raw['time'].max()}), {len(labeled)} label'langan window")
        print(f"  {labeled['label'].value_counts().to_dict()}")
        all_labeled.append(labeled)

    combined = pd.concat(all_labeled, ignore_index=True)
    combined.to_parquet(COMBINED_OUT, index=False)
    print(f"\nJami: {len(combined)} qator -> {COMBINED_OUT}")
    print(combined.groupby(["symbol", "timeframe"])["label"].count())


if __name__ == "__main__":
    main()
