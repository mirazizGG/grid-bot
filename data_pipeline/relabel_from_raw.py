"""Mavjud (endi tarixiy ma'lumot bilan kengaytirilgan) raw parquet fayllarni
MT5'ga qayta ulanmasdan qayta label'laydi -- multi_symbol_dataset.py'ning
label_windows() funksiyasidan foydalanadi."""
import os

import pandas as pd

from multi_symbol_dataset import label_windows, OUT_DIR, COMBINED_OUT

SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAMES = ["M15", "H1"]


def main():
    all_labeled = []
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            raw_path = os.path.join(OUT_DIR, f"{symbol}_{timeframe}_raw.parquet")
            raw = pd.read_parquet(raw_path)
            print(f"--- {symbol} {timeframe}: {len(raw)} bar ---")

            labeled = label_windows(raw, symbol, timeframe)
            labeled_path = os.path.join(OUT_DIR, f"{symbol}_{timeframe}_labeled.parquet")
            labeled.to_parquet(labeled_path, index=False)

            print(f"  {len(labeled)} label'langan window, {labeled['label'].value_counts().to_dict()}")
            all_labeled.append(labeled)

    combined = pd.concat(all_labeled, ignore_index=True)
    combined.to_parquet(COMBINED_OUT, index=False)
    print(f"\nJami: {len(combined)} qator -> {COMBINED_OUT}")


if __name__ == "__main__":
    main()
