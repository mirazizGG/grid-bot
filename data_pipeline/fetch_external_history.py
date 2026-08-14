"""ejtraderLabs/historical-data (GitHub, ochiq/bepul) dan qo'shimcha tarixiy
OHLC ma'lumot yuklab, bizning MT5'dan olingan ma'lumotga BIRLASHTIRADI --
shu bilan o'qitish datasetini ~2.5 yildan ~10+ yilga kengaytiradi.

Manba: https://github.com/ejtraderLabs/historical-data (MIT-uslub litsenziya,
umumiy narx ma'lumoti, mualliflik huquqi ostidagi kontent emas).
"""
import io
import os

import numpy as np
import pandas as pd
import requests

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_DIR = os.path.join(_THIS_DIR, "output", "multi")

BASE_URL = "https://raw.githubusercontent.com/ejtraderLabs/historical-data/main"

# Broker "point" formatidan haqiqiy narxga o'tkazish uchun ko'paytiruvchi
SCALE = {
    "XAUUSD": 100,
    "EURUSD": 100_000,
    "GBPUSD": 100_000,
    "USDJPY": 1_000,
}

TF_SUFFIX = {"M15": "m15", "H1": "h1"}


def download_external(symbol: str, timeframe: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{symbol}/{symbol}{TF_SUFFIX[timeframe]}.csv"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))

    scale = SCALE[symbol]
    df["time"] = pd.to_datetime(df["Date"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col] / scale
    df["volume"] = df["tick_volume"]
    return df[["time", "open", "high", "low", "close", "volume"]].sort_values("time").reset_index(drop=True)


def merge_with_mt5(symbol: str, timeframe: str) -> pd.DataFrame:
    mt5_path = os.path.join(MULTI_DIR, f"{symbol}_{timeframe}_raw.parquet")
    mt5_df = pd.read_parquet(mt5_path)

    external_df = download_external(symbol, timeframe)

    combined = pd.concat([external_df, mt5_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return combined


def main():
    os.makedirs(MULTI_DIR, exist_ok=True)
    for symbol in SCALE:
        for timeframe in TF_SUFFIX:
            print(f"--- {symbol} {timeframe} ---")
            try:
                combined = merge_with_mt5(symbol, timeframe)
            except Exception as exc:
                print(f"  XATO (o'tkazib yuborildi): {exc}")
                continue

            out_path = os.path.join(MULTI_DIR, f"{symbol}_{timeframe}_raw.parquet")
            combined.to_parquet(out_path, index=False)
            print(f"  {len(combined)} bar jami "
                  f"({combined['time'].min()} .. {combined['time'].max()})")


if __name__ == "__main__":
    main()
