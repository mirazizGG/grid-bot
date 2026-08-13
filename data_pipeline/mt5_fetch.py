"""MT5 terminalidan XAUUSD M15 tarixiy OHLC ma'lumotini yuklab, parquet sifatida saqlaydi.

Ishga tushirishdan oldin MetaTrader5 terminali ochiq va hisobga kirilgan bo'lishi kerak.
"""
import os
import sys

import MetaTrader5 as mt5
import pandas as pd

from config import SYMBOL, TIMEFRAME_NAME, HISTORY_BARS, RAW_DATA_PATH

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


def fetch_ohlc(symbol: str, timeframe_name: str, n_bars: int) -> pd.DataFrame:
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() muvaffaqiyatsiz: {mt5.last_error()}")

    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Symbol tanlab bo'lmadi: {symbol} ({mt5.last_error()})")

        timeframe = TIMEFRAME_MAP[timeframe_name]
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Ma'lumot olinmadi: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"tick_volume": "volume"})
        df = df[["time", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("time").reset_index(drop=True)
        return df
    finally:
        mt5.shutdown()


def main():
    df = fetch_ohlc(SYMBOL, TIMEFRAME_NAME, HISTORY_BARS)
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_parquet(RAW_DATA_PATH, index=False)
    print(f"{len(df)} ta sham saqlandi -> {RAW_DATA_PATH}")
    print(f"Sana oralig'i: {df['time'].min()} .. {df['time'].max()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"XATO: {exc}", file=sys.stderr)
        sys.exit(1)
