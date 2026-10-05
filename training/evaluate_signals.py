"""Bazadagi (backend/analyses.db) signallarni MT5 tarixidagi keyingi narx
harakati bilan solishtirib, haqiqiy natijani (TP urildimi / SL urildimi)
hisoblaydi.

- Buy/Sell signallar: saqlangan entry/TP/SL bo'yicha baholanadi.
- Hold signallar: Gemini yo'nalishi (agar Buy/Sell bo'lsa) bo'yicha ATR 2:1
  darajalar bilan "agar signal berilganida" nima bo'lardi -- gipotetik baholanadi.
  Shu bilan Hold filtri foyda berayotganini tekshirish mumkin.

Baholash 1-minutlik barlar bilan, TP:SL = +2R:-1R. Bir barda ham TP, ham SL
urilsa konservativ ravishda SL deb hisoblanadi. Spred hisobga olinmaydi
(Buy ask'dan kiradi, lekin chiqish bid barlari bilan tekshiriladi -- bu spredni
qisman hisobga oladi, Sell uchun esa aksincha yengilroq).

Ishga tushirish (MT5 terminali ochiq bo'lishi kerak):
    python training/evaluate_signals.py [--horizon-hours 24] [--utc-offset 3]
"""
import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
import time

BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

import MetaTrader5 as mt5  # noqa: E402
import numpy as np  # noqa: E402

from mt5_context import compute_trade_levels, TP_ATR_MULT, SL_ATR_MULT  # noqa: E402

DB_PATH = os.path.join(BACKEND_DIR, "analyses.db")


def simulate(direction: str, tp: float, sl: float, highs, lows) -> str:
    """Barlar bo'ylab yurib, birinchi urilgan darajani qaytaradi: TP / SL / OPEN."""
    for h, l in zip(highs, lows):
        if direction == "Buy":
            hit_sl, hit_tp = l <= sl, h >= tp
        else:
            hit_sl, hit_tp = h >= sl, l <= tp
        if hit_sl:  # bir barda ikkalasi ham bo'lsa -- konservativ: SL
            return "SL"
        if hit_tp:
            return "TP"
    return "OPEN"


def detect_server_offset_hours() -> int | None:
    """Broker server vaqti va UTC farqini (soatda) aniqlaydi. Bozor yopiq bo'lsa
    (tick eskirgan) None qaytaradi."""
    tick = mt5.symbol_info_tick("XAUUSD")
    if tick is None or abs(time.time() - tick.time) > 3600 * 6:
        return None
    return round((tick.time - time.time()) / 3600)


def load_rows():
    con = sqlite3.connect(DB_PATH)
    try:
        rows = con.execute("select id, timestamp, final_signal, raw_result from analyses order by id").fetchall()
    finally:
        con.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-hours", type=float, default=24.0, help="signaldan keyin qancha vaqt kuzatiladi")
    ap.add_argument("--utc-offset", type=int, default=None, help="broker server vaqtining UTC'dan farqi (soat); bo'sh bo'lsa avtomatik")
    args = ap.parse_args()

    if not mt5.initialize():
        sys.exit("MT5 terminaliga ulanib bo'lmadi (terminal ochiqmi?)")

    offset = args.utc_offset if args.utc_offset is not None else detect_server_offset_hours()
    if offset is None:
        mt5.shutdown()
        sys.exit("Server vaqt farqini aniqlab bo'lmadi (bozor yopiq?). --utc-offset bering.")
    print(f"Broker server vaqti = UTC{offset:+d}; kuzatuv oynasi {args.horizon_hours:g} soat\n")

    results = []
    for rid, ts, final_signal, raw in load_rows():
        data = json.loads(raw)
        ctx = data.get("market_context")
        if not ctx:
            results.append((rid, ts, final_signal, None, "MT5 konteksti yo'q", None))
            continue

        if final_signal in ("Buy", "Sell"):
            direction, kind = final_signal, "signal"
            entry, tp, sl = data["entry_price"], data["tp_price"], data["sl_price"]
        else:
            direction = data.get("vision_ai", {}).get("signal")
            if direction not in ("Buy", "Sell"):
                results.append((rid, ts, final_signal, None, "Hold, Gemini ham Hold", None))
                continue
            kind = "hold-gipotetik"
            lv = compute_trade_levels(ctx, direction)
            entry, tp, sl = lv["entry_price"], lv["tp_price"], lv["sl_price"]

        start = dt.datetime.fromisoformat(ts) + dt.timedelta(hours=offset)  # UTC -> server vaqti
        end = start + dt.timedelta(hours=args.horizon_hours)
        bars = mt5.copy_rates_range(ctx["symbol"], mt5.TIMEFRAME_M1, start, end)
        if bars is None or len(bars) == 0:
            results.append((rid, ts, final_signal, direction, f"{kind}: tarix yo'q", None))
            continue

        outcome = simulate(direction, tp, sl, bars["high"], bars["low"])
        results.append((rid, ts, final_signal, direction, kind, outcome))

    mt5.shutdown()

    print(f"{'id':>3}  {'vaqt (UTC)':<19}  {'final':<5} {'yo`nalish':<8} {'tur':<16} natija")
    for rid, ts, final, direction, kind, outcome in results:
        print(f"{rid:>3}  {ts[:19]:<19}  {final:<5} {direction or '-':<8} {kind:<16} {outcome or '-'}")

    print("\n=== Xulosa (TP=+%gR, SL=-%gR, OPEN hisobga olinmaydi) ===" % (TP_ATR_MULT / SL_ATR_MULT, 1))
    for label in ("signal", "hold-gipotetik"):
        subset = [r for r in results if r[4] == label and r[5] in ("TP", "SL")]
        open_n = sum(1 for r in results if r[4] == label and r[5] == "OPEN")
        n = len(subset)
        if n == 0:
            print(f"{label:<16} n=0 (OPEN: {open_n})")
            continue
        wins = sum(1 for r in subset if r[5] == "TP")
        total_r = wins * (TP_ATR_MULT / SL_ATR_MULT) - (n - wins)
        print(f"{label:<16} n={n}  TP={wins}  SL={n - wins}  win-rate={wins / n:.0%}  "
              f"jami={total_r:+.1f}R  o'rtacha={total_r / n:+.2f}R  (OPEN: {open_n})")
    print("\nEslatma: namuna juda kichik -- bu raqamlar statistik xulosa emas, faqat yo'nalish.")


if __name__ == "__main__":
    main()
