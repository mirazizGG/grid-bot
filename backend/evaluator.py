"""Saqlangan signallarni MT5 tarixidagi keyingi narx harakati bilan solishtirib,
natijani (TP urildimi / SL urildimi) bazaga yozadi.

- Buy/Sell signallar: saqlangan entry/TP/SL bo'yicha baholanadi ("signal").
- Hold signallar: Gemini yo'nalishi (Buy/Sell bo'lsa) bo'yicha ATR 2:1 darajalar
  bilan "agar signal berilganida" nima bo'lardi -- gipotetik ("hold-gipotetik").
  Shu bilan Hold filtri foyda berayotganini tekshirish mumkin.

1-minutlik barlar, TP=+2R / SL=-1R. Bir barda ham TP, ham SL urilsa
konservativ ravishda SL hisoblanadi. Spred alohida hisobga olinmaydi.
"""
import datetime as dt
import json

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from db import SessionLocal, Analysis
from mt5_context import compute_trade_levels, TP_ATR_MULT, SL_ATR_MULT

DEFAULT_HORIZON_HOURS = 24.0
FINAL_OUTCOMES = ("TP", "SL", "EXPIRED", "SKIP")


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


def _plan(row: Analysis, data: dict):
    """(kind, direction, entry, tp, sl) yoki (None, sabab) qaytaradi."""
    ctx = data.get("market_context")
    if not ctx:
        return None, "MT5 konteksti yo'q"
    if row.final_signal in ("Buy", "Sell"):
        return ("signal", row.final_signal, row.entry_price, row.tp_price, row.sl_price), None
    direction = data.get("vision_ai", {}).get("signal")
    if direction not in ("Buy", "Sell"):
        return None, "Hold, Gemini ham Hold"
    lv = compute_trade_levels(ctx, direction)
    return ("hold-gipotetik", direction, lv["entry_price"], lv["tp_price"], lv["sl_price"]), None


def evaluate_pending(horizon_hours: float = DEFAULT_HORIZON_HOURS, utc_offset: int | None = None) -> dict:
    """Hali yakunlanmagan (outcome=None/OPEN/NO_DATA) tahlillarni baholab, bazaga yozadi.

    utc_offset: tahlilda server farqi saqlanmagan eski yozuvlar uchun zaxira qiymat (soat)."""
    if mt5 is None or not mt5.initialize():
        raise RuntimeError("MT5 terminaliga ulanib bo'lmadi (terminal ochiqmi?)")

    session = SessionLocal()
    now = dt.datetime.utcnow()
    counts: dict[str, int] = {}
    try:
        rows = session.query(Analysis).filter(
            (Analysis.outcome.is_(None)) | (Analysis.outcome.in_(("OPEN", "NO_DATA")))
        ).all()
        for row in rows:
            data = json.loads(row.raw_result)
            plan, reason = _plan(row, data)
            if plan is None:
                row.outcome, row.outcome_kind = "SKIP", None
            else:
                kind, direction, _entry, tp, sl = plan
                ctx = data["market_context"]
                offset = ctx.get("server_utc_offset_hours", utc_offset)
                if offset is None:
                    row.outcome, row.outcome_kind = "NO_DATA", kind  # server vaqti noma'lum -- keyinroq/--utc-offset bilan
                else:
                    start = row.timestamp + dt.timedelta(hours=offset)  # UTC -> broker server vaqti
                    end = start + dt.timedelta(hours=horizon_hours)
                    bars = mt5.copy_rates_range(ctx["symbol"], mt5.TIMEFRAME_M1, start, end)
                    if bars is None or len(bars) == 0:
                        row.outcome, row.outcome_kind = "NO_DATA", kind
                    else:
                        outcome = simulate(direction, tp, sl, bars["high"], bars["low"])
                        window_over = now >= row.timestamp + dt.timedelta(hours=horizon_hours)
                        if outcome == "OPEN" and window_over:
                            outcome = "EXPIRED"  # oyna tugadi, TP ham, SL ham urilmadi
                        row.outcome, row.outcome_kind = outcome, kind
            row.outcome_checked_at = now
            counts[row.outcome] = counts.get(row.outcome, 0) + 1
        session.commit()
    finally:
        session.close()
        mt5.shutdown()
    return {"checked": sum(counts.values()), "by_outcome": counts}


def summarize(session) -> dict:
    """Natijalar bo'yicha statistika: signal va hold-gipotetik uchun alohida."""
    rr = TP_ATR_MULT / SL_ATR_MULT
    out = {}
    for kind in ("signal", "hold-gipotetik"):
        rows = session.query(Analysis.outcome).filter(Analysis.outcome_kind == kind).all()
        outcomes = [r[0] for r in rows]
        tp, sl = outcomes.count("TP"), outcomes.count("SL")
        n = tp + sl
        total_r = tp * rr - sl
        out[kind] = {
            "n": n,
            "tp": tp,
            "sl": sl,
            "win_rate": round(tp / n, 3) if n else None,
            "total_r": round(total_r, 2),
            "avg_r": round(total_r / n, 3) if n else None,
            "open": outcomes.count("OPEN"),
            "expired": outcomes.count("EXPIRED"),
            "no_data": outcomes.count("NO_DATA"),
        }
    return out
