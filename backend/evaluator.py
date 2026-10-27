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
import math

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from db import SessionLocal, Analysis
from mt5_context import compute_trade_levels, TP_ATR_MULT, SL_ATR_MULT, MT5_LOCK

DEFAULT_HORIZON_HOURS = 24.0
FINAL_OUTCOMES = ("TP", "SL", "EXPIRED", "SKIP")
MODEL_COMPARISON_VERSION = 1


def create_model_comparison(ctx, quote, vision, qwen):
    """Freeze identical hypothetical 2:1 ATR plans at publication, not model price guesses."""
    plans = {}
    for name, model in (("gemini", vision), ("qwen", qwen)):
        signal = model.get("signal")
        plan = {"signal": signal, "outcome": "SKIP", "version": MODEL_COMPARISON_VERSION}
        if model.get("error") or signal not in ("Buy", "Sell"):
            plan["reason"] = "Model yo'nalish bermadi yoki ishlamadi"
        elif name == "gemini" and model.get("entry_type") != "market":
            plan["reason"] = "Gemini kirishni kutishni tavsiya qildi yoki turini ko'rsatmadi"
        elif not ctx or not quote or not isinstance(ctx.get("atr_14"), (int, float)) or not math.isfinite(ctx["atr_14"]) or ctx["atr_14"] <= 0:
            plan["reason"] = "Yaroqli MT5 narxi yo'q"
        else:
            side = "ask" if signal == "Buy" else "bid"
            if abs(quote[side] - ctx[side]) > 0.25 * ctx["atr_14"]:
                plan["reason"] = "Tahlil davomida narx o'zgargan"
            else:
                plan.update(compute_trade_levels({**ctx, **quote}, signal))
                plan.update(outcome=None, horizon_hours=DEFAULT_HORIZON_HOURS, utc_offset=0)
        plans[name] = plan
    return plans


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


def _partial_minute_outcome(symbol, direction, tp, sl, start, end):
    """Use real bid/ask ticks after publication; never infer from pre-signal OHLC."""
    if end <= start:
        return "OPEN"
    utc = lambda value: value.replace(tzinfo=dt.timezone.utc)
    ticks = mt5.copy_ticks_range(symbol, utc(start), utc(end), mt5.COPY_TICKS_INFO)
    if ticks is None or len(ticks) == 0:
        return "NO_DATA"
    start_ms, end_ms = utc(start).timestamp() * 1000, utc(end).timestamp() * 1000
    seen = False
    for tick in ticks:
        tick_ms = int(tick["time_msc"])
        if not start_ms <= tick_ms < end_ms:
            continue
        price = float(tick["bid"] if direction == "Buy" else tick["ask"])
        if not math.isfinite(price) or price <= 0:
            continue
        seen = True
        if (direction == "Buy" and price <= sl) or (direction == "Sell" and price >= sl):
            return "SL"
        if (direction == "Buy" and price >= tp) or (direction == "Sell" and price <= tp):
            return "TP"
    return "OPEN" if seen else "NO_DATA"


def _plan(row: Analysis, data: dict):
    """(kind, direction, entry, tp, sl) yoki (None, sabab) qaytaradi."""
    ctx = data.get("market_context")
    if not ctx:
        return None, "MT5 konteksti yo'q"
    plan_type = data.get("evaluation_plan")
    if row.final_signal in ("Buy", "Sell"):
        if plan_type != "market":
            return None, "Kirish sharti yoki narx vaqti tasdiqlanmagan"
        return ("signal", row.final_signal, row.entry_price, row.tp_price, row.sl_price), None
    if plan_type != "hold_market" or not data.get("publication_quote"):
        return None, "Gipotetik kirish uchun joriy narx tasdiqlanmagan"
    direction = data.get("vision_ai", {}).get("signal")
    if direction not in ("Buy", "Sell"):
        return None, "Hold, Gemini ham Hold"
    lv = compute_trade_levels({**ctx, **data["publication_quote"]}, direction)
    return ("hold-gipotetik", direction, lv["entry_price"], lv["tp_price"], lv["sl_price"]), None


def evaluate_pending(horizon_hours: float = DEFAULT_HORIZON_HOURS, utc_offset: int | None = None) -> dict:
    with MT5_LOCK:
        return _evaluate_pending(horizon_hours, utc_offset)


def _evaluate_price_path(symbol, direction, tp, sl, timestamp, now, horizon_hours, utc_offset):
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (tp, sl)):
        return "NO_DATA"
    signal_time = timestamp + dt.timedelta(hours=utc_offset or 0)
    start = signal_time.replace(second=0, microsecond=0)
    if start < signal_time:
        start += dt.timedelta(minutes=1)
    end = min(signal_time + dt.timedelta(hours=horizon_hours), now + dt.timedelta(hours=utc_offset or 0))
    first = _partial_minute_outcome(symbol, direction, tp, sl, signal_time, min(start, end))
    if first != "OPEN":
        return first
    window_over = now >= timestamp + dt.timedelta(hours=horizon_hours)
    if end <= start:
        return "EXPIRED" if window_over else "OPEN"
    epoch = lambda value: int(value.replace(tzinfo=dt.timezone.utc).timestamp())
    bars = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, epoch(start), epoch(end))
    if bars is None or len(bars) == 0:
        return "NO_DATA"
    complete_end = epoch(end.replace(second=0, microsecond=0))
    usable = bars[bars["time"] + 60 <= complete_end]
    expected = epoch(start)
    prefix = []
    for bar in usable:
        if int(bar["time"]) != expected:
            break
        prefix.append(bar)
        expected += 60
    outcome = simulate(direction, tp, sl, (b["high"] for b in prefix), (b["low"] for b in prefix))
    if outcome == "OPEN" and expected < complete_end:
        return "NO_DATA"
    if outcome == "OPEN" and window_over:
        tail_start = end.replace(second=0, microsecond=0)
        if end > tail_start:
            tail = _partial_minute_outcome(symbol, direction, tp, sl, tail_start, end)
            if tail != "OPEN":
                return tail
        return "EXPIRED"
    return outcome


def _evaluate_pending(horizon_hours: float, utc_offset: int | None) -> dict:
    """Hali yakunlanmagan (outcome=None/OPEN/NO_DATA) tahlillarni baholab, bazaga yozadi.

    MT5 Python API bar vaqtlarini UTC qaytaradi. utc_offset faqat alohida
    tekshirilgan, noodatiy tarix uchun qo'lda tuzatish sifatida ishlatiladi."""
    if not math.isfinite(horizon_hours) or horizon_hours <= 0:
        raise ValueError("Kuzatuv muddati musbat son bo'lishi kerak")
    if mt5 is None or not mt5.initialize():
        raise RuntimeError("MT5 terminaliga ulanib bo'lmadi (terminal ochiqmi?)")

    session = SessionLocal()
    now = dt.datetime.utcnow()
    counts: dict[str, int] = {}
    model_checked = 0
    try:
        rows = session.query(Analysis).all()
        for row in rows:
            data = json.loads(row.raw_result)
            if row.outcome not in FINAL_OUTCOMES:
                plan, _ = _plan(row, data)
                if plan is None:
                    row.outcome, row.outcome_kind = "SKIP", None
                else:
                    kind, direction, _entry, tp, sl = plan
                    row.outcome = _evaluate_price_path(data["market_context"]["symbol"], direction, tp, sl,
                                                       row.timestamp, now, horizon_hours, utc_offset)
                    row.outcome_kind = kind
                row.outcome_checked_at = now
                row.outcome_version = 3
                counts[row.outcome] = counts.get(row.outcome, 0) + 1
            changed = False
            for model_plan in data.get("model_comparison", {}).values():
                if model_plan.get("version") != MODEL_COMPARISON_VERSION or model_plan.get("outcome") in FINAL_OUTCOMES:
                    continue
                model_plan["outcome"] = _evaluate_price_path(data["market_context"]["symbol"], model_plan["signal"],
                    model_plan.get("tp_price"), model_plan.get("sl_price"), row.timestamp, now,
                    model_plan["horizon_hours"], model_plan["utc_offset"])
                model_plan["checked_at"] = now.isoformat() + "Z"
                changed = True
                model_checked += 1
            if changed:
                row.raw_result = json.dumps(data, ensure_ascii=False)
        session.commit()
    finally:
        session.close()
        mt5.shutdown()
    return {"checked": sum(counts.values()), "by_outcome": counts, "model_checked": model_checked}


def summarize_models(rows):
    out = {}
    for name in ("gemini", "qwen"):
        outcomes = []
        for row in rows:
            plan = json.loads(row.raw_result).get("model_comparison", {}).get(name)
            if plan and plan.get("version") == MODEL_COMPARISON_VERSION:
                outcomes.append(plan.get("outcome"))
        tp, sl = outcomes.count("TP"), outcomes.count("SL")
        n = tp + sl
        out[name] = {"n": n, "tp": tp, "sl": sl, "win_rate": round(tp / n, 3) if n else None,
                     "open": outcomes.count("OPEN"), "expired": outcomes.count("EXPIRED"),
                     "no_data": outcomes.count("NO_DATA"), "pending": outcomes.count(None), "skip": outcomes.count("SKIP")}
    return out


def summarize(session) -> dict:
    """Natijalar bo'yicha statistika: signal va hold-gipotetik uchun alohida."""
    rr = TP_ATR_MULT / SL_ATR_MULT
    out = {}
    for kind in ("signal", "hold-gipotetik"):
        rows = session.query(Analysis.outcome).filter(Analysis.outcome_kind == kind, Analysis.outcome_version == 3).all()
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
            "legacy": session.query(Analysis).filter(Analysis.outcome_kind == kind,
                       (Analysis.outcome_version.is_(None)) | (Analysis.outcome_version != 3)).count(),
        }
    return out
