"""Saqlangan signallarni MT5 tarixidagi keyingi narx harakati bilan solishtirib,
natijani (TP urildimi / SL urildimi) bazaga yozadi.

- Buy/Sell signallar: saqlangan entry/TP/SL bo'yicha baholanadi ("signal").
- Hold signallar: Gemini yo'nalishi (Buy/Sell bo'lsa) bo'yicha ATR 2:1 darajalar
  bilan "agar signal berilganida" nima bo'lardi -- gipotetik ("hold-gipotetik").
  Shu bilan Hold filtri foyda berayotganini tekshirish mumkin.

Bid/ask ticklar bo‘yicha birinchi TP yoki SL aniqlanadi. Ma’lumotdagi
uzilishlar aniqlansa NO_DATA qaytariladi. Komissiya va slippage hisoblanmaydi.
"""
import datetime as dt
import json
import math
import bridge_client
import mt5_clock

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from db import SessionLocal, Analysis, patch_analysis_result
from mt5_context import compute_trade_levels, TP_ATR_MULT, SL_ATR_MULT, MT5_LOCK

DEFAULT_HORIZON_HOURS = 24.0
FINAL_OUTCOMES = ("TP", "SL", "EXPIRED", "SKIP")
MODEL_COMPARISON_VERSION = 4
OUTCOME_VERSION = 6
MAX_TICK_GAP_SECONDS = 60
TICK_CHUNK_MINUTES = 5
TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


def _tick_source(clock_profile=None):
    profile = mt5_clock.normalize_profile(clock_profile)
    return (bridge_client.MT5Proxy(clock_profile=profile) if bridge_client.enabled()
            else mt5_clock.NativeMT5Proxy(mt5, profile) if mt5 is not None else None)


def evaluation_parameters(clock_profile=None):
    return {"horizon_hours": DEFAULT_HORIZON_HOURS, "utc_offset": 0,
            "clock_profile": mt5_clock.normalize_profile(clock_profile)}


def publication_clock_profile(ctx, quote):
    """Freeze matching source clocks; an unannotated pair keeps legacy UTC mode."""
    if not isinstance(ctx, dict) or not isinstance(quote, dict):
        raise ValueError("MT5 konteksti yoki narxi mavjud emas")
    context_has = "clock_profile" in ctx
    quote_has = "clock_profile" in quote
    if context_has != quote_has:
        raise ValueError("MT5 konteksti va narxining vaqt siyosati mos emas")
    if not context_has:
        return mt5_clock.normalize_profile(None)
    if ctx["clock_profile"] is None or quote["clock_profile"] is None:
        raise ValueError("MT5 vaqt siyosati yaroqsiz")
    profile = mt5_clock.normalize_profile(ctx["clock_profile"])
    if profile != mt5_clock.normalize_profile(quote["clock_profile"]):
        raise ValueError("MT5 konteksti va narxining vaqt siyosati mos emas")
    return profile


def _stored_clock_profile(parameters):
    """Validate the recorded policy without consulting today's host settings."""
    if not isinstance(parameters, dict) or "clock_profile" not in parameters or parameters["clock_profile"] is None:
        raise ValueError("Signalning asl vaqt siyosati saqlanmagan; tarix tekshirilmaydi")
    horizon, offset = parameters.get("horizon_hours"), parameters.get("utc_offset")
    if (type(horizon) not in (int, float) or not math.isfinite(horizon) or horizon != DEFAULT_HORIZON_HOURS or
            type(offset) is not int or offset != 0):
        raise ValueError("Saqlangan signalning baholash shartlari 24 soat va UTC+0 bo'lishi kerak")
    return mt5_clock.normalize_profile(parameters["clock_profile"])


def create_model_comparison(ctx, quote, vision, qwen, timestamp=None):
    """Freeze a hypothetical 2:1 ATR plan when each model's result is available."""
    timestamp = timestamp or dt.datetime.utcnow()
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(dt.timezone.utc).replace(tzinfo=None)
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
            try:
                profile = publication_clock_profile(ctx, quote)
            except ValueError as exc:
                plan["reason"] = str(exc)
                plans[name] = plan
                continue
            side = "ask" if signal == "Buy" else "bid"
            if abs(quote[side] - ctx[side]) > 0.25 * ctx["atr_14"]:
                plan["reason"] = "Tahlil davomida narx o'zgargan"
            else:
                plan.update(compute_trade_levels({**ctx, **quote}, signal))
                plan.update(outcome=None, **evaluation_parameters(profile), timestamp=timestamp.isoformat() + "Z")
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


def _tick_path_outcome(symbol, direction, tp, sl, start, end, clock_profile=None):
    """Replay quotes in bounded chunks; missing/invalid quote intervals are unknown.

    Gap detection cannot prove that a broker delivered every tick; it rejects
    visible holes and avoids treating sparse/missing history as a successful trade.
    """
    if direction not in ("Buy", "Sell") or end <= start:
        return "NO_DATA"
    try:
        frozen_profile = mt5_clock.normalize_profile(clock_profile)
    except mt5_clock.ClockProfileError:
        return "NO_DATA"
    utc = lambda value: value.astimezone(dt.timezone.utc) if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    to_ms = lambda value: utc(value).timestamp() * 1000
    gap_ms = MAX_TICK_GAP_SECONDS * 1000
    previous_ms = to_ms(start)
    cursor = start
    seen = False
    pending_ms, pending_hit = None, None
    while cursor < end:
        chunk_end = min(cursor + dt.timedelta(minutes=TICK_CHUNK_MINUTES), end)
        try:
            source = _tick_source(frozen_profile)
            if source is None:
                return "NO_DATA"
            ticks = source.copy_ticks_range(symbol, utc(cursor), utc(chunk_end), source.COPY_TICKS_INFO)
        except (bridge_client.BridgeError, mt5_clock.ClockProfileError) as exc:
            bridge_client.record_error(str(exc))
            return "NO_DATA"
        if ticks is None:
            return "NO_DATA"
        for tick in ticks:
            tick_ms = int(tick["time_msc"])
            # copy_ticks_range includes its upper bound: each tick belongs to
            # one chunk only, and a quote at the horizon is not post-publication.
            if not to_ms(cursor) <= tick_ms < to_ms(chunk_end):
                continue
            if tick_ms < previous_ms:
                return "NO_DATA"
            if pending_ms is not None and tick_ms > pending_ms:
                return pending_hit
            if tick_ms - previous_ms > gap_ms:
                return "NO_DATA"
            bid, ask = float(tick["bid"]), float(tick["ask"])
            if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask < bid:
                return "NO_DATA"
            price = bid if direction == "Buy" else ask
            hit_sl = (direction == "Buy" and price <= sl) or (direction == "Sell" and price >= sl)
            hit_tp = (direction == "Buy" and price >= tp) or (direction == "Sell" and price <= tp)
            if hit_sl or hit_tp:
                # When ticks share a millisecond, their order is not reliable.
                pending_ms = tick_ms
                pending_hit = "SL" if hit_sl or pending_hit == "SL" else "TP"
            previous_ms = tick_ms
            seen = True
        if pending_hit is not None:
            return pending_hit
        if to_ms(chunk_end) - previous_ms > gap_ms:
            return "NO_DATA"
        cursor = chunk_end
    return "OPEN" if seen else "NO_DATA"


def _partial_minute_outcome(symbol, direction, tp, sl, start, end):
    """Compatibility helper for a partial interval, using the same tick rules."""
    if end <= start:
        return "OPEN"
    return _tick_path_outcome(symbol, direction, tp, sl, start, end)


def _plan(row: Analysis, data: dict):
    """(kind, direction, entry, tp, sl) yoki (None, sabab) qaytaradi."""
    ctx = data.get("market_context")
    if not ctx:
        return None, "MT5 konteksti yo'q"
    age = data.get("image_age_minutes")
    timeframe = ctx.get("timeframe", "M15")
    if (data.get("image_freshness") == "stale_file_timestamp" or
            (isinstance(age, (float, int)) and math.isfinite(age) and
             age > max(10, TIMEFRAME_MINUTES.get(timeframe, 15)))):
        return None, "Rasm eski bo'lishi mumkin"
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


def _evaluate_price_path(symbol, direction, tp, sl, timestamp, now, horizon_hours, utc_offset, clock_profile=None):
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (tp, sl)):
        return "NO_DATA"
    if direction not in ("Buy", "Sell") or (direction == "Buy" and tp <= sl) or (direction == "Sell" and tp >= sl):
        return "NO_DATA"
    if utc_offset not in (None, 0) or type(horizon_hours) not in (int, float) or not math.isfinite(horizon_hours) or horizon_hours <= 0:
        return "NO_DATA"
    signal_time = timestamp.astimezone(dt.timezone.utc).replace(tzinfo=None) if timestamp.tzinfo else timestamp
    now = now.astimezone(dt.timezone.utc).replace(tzinfo=None) if now.tzinfo else now
    end = min(signal_time + dt.timedelta(hours=horizon_hours), now)
    if end <= signal_time:
        return "OPEN"
    result = _tick_path_outcome(symbol, direction, tp, sl, signal_time, end, clock_profile)
    if result == "OPEN" and now >= signal_time + dt.timedelta(hours=horizon_hours):
        return "EXPIRED"
    return result


def _evaluate_pending(horizon_hours: float, utc_offset: int | None) -> dict:
    """Hali yakunlanmagan (outcome=None/OPEN/NO_DATA) tahlillarni baholab, bazaga yozadi.

    Saqlangan rejalar 24 soat, UTC+0 va o'z vaqtida saqlangan MT5 clock_profile
    bo'yicha baholanadi. Bugungi sozlama eski rejaga qo'llanmaydi; vaqt siyosati
    yo'q eski yozuv NO_DATA va legacy holatida qoladi."""
    if isinstance(horizon_hours, bool) or not math.isfinite(horizon_hours) or horizon_hours <= 0:
        raise ValueError("Kuzatuv muddati musbat son bo'lishi kerak")
    # Published plans have one fixed policy. Request parameters must never
    # silently reinterpret or finalize the same signal under another policy.
    if horizon_hours != DEFAULT_HORIZON_HOURS or utc_offset not in (None, 0):
        raise ValueError("Saqlangan rejalar 24 soat va UTC+0 bo'yicha baholanadi; so'rov bu shartlarni o'zgartira olmaydi")
    source = _tick_source()
    try:
        connected = source is not None and source.initialize()
    except (bridge_client.BridgeError, mt5_clock.ClockProfileError) as exc:
        bridge_client.record_error(str(exc))
        raise RuntimeError("Windows host bridge yoki MT5 bilan aloqa yo'q") from exc
    if not connected:
        raise RuntimeError("MT5 terminaliga ulanib bo'lmadi (terminal ochiqmi?)")

    session = SessionLocal()
    now = dt.datetime.utcnow()
    counts: dict[str, int] = {}
    model_checked = 0
    pending_patches = []
    try:
        rows = session.query(Analysis).all()
        for row in rows:
            data = json.loads(row.raw_result)
            patches = {}
            if row.outcome not in FINAL_OUTCOMES:
                current_policy = True
                plan, _ = _plan(row, data)
                if plan is None:
                    row.outcome, row.outcome_kind = "SKIP", None
                else:
                    kind, direction, _entry, tp, sl = plan
                    parameters = data.get("evaluation_parameters")
                    try:
                        profile = _stored_clock_profile(parameters)
                    except ValueError as exc:
                        row.outcome = "NO_DATA"
                        current_policy = False
                        patches["$.evaluation_error"] = str(exc)
                        patches["$.evaluation_clock_status"] = "legacy_unknown"
                    else:
                        if profile["history_mode"] == "unverified":
                            row.outcome = "NO_DATA"
                            patches["$.evaluation_error"] = "MT5 tick tarixining vaqt domeni hali tasdiqlanmagan"
                        else:
                            row.outcome = _evaluate_price_path(data["market_context"]["symbol"], direction, tp, sl,
                                row.timestamp, now, parameters["horizon_hours"], parameters["utc_offset"], profile)
                            if "evaluation_error" in data:
                                patches["$.evaluation_error"] = None
                    row.outcome_kind = kind
                row.outcome_checked_at = now
                if current_policy:
                    row.outcome_version = OUTCOME_VERSION
                elif row.outcome_version == OUTCOME_VERSION:
                    row.outcome_version = None
                counts[row.outcome] = counts.get(row.outcome, 0) + 1
            for name, model_plan in data.get("model_comparison", {}).items():
                if name not in ("gemini", "qwen") or not isinstance(model_plan, dict) or model_plan.get("outcome") in FINAL_OUTCOMES:
                    continue
                valid_clock = False
                try:
                    if model_plan.get("version") != MODEL_COMPARISON_VERSION:
                        raise ValueError("Eski model rejasining vaqt siyosati tasdiqlanmagan")
                    profile = _stored_clock_profile(model_plan)
                    valid_clock = True
                    if profile["history_mode"] == "unverified":
                        raise ValueError("MT5 tick tarixining vaqt domeni hali tasdiqlanmagan")
                    published = dt.datetime.fromisoformat(model_plan["timestamp"].replace("Z", "+00:00"))
                    if published.tzinfo is None:
                        raise ValueError("Model rejasining UTC vaqti tasdiqlanmagan")
                    published = published.astimezone(dt.timezone.utc).replace(tzinfo=None)
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    outcome = "NO_DATA"
                    patches[f"$.model_comparison.{name}.reason"] = str(exc)
                    if not valid_clock and model_plan.get("version") == MODEL_COMPARISON_VERSION:
                        patches[f"$.model_comparison.{name}.version"] = None
                else:
                    outcome = _evaluate_price_path(data["market_context"]["symbol"], model_plan["signal"],
                        model_plan.get("tp_price"), model_plan.get("sl_price"), published, now,
                        model_plan["horizon_hours"], model_plan["utc_offset"], profile)
                patches[f"$.model_comparison.{name}.outcome"] = outcome
                patches[f"$.model_comparison.{name}.checked_at"] = now.isoformat() + "Z"
                model_checked += 1
            if patches:
                pending_patches.append((row.id, patches))
        # No writes or additional ORM queries occur during tick replay above.
        # Applying a JSON patch also autoflushes dirty outcome columns, so defer
        # every patch until all slow MT5 work is finished. This keeps SQLite's
        # writer lock out of replay while retaining atomic path-level updates.
        for record_id, patches in pending_patches:
            patch_analysis_result(session, record_id, patches)
        session.commit()
    finally:
        session.close()
        source.shutdown()
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
        rows = session.query(Analysis.outcome).filter(Analysis.outcome_kind == kind, Analysis.outcome_version == OUTCOME_VERSION).all()
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
                       (Analysis.outcome_version.is_(None)) | (Analysis.outcome_version != OUTCOME_VERSION)).count(),
        }
    return out
