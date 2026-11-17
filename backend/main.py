"""FastAPI backend: chart rasmini custom model + Gemini vision bilan tahlil qilib,
yakuniy Buy/Sell/Hold + TP/SL signalini beradi va tarixni saqlaydi."""
import os
import asyncio
import time
import uuid
import math
import ipaddress
import re
import logging
import warnings as python_warnings
import datetime as dt
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Query, Response
from sqlalchemy import func
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from PIL import Image
import io
import json
from urllib.request import urlopen

import custom_model
import config
import bridge_client
import evaluator
import gemini_vision
import mt5_context
import ollama_vision
from consensus import compute_consensus
from db import SessionLocal, Analysis, save_analysis, claim_qwen_analysis, patch_analysis_result
from config import GEMINI_API_KEY, GEMINI_MODEL, OLLAMA_URL, QWEN_MODEL, QWEN_TIMEOUT

UPLOAD_DIR = str(config.UPLOAD_DIR)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Grid-Bot Chart Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_IMAGE_BYTES + 128 * 1024  # multipart headers and form fields
MAX_IMAGE_PIXELS = 25_000_000
_analysis_slots = asyncio.Semaphore(2)
_qwen_slots = asyncio.Semaphore(2)


@app.middleware("http")
async def local_request_guard(request: Request, call_next):
    # Host validation prevents an external hostname resolving to loopback from
    # bypassing same-origin protection (DNS rebinding).
    try:
        from network_guard import trusted_client
        is_local = bool(request.client and trusted_client(request.client.host))
    except ValueError:
        is_local = False
    if not is_local or request.url.hostname not in ("localhost", "127.0.0.1", "::1"):
        return JSONResponse({"detail": "Faqat lokal manzilga ruxsat beriladi"}, status_code=403)
    origin = request.headers.get("origin")
    expected_origin = f"{request.url.scheme}://{request.url.netloc}"
    if (origin is not None and origin != expected_origin) or request.headers.get("sec-fetch-site") == "cross-site":
        return JSONResponse({"detail": "Boshqa saytdan so'rovga ruxsat berilmagan"}, status_code=403)
    if request.method == "POST" and request.headers.get("x-grid-bot-request") != "1":
        return JSONResponse({"detail": "Lokal so'rov sarlavhasi talab qilinadi"}, status_code=403)
    if request.url.path == "/analyze" and request.method == "POST":
        length = request.headers.get("content-length")
        if length is None:
            return JSONResponse({"detail": "So'rov hajmi ko'rsatilmagan"}, status_code=411)
        try:
            size = int(length)
        except ValueError:
            return JSONResponse({"detail": "So'rov hajmi noto'g'ri"}, status_code=400)
        if size < 0 or size > MAX_REQUEST_BYTES:
            return JSONResponse({"detail": "Rasm hajmi juda katta"}, status_code=413)
        if _analysis_slots.locked():
            return JSONResponse({"detail": "Tahlil band, birozdan keyin qayta urinib ko'ring"}, status_code=429)
        async with _analysis_slots:
            return await call_next(request)
    if request.method == "POST" and re.fullmatch(r"/history/\d+/qwen", request.url.path):
        if _qwen_slots.locked():
            return JSONResponse({"detail": "Qwen band, birozdan keyin qayta urinib ko'ring"}, status_code=429)
        async with _qwen_slots:
            return await call_next(request)
    return await call_next(request)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
MIN_STALE_IMAGE_MINUTES = 10  # rasm shundan (va timeframe bir bar vaqtidan) eski bo'lsa ogohlantiramiz


def _image_age_minutes(file_modified_ms: float | None) -> float | None:
    if not file_modified_ms or not math.isfinite(file_modified_ms) or file_modified_ms > time.time() * 1000 + 5000:
        return None  # vaqt noma'lum (masalan clipboard'dan qo'yilgan rasm)
    age = (time.time() * 1000 - file_modified_ms) / 60000
    return max(age, 0.0)


def _fmt_age(minutes: float) -> str:
    if minutes < 90:
        return f"{minutes:.0f} daqiqa"
    if minutes < 48 * 60:
        return f"{minutes / 60:.1f} soat"
    return f"{minutes / 1440:.0f} kun"


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    symbol: str | None = Form(None),
    timeframe: str | None = Form(None),
    file_modified_ms: float | None = Form(None),  # brauzerdagi File.lastModified (ms, UTC epoch)
):
    settings = config.get_settings()
    if not settings["analysis_enabled"]:
        raise HTTPException(503, "Yangi tahlillar boshqaruv panelida vaqtincha to'xtatilgan")
    if not settings["gemini_enabled"]:
        raise HTTPException(503, "Gemini boshqaruv panelida o'chirilgan")
    symbol = symbol or settings["default_symbol"]
    timeframe = timeframe or settings["default_timeframe"]
    symbol = symbol.strip().upper()
    timeframe = timeframe.strip().upper()
    if not re.fullmatch(r"[A-Z0-9._#-]{1,32}", symbol) or timeframe not in TIMEFRAME_MINUTES:
        raise HTTPException(400, "Symbol yoki timeframe noto'g'ri")
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
        raise HTTPException(400, f"Qo'llab-quvvatlanmaydigan fayl turi: {file.content_type}")

    image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Rasm hajmi 10 MB dan oshmasligi kerak")

    try:
        with python_warnings.catch_warnings():
            python_warnings.simplefilter("error", Image.DecompressionBombWarning)
            pil_image = Image.open(io.BytesIO(image_bytes))
            if pil_image.width * pil_image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Rasm o'lchami juda katta")
            pil_image.verify()
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image.load()
    except Exception:
        raise HTTPException(400, "Rasm faylini ochib bo'lmadi")

    filename = f"{uuid.uuid4().hex}.png"
    saved_path = os.path.join(UPLOAD_DIR, filename)

    market_ctx = await run_in_threadpool(mt5_context.get_context, symbol, timeframe)

    custom_result = await run_in_threadpool(custom_model.analyze, market_ctx)
    started = time.monotonic()
    try:
        vision_result = await run_in_threadpool(gemini_vision.analyze_image, image_bytes, file.content_type,
                                                market_context=market_ctx, settings=settings)
    except Exception as exc:
        logging.getLogger(__name__).warning("Gemini tahlili amalga oshmadi (%s)", type(exc).__name__)
        raise HTTPException(502, "Gemini tahlili amalga oshmadi. Xizmat va API sozlamalarini tekshiring.")
    # Optional local inference is requested separately after the primary signal
    # is published. It cannot occupy an analysis slot or age the primary quote.
    qwen_result = {"source": "qwen", "model": settings["qwen_model"], "status": "not_requested"}
    if not settings["qwen_enabled"]:
        qwen_result.update(status="disabled", error="Qwen boshqaruv panelida o'chirilgan")

    # Complete potentially expensive image encoding before freezing live prices.
    await run_in_threadpool(lambda: pil_image.convert("RGB").save(saved_path, format="PNG"))

    consensus = compute_consensus(custom_result, vision_result, settings)
    consensus["policy"] = {key: settings[key] for key in ("min_custom_confidence", "min_vision_override_confidence",
                                                         "custom_model_weight", "vision_ai_weight")}
    consensus["qwen_vision"] = qwen_result

    publication_quote = None
    if market_ctx:
        publication_quote = await run_in_threadpool(mt5_context.get_quote, market_ctx["symbol"])
    published_at = dt.datetime.utcnow()
    if market_ctx is None or (consensus["final_signal"] != "Hold" and publication_quote is None):
        consensus.update(final_signal="Hold", final_confidence=None, agreement=False,
                         entry_price=None, tp_price=None, sl_price=None,
                         entry_type=None, entry_note=None,
                         note="MT5 narxi mavjud emas yoki eskirgan: qayta tahlil qiling. Yakuniy qaror Hold.")

    if publication_quote and consensus["final_signal"] != "Hold":
        side = "ask" if consensus["final_signal"] == "Buy" else "bid"
        movement = abs(publication_quote[side] - market_ctx[side])
        atr = market_ctx.get("atr_14", 0)
        if not atr or movement > 0.25 * atr:
            consensus.update(final_signal="Hold", final_confidence=None, agreement=False,
                             entry_price=None, tp_price=None, sl_price=None, entry_type=None, entry_note=None,
                             note="Tahlil davomida narx sezilarli o'zgardi. Yangi chart bilan qayta tahlil qiling.")

    if consensus["final_signal"] != "Hold" and consensus.get("entry_type") != "market":
        consensus.update(final_signal="Hold", final_confidence=None, agreement=False,
                         entry_price=None, tp_price=None, sl_price=None, entry_type=None, entry_note=None,
                         note="Gemini kutish shartini ko'rsatdi yoki kirish turi noma'lum. Shart bajarilgach qayta tahlil qiling.")

    # MT5'dan jonli ma'lumot olingan bo'lsa, TP/SL/Entry'ni Gemini taxminiga
    # emas, ATR asosidagi aniq hisob-kitobga almashtiramiz -- hallucinatsiya
    # xavfini yo'qotadi.
    if market_ctx and consensus["final_signal"] != "Hold":
        levels = mt5_context.compute_trade_levels({**market_ctx, **publication_quote}, consensus["final_signal"])
        consensus.update(levels)
        consensus["price_source"] = "mt5_live"
        consensus["evaluation_plan"] = "market"
    else:
        consensus["price_source"] = "vision_estimate"
    consensus["market_context"] = market_ctx
    consensus["publication_quote"] = publication_quote
    if (consensus["final_signal"] == "Hold" and publication_quote and market_ctx and
            vision_result.get("entry_type") == "market" and vision_result.get("signal") in ("Buy", "Sell")):
        side = "ask" if vision_result["signal"] == "Buy" else "bid"
        atr = market_ctx.get("atr_14", 0)
        if atr and abs(publication_quote[side] - market_ctx[side]) <= 0.25 * atr:
            consensus["evaluation_plan"] = "hold_market"

    warnings = []
    model_warning = custom_model.provenance_status()["warning"]
    if model_warning:
        warnings.append(model_warning)
    stale_image = False
    if time.monotonic() - started > 60:
        warnings.append("Tahlil 1 daqiqadan ko'p davom etdi. MT5 narxi tahlil boshida olingan; savdodan oldin yangisini tekshiring.")
    if market_ctx is None:
        warnings.append("MT5 terminalini va tanlangan symbolni tekshiring. Gemini xulosasi faqat ma'lumot uchun ko'rsatilgan.")
    age_min = _image_age_minutes(file_modified_ms)
    consensus["image_freshness"] = "unknown" if age_min is None else "recent_file_timestamp"
    if age_min is not None:
        consensus["image_age_minutes"] = round(age_min, 1)
        tf_min = TIMEFRAME_MINUTES.get(timeframe.strip().upper(), 15)
        if age_min > max(MIN_STALE_IMAGE_MINUTES, tf_min):
            stale_image = True
            consensus["image_freshness"] = "stale_file_timestamp"
            warnings.append(
                f"Rasm {_fmt_age(age_min)} oldin olingan, narx/indikatorlar esa hozirgi. Natija "
                "ishonchsiz bo'lishi mumkin -- MT5'da yangi skrinshot oling. Yakuniy qaror Hold."
            )
    else:
        warnings.append("Rasm olingan vaqt noma'lum. Screenshot ayni vaqtdagi chartga tegishli ekanini tekshiring.")
    if age_min is not None:
        warnings.append("Fayl vaqti screenshot olingan vaqtni tasdiqlamaydi; chartning joriyligini tekshiring.")
    if stale_image:
        consensus.update(final_signal="Hold", final_confidence=None, agreement=False,
                         entry_price=None, tp_price=None, sl_price=None,
                         entry_type=None, entry_note=None, price_source=None,
                         note="Rasm fayli eski: yangi screenshot bilan qayta tahlil qiling. Yakuniy qaror Hold.")
        consensus.pop("evaluation_plan", None)
    consensus["warnings"] = warnings
    consensus["published_at"] = published_at.isoformat() + "Z"
    if consensus.get("evaluation_plan"):
        consensus["evaluation_parameters"] = evaluator.evaluation_parameters()
    consensus["model_comparison"] = evaluator.create_model_comparison(market_ctx, publication_quote, vision_result, qwen_result, published_at)
    if stale_image:
        for plan in consensus["model_comparison"].values():
            plan.update(outcome="SKIP", reason="Rasm eski bo'lishi mumkin")

    try:
        record = await run_in_threadpool(save_analysis, f"/uploads/{filename}", consensus)
    except Exception:
        os.unlink(saved_path)
        raise

    return record.to_dict()


@app.post("/history/{record_id}/qwen")
async def analyze_qwen(record_id: int):
    settings = config.get_settings()
    if not settings["analysis_enabled"] or not settings["qwen_enabled"]:
        raise HTTPException(409, "Qwen tahlili boshqaruv panelida to'xtatilgan")
    record, claimed = await run_in_threadpool(claim_qwen_analysis, record_id, settings["qwen_model"], max(30, settings["qwen_timeout"] + 30))
    if record is None:
        raise HTTPException(404, "Tahlil topilmadi")
    if not claimed:
        status = record["detail"].get("qwen_vision", {}).get("status")
        return JSONResponse(record, status_code=202 if status == "running" else 200)

    data = record["detail"]
    request_id = data["qwen_vision"]["request_id"]
    ctx = data.get("market_context")
    quote = None
    try:
        # Only load a generated upload, never an arbitrary path from a backup.
        filename = str(record["image_path"]).removeprefix("/uploads/")
        if not re.fullmatch(r"[0-9a-f]{32}\.png", filename):
            raise ValueError("Saqlangan rasm yo'li noto'g'ri")
        image_bytes = await run_in_threadpool(Path(UPLOAD_DIR, filename).read_bytes)
        result = await run_in_threadpool(ollama_vision.analyze_image, image_bytes, ctx,
                                        model=settings["qwen_model"], timeout=settings["qwen_timeout"])
        if ctx:
            quote = await run_in_threadpool(mt5_context.get_quote, ctx["symbol"])
        result.update(status="completed", completed_at=dt.datetime.utcnow().isoformat() + "Z")
    except Exception as exc:
        logging.getLogger(__name__).warning("Qwen tahlili amalga oshmadi (%s)", type(exc).__name__)
        result = {"source": "qwen", "model": settings["qwen_model"], "status": "error",
                  "error": "Qwen tahlili amalga oshmadi. Ollama va saqlangan rasmni tekshiring."}
    published_at = dt.datetime.utcnow()
    plan = evaluator.create_model_comparison(ctx, quote, {"signal": "Hold"}, result, published_at)["qwen"]
    primary_time = dt.datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
    elapsed_minutes = max(0, (published_at.replace(tzinfo=dt.timezone.utc) - primary_time).total_seconds() / 60)
    max_age_minutes = max(MIN_STALE_IMAGE_MINUTES, TIMEFRAME_MINUTES.get((ctx or {}).get("timeframe"), 15))
    image_age = data.get("image_age_minutes", 0)
    context_time = (ctx or {}).get("quote_checked_at", primary_time.timestamp())
    context_age = published_at.replace(tzinfo=dt.timezone.utc).timestamp() - context_time
    if (data.get("image_freshness") == "stale_file_timestamp" or
            image_age + elapsed_minutes > max_age_minutes or context_age > max_age_minutes * 60):
        plan.update(outcome="SKIP", reason="Rasm yoki saqlangan MT5 konteksti eskirgan; qayta tahlil qiling")
    plan["publication_quote"] = quote

    def save_result():
        session = SessionLocal()
        try:
            patch_analysis_result(session, record_id, {"$.qwen_vision": result, "$.model_comparison.qwen": plan},
                                  qwen_request_id=request_id)
            session.commit()
            session.expire_all()
            return session.get(Analysis, record_id).to_dict()
        finally:
            session.close()

    return await run_in_threadpool(save_result)


@app.get("/history")
def history(response: Response, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
            symbol: str | None = None, signal: str | None = Query(None, pattern="^(Buy|Sell|Hold)$")):
    session = SessionLocal()
    try:
        query = session.query(Analysis)
        if symbol:
            symbol = symbol.strip().upper()
            if not re.fullmatch(r"[A-Z0-9._#-]{1,32}", symbol):
                raise HTTPException(400, "Symbol noto'g'ri")
            query = query.filter(func.json_extract(Analysis.raw_result, "$.market_context.symbol") == symbol)
        if signal:
            query = query.filter(Analysis.final_signal == signal)
        response.headers["X-Total-Count"] = str(query.count())
        rows = (
            query
            .order_by(Analysis.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in rows]
    finally:
        session.close()


@app.get("/stats")
def stats():
    session = SessionLocal()
    try:
        rows = session.query(Analysis).all()
        eligible = []
        for row in rows:
            try:
                data = json.loads(row.raw_result or "{}")
                ctx = data.get("market_context") or {}
                age = data.get("image_age_minutes")
                stale = data.get("image_freshness") == "stale_file_timestamp" or (
                    isinstance(age, (int, float)) and math.isfinite(age)
                    and age > max(MIN_STALE_IMAGE_MINUTES, TIMEFRAME_MINUTES.get(ctx.get("timeframe"), 15)))
            except (TypeError, ValueError):
                stale = False
            if not stale:
                eligible.append(row)
        total = len(eligible)
        by_signal = {"Buy": 0, "Sell": 0, "Hold": 0}
        agreement_count = 0
        for r in eligible:
            by_signal[r.final_signal] = by_signal.get(r.final_signal, 0) + 1
            if r.agreement:
                agreement_count += 1
        return {
            "total": total,
            "excluded_stale_images": len(rows) - total,
            "by_signal": by_signal,
            "agreement_rate": round(agreement_count / total, 3) if total else 0,
            "evaluation": evaluator.summarize(session),
            "model_comparison": evaluator.summarize_models(eligible),
        }
    finally:
        session.close()


@app.post("/evaluate")
def evaluate(horizon_hours: float = evaluator.DEFAULT_HORIZON_HOURS, utc_offset: int | None = None):
    """Hali baholanmagan signallarni MT5 tarixi bilan solishtiradi (MT5 ochiq bo'lishi kerak)."""
    try:
        return evaluator.evaluate_pending(horizon_hours, utc_offset)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/services")
def services(symbol: str = "XAUUSD"):
    """Cheap checks only: no paid Gemini requests or model inference."""
    settings = config.get_settings()
    quote = None
    acquired = mt5_context.MT5_LOCK.acquire(blocking=False)
    try:
        if acquired:
            quote = mt5_context.get_quote(symbol.strip().upper())
    finally:
        if acquired:
            mt5_context.MT5_LOCK.release()
    mt5_bridge_status = bridge_client.status()
    qwen = {"status": "unavailable", "model": settings["qwen_model"]}
    try:
        if bridge_client.enabled():
            models = bridge_client.request("ollama_tags", {}, timeout=3).get("models", [])
        else:
            with urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as response:
                models = json.load(response).get("models", [])
        qwen["status"] = "ready" if any(m.get("name") == settings["qwen_model"] or m.get("model") == settings["qwen_model"] for m in models) else "model_missing"
    except (OSError, ValueError, TypeError, bridge_client.BridgeError):
        pass
    qwen["enabled"] = settings["qwen_enabled"]
    return {"mt5": {"status": "ready" if quote else "unavailable" if acquired else "busy", "symbol": symbol.strip().upper(),
                    "bridge": mt5_bridge_status,
                    "quote_age_seconds": max(0, round(time.time() - quote["tick_time"])) if quote else None},
            "gemini": {"status": "configured" if settings["gemini_api_key"] else "not_configured", "model": settings["gemini_model"],
                       "enabled": settings["gemini_enabled"]},
            "qwen": qwen, "custom_model": custom_model.provenance_status()}


def control_activity():
    return {"paused": not config.get_settings()["analysis_enabled"],
            "in_flight": 2 - _analysis_slots._value, "qwen_in_flight": 2 - _qwen_slots._value}


from control_api import build_router
app.include_router(build_router(lambda: services(config.get_settings()["default_symbol"]), control_activity))


# Frontend'ni oxirida mount qilamiz -- shu tartibda yuqoridagi API yo'llar
# ("/analyze", "/history" va h.k.) ustunlik qiladi, qolgan hamma narsa
# static fayllardan (index.html, app.js, style.css) xizmat qiladi.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
