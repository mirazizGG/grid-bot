"""FastAPI backend: chart rasmini custom model + Gemini vision bilan tahlil qilib,
yakuniy Buy/Sell/Hold + TP/SL signalini beradi va tarixni saqlaydi."""
import os
import time
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

import custom_model
import evaluator
import gemini_vision
import mt5_context
from consensus import compute_consensus
from db import SessionLocal, Analysis, save_analysis

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Grid-Bot Chart Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
MIN_STALE_IMAGE_MINUTES = 10  # rasm shundan (va timeframe bir bar vaqtidan) eski bo'lsa ogohlantiramiz


def _image_age_minutes(file_modified_ms: float | None) -> float | None:
    if not file_modified_ms:
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
    symbol: str = Form("XAUUSD"),
    timeframe: str = Form("M15"),
    file_modified_ms: float | None = Form(None),  # brauzerdagi File.lastModified (ms, UTC epoch)
):
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
        raise HTTPException(400, f"Qo'llab-quvvatlanmaydigan fayl turi: {file.content_type}")

    image_bytes = await file.read()

    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.verify()
        pil_image = Image.open(io.BytesIO(image_bytes))  # verify() dan keyin qayta ochish kerak
    except Exception:
        raise HTTPException(400, "Rasm faylini ochib bo'lmadi")

    filename = f"{uuid.uuid4().hex}.png"
    saved_path = os.path.join(UPLOAD_DIR, filename)
    pil_image.convert("RGB").save(saved_path, format="PNG")

    market_ctx = mt5_context.get_context(symbol.strip().upper(), timeframe.strip().upper())

    custom_result = custom_model.analyze(market_ctx)
    try:
        vision_result = gemini_vision.analyze_image(image_bytes, file.content_type, market_context=market_ctx)
    except Exception as exc:
        raise HTTPException(502, f"Gemini API xatosi: {exc}")

    consensus = compute_consensus(custom_result, vision_result)

    # MT5'dan jonli ma'lumot olingan bo'lsa, TP/SL/Entry'ni Gemini taxminiga
    # emas, ATR asosidagi aniq hisob-kitobga almashtiramiz -- hallucinatsiya
    # xavfini yo'qotadi.
    if market_ctx and consensus["final_signal"] != "Hold":
        levels = mt5_context.compute_trade_levels(market_ctx, consensus["final_signal"])
        consensus.update(levels)
        consensus["price_source"] = "mt5_live"
    else:
        consensus["price_source"] = "vision_estimate"
    consensus["market_context"] = market_ctx

    warnings = []
    age_min = _image_age_minutes(file_modified_ms)
    if age_min is not None:
        consensus["image_age_minutes"] = round(age_min, 1)
        tf_min = TIMEFRAME_MINUTES.get(timeframe.strip().upper(), 15)
        if age_min > max(MIN_STALE_IMAGE_MINUTES, tf_min):
            warnings.append(
                f"Rasm {_fmt_age(age_min)} oldin olingan, narx/indikatorlar esa hozirgi. Natija "
                "ishonchsiz bo'lishi mumkin -- MT5'da yangi skrinshot oling."
            )
    consensus["warnings"] = warnings

    record = save_analysis(f"/uploads/{filename}", consensus)

    return record.to_dict()


@app.get("/history")
def history(limit: int = 20, offset: int = 0):
    session = SessionLocal()
    try:
        rows = (
            session.query(Analysis)
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
        total = len(rows)
        by_signal = {"Buy": 0, "Sell": 0, "Hold": 0}
        agreement_count = 0
        for r in rows:
            by_signal[r.final_signal] = by_signal.get(r.final_signal, 0) + 1
            if r.agreement:
                agreement_count += 1
        return {
            "total": total,
            "by_signal": by_signal,
            "agreement_rate": round(agreement_count / total, 3) if total else 0,
            "evaluation": evaluator.summarize(session),
        }
    finally:
        session.close()


@app.post("/evaluate")
def evaluate(horizon_hours: float = evaluator.DEFAULT_HORIZON_HOURS, utc_offset: int | None = None):
    """Hali baholanmagan signallarni MT5 tarixi bilan solishtiradi (MT5 ochiq bo'lishi kerak)."""
    try:
        return evaluator.evaluate_pending(horizon_hours, utc_offset)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.get("/health")
def health():
    return {"status": "ok"}


# Frontend'ni oxirida mount qilamiz -- shu tartibda yuqoridagi API yo'llar
# ("/analyze", "/history" va h.k.) ustunlik qiladi, qolgan hamma narsa
# static fayllardan (index.html, app.js, style.css) xizmat qiladi.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
