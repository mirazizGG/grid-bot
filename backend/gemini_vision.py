"""Gemini API (vision) orqali chart rasmini mustaqil tahlil qilish."""
import json
import math
import threading

import httpx
from google import genai
from google.genai import errors
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL

_client = None
_CLIENT_LOCK = threading.RLock()


class GeminiResponseError(ValueError):
    """A locally detected invalid provider response, without retaining its text."""


def safe_error(exc) -> dict:
    """Classify failures using fixed text; never expose provider exception text."""
    code, kind = None, "unknown"
    if isinstance(exc, GeminiResponseError):
        kind = "response"
    elif isinstance(exc, httpx.TimeoutException):
        kind = "timeout"
    elif isinstance(exc, httpx.TransportError):
        kind = "network"
    elif isinstance(exc, errors.APIError):
        raw_code = getattr(exc, "code", None)
        code = raw_code if type(raw_code) is int and 400 <= raw_code <= 599 else None
        statuses = {"UNAUTHENTICATED", "PERMISSION_DENIED", "NOT_FOUND", "RESOURCE_EXHAUSTED",
                    "INVALID_ARGUMENT", "FAILED_PRECONDITION", "INTERNAL", "UNAVAILABLE", "DEADLINE_EXCEEDED"}
        raw_status = getattr(exc, "status", None)
        status = raw_status if isinstance(raw_status, str) and raw_status in statuses else None
        reason_kinds = {"API_KEY_INVALID": "authentication", "BILLING_DISABLED": "billing",
                        "CONSUMER_INVALID": "project", "CONSUMER_SUSPENDED": "project",
                        "SERVICE_DISABLED": "project", "PROJECT_DELETED": "project",
                        "API_KEY_SERVICE_BLOCKED": "permission", "API_KEY_HTTP_REFERRER_BLOCKED": "permission",
                        "API_KEY_IP_ADDRESS_BLOCKED": "permission", "API_KEY_ANDROID_APP_BLOCKED": "permission",
                        "API_KEY_IOS_APP_BLOCKED": "permission"}
        payload = getattr(exc, "details", None)
        if isinstance(payload, dict):
            payload = payload.get("error", payload)
            details = payload.get("details", []) if isinstance(payload, dict) else []
        else:
            details = payload if isinstance(payload, list) else []
        reason_kind = None
        if isinstance(details, list):
            for detail in details[:16]:
                if not isinstance(detail, dict) or detail.get("@type") != "type.googleapis.com/google.rpc.ErrorInfo":
                    continue
                reason = detail.get("reason")
                if isinstance(reason, str) and reason in reason_kinds:
                    reason_kind = reason_kinds[reason]
                    break
        if code == 429:
            kind = "quota"
        elif code is not None and code >= 500:
            kind = "service"
        elif reason_kind:
            kind = reason_kind
        elif code == 401:
            kind = "authentication"
        elif code == 402:
            kind = "billing"
        elif code == 403:
            kind = "permission"
        elif code == 404:
            kind = "model"
        elif code == 400:
            kind = "request"
        elif code is None:
            kind = {"UNAUTHENTICATED": "authentication", "PERMISSION_DENIED": "permission",
                    "NOT_FOUND": "model", "RESOURCE_EXHAUSTED": "quota", "INVALID_ARGUMENT": "request",
                    "FAILED_PRECONDITION": "request", "INTERNAL": "service", "UNAVAILABLE": "service",
                    "DEADLINE_EXCEEDED": "service"}.get(status, "unknown")
    messages = {
        "authentication": "Gemini API kaliti tasdiqlanmadi. Kalitni tekshiring.",
        "billing": "Gemini loyihasining to'lov holati so'rovga ruxsat bermadi. AI Studio billing sozlamalarini tekshiring.",
        "project": "Gemini loyihasi yoki undagi API xizmati so'rovga ruxsat bermadi. AI Studio loyiha sozlamalarini tekshiring.",
        "permission": "Gemini bu kalit yoki loyihaga ruxsat bermadi. Kalit cheklovlari va loyiha ruxsatlarini tekshiring.",
        "model": "Gemini modeli ushbu API yoki loyiha uchun topilmadi. Model nomi va kirish huquqini tekshiring.",
        "quota": "Gemini kvotasi yoki so'rov chegarasi oshgan. AI Studio dagi limitlarni tekshiring.",
        "request": "Gemini so'rovni qabul qilmadi. Model va so'rov sozlamalarini tekshiring.",
        "service": "Gemini xizmati so'rovni bajarolmadi. Xizmat holatini tekshiring va keyinroq qayta urinib ko'ring.",
        "timeout": "Gemini javobini kutish muddati tugadi. Keyinroq qayta urinib ko'ring.",
        "network": "Gemini bilan aloqa o'rnatilmadi. Internet, DNS, TLS va proksi sozlamalarini tekshiring.",
        "response": "Gemini javobi kutilgan tuzilishga mos kelmadi. Qayta tahlil qiling.",
        "unknown": "Gemini tahlili tugamadi. Server jurnali va xizmat sozlamalarini tekshiring.",
    }
    message = messages[kind] + (f" (HTTP {code})" if code is not None else "")
    return {"kind": kind, "code": code, "message": message}


def _get_client():
    global _client
    with _CLIENT_LOCK:
        if _client is None:
            if not GEMINI_API_KEY.strip():
                raise RuntimeError("Gemini API kaliti sozlanmagan. Boshqaruv panelida Gemini kalitini kiriting.")
            _client = genai.Client(api_key=GEMINI_API_KEY)
        return _client


def configure(api_key, model):
    global GEMINI_API_KEY, GEMINI_MODEL, _client
    with _CLIENT_LOCK:
        if GEMINI_API_KEY != api_key:
            _client = None
        GEMINI_API_KEY, GEMINI_MODEL = api_key, model


_SYSTEM_PROMPT = """Sen tajribali texnik tahlilchisan. Senga savdo chart (candlestick) \
rasmi beriladi, ba'zan unga qo'shimcha ravishda MT5'dan olingan JONLI (real) narx va \
indikator ma'lumotlari ham beriladi. Trend, support/resistance darajalarini, sham \
pattern'larini tahlil qil va savdo qarori chiqar.

Agar JONLI MA'LUMOT berilgan bo'lsa:
- U haqiqiy, aniq raqamlar -- rasmdan narxni o'qishga urinma, shu raqamlarga tayan.
- RSI (70+ = overbought, 30- = oversold), MACD (signal chiziqdan yuqori/past kesishuvi) \
  va ATR (volatillik)ni qaror qabul qilishda albatta hisobga ol.
- entry_price SIFATIDA HAR DOIM berilgan bid (Sell uchun) yoki ask (Buy uchun) qiymatini qo'y -- \
  signal Buy/Sell bo'lsa buni hech qachon null qoldirma.
- tp_price/sl_price MAJBURIY: signal Buy yoki Sell bo'lsa, IKKALASINI HAM albatta son sifatida \
  ber -- hech qachon null qoldirma. O'ZINGNING mustaqil fikringni ishlat: rasmda ko'rinadigan \
  support/resistance darajalari, oxirgi high/low va ATR (berilgan bo'lsa)ga asoslanib hisobla. \
  Aniq son bilmasang ham, ATR yoki oxirgi high/low asosida oqilona baho ber -- bo'sh qoldirish \
  taqiqlangan. Bu backend'ning alohida hisob-kitobidan farq qilishi mumkin -- bu normal, \
  ikkalasi alohida ko'rsatiladi. FAQAT signal Hold bo'lsa barchasini null qo'y.

Qo'shimcha qoidalar (sham jadvali, swing struktura va yuqori timeframe berilgan bo'lsa):
- Rasmdagi sham shakli va chiziqlarni sham jadvali bilan solishtir; ikkisi ziddiyatda bo'lsa \
  jadvaldagi raqamlarga ishon (rasmdan narxni aniq o'qish ishonchsiz).
- Yuqori timeframe trendiga qarshi signal bermaslikka harakat qil; qarshi bo'lsa ishonchni pasaytir.
- ADX 20 dan past (yon bozor) bo'lsa, narx range chetida bo'lmasa HOLD ber -- yon bozorda \
  trend signallari ko'pincha yolg'on chiqadi.
- Swing struktura (HH/HL yoki LH/LL) trend yo'nalishini tasdiqlamasa, ishonchni pasaytir.
- Ishonch (confidence)ni halol ber: shubha bo'lsa 0.5 atrofida, faqat bir nechta mustaqil \
  omil (rasm + indikator + struktura + yuqori TF) bir yo'nalishni ko'rsatsa 0.7 dan yuqori.

Agar JONLI MA'LUMOT berilmagan bo'lsa, rasmda narx o'qi ko'rinsa taxminiy son ber, \
ko'rinmasa null qo'y.

FAQAT quyidagi JSON formatida javob ber, boshqa hech qanday matn yozma:
{
  "signal": "Buy" | "Sell" | "Hold",
  "confidence": 0.0 dan 1.0 gacha son,
  "trend": qisqa trend tavsifi (masalan "yuqoriga trend, oxirgi pullback"),
  "reasoning": 1-2 jumlali asos (jonli ma'lumot berilgan bo'lsa RSI/MACD'ga ham ishora qil),
  "entry_type": "market" | "wait_pullback" | "wait_breakout" | null,
  "entry_note": savdoga QACHON va QANDAY kirish kerakligi haqida 1 jumlali aniq \
tavsiya. signal Hold bo'lsa null qo'y.,
  "entry_price": yuqoridagi qoidaga ko'ra son yoki null,
  "tp_price": yuqoridagi qoidaga ko'ra son yoki null,
  "sl_price": yuqoridagi qoidaga ko'ra son yoki null
}

entry_type qiymatlari: "market" = hozir bozor narxida kirish mumkin, "wait_pullback" = \
narx orqaga qaytishini kutib pastroq/yuqoriroq narxdan kirish yaxshiroq, "wait_breakout" = \
muhim darajani buzib o'tishini kutib kirish yaxshiroq."""


PROMPT_VERSION = "v2-structure"  # baholashda eski/yangi promptni solishtirish uchun


def _format_extra_context(ctx: dict) -> str:
    """Rasmda ko'rinadigan, lekin raqam sifatida aniqroq bo'lgan ma'lumotlar:
    sham jadvali, swing struktura, yuqori timeframe trendi."""
    parts = []

    candles = ctx.get("candles")
    if candles:
        rows = "\n".join(
            f"  {i - len(candles) + 1:>3}: O {o} H {h} L {l} C {c} vol {v}"
            for i, (o, h, l, c, v) in enumerate(candles)
        )
        parts.append(f"Oxirgi {len(candles)} sham (eski -> yangi, 0 = oxirgi bar):\n{rows}")

    swings = ctx.get("swings") or {}
    if swings.get("swing_highs") or swings.get("swing_lows"):
        sh = ", ".join(f"{p} ({b} bar oldin)" for b, p in swings.get("swing_highs", []))
        sl = ", ".join(f"{p} ({b} bar oldin)" for b, p in swings.get("swing_lows", []))
        parts.append(
            f"Swing nuqtalar (yangisi birinchi): high: {sh or '-'}; low: {sl or '-'}\n"
            f"Struktura: {swings.get('structure') or 'aniqlanmadi'}"
        )

    higher = ctx.get("higher_tf") or {}
    if higher:
        lines = [f"  {tf}: trend {s['direction']}, ADX {s['adx']}, RSI {s['rsi']}" for tf, s in higher.items()]
        parts.append("Yuqori timeframe trendi:\n" + "\n".join(lines))

    return "\n".join(parts)


def _format_context(ctx: dict) -> str:
    base = _format_base_context(ctx)
    extra = _format_extra_context(ctx)
    return f"{base}\n{extra}" if extra else base


def _format_base_context(ctx: dict) -> str:
    return (
        f"JONLI MA'LUMOT ({ctx['symbol']}, {ctx['timeframe']}):\n"
        f"- Bid/Ask: {ctx['bid']} / {ctx['ask']}\n"
        f"- Oxirgi close: {ctx['last_close']}\n"
        f"- RSI(14): {ctx['rsi_14']}\n"
        f"- MACD: {ctx['macd']} (signal chiziq: {ctx['macd_signal']})\n"
        f"- ATR(14): {ctx['atr_14']}\n"
        f"- SMA(20)/SMA(50)/SMA(100): {ctx['sma_20']} / {ctx['sma_50']} / {ctx['sma_100']}\n"
        f"- Bollinger %B: {ctx['bb_percent_b']} (0=quyi band, 1=yuqori band)\n"
        f"- Momentum(10 bar): {ctx['momentum_10']}\n"
        f"- ADX(14): {ctx['adx_14']} (25+ = kuchli trend, 20- = trend yo'q/flat)\n"
        f"- Stochastic %K/%D: {ctx['stoch_k']} / {ctx['stoch_d']}\n"
        f"- Oxirgi 30 bar high/low: {ctx['recent_high_30bars']} / {ctx['recent_low_30bars']}"
    )


def _finite_number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeminiResponseError(f"Gemini javobida {field} son bo'lishi kerak")
    try:
        number = float(value)
    except OverflowError as exc:
        raise GeminiResponseError(f"Gemini javobida {field} qiymati noto'g'ri") from exc
    if not math.isfinite(number):
        raise GeminiResponseError(f"Gemini javobida {field} qiymati chekli son bo'lishi kerak")
    return number


def _validate_response(text: str) -> dict:
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise GeminiResponseError("Gemini javobi kutilgan JSON formatida emas") from exc
    if not isinstance(parsed, dict):
        raise GeminiResponseError("Gemini javobi JSON obyekt bo'lishi kerak")

    signal = parsed.get("signal")
    confidence = _finite_number(parsed.get("confidence"), "confidence")
    if signal not in ("Buy", "Sell", "Hold") or not 0 <= confidence <= 1:
        raise GeminiResponseError("Gemini javobida signal yoki ishonch qiymati noto'g'ri")
    for field in ("trend", "reasoning"):
        if not isinstance(parsed.get(field), str):
            raise GeminiResponseError(f"Gemini javobida {field} matn bo'lishi kerak")
    entry_type = parsed.get("entry_type")
    if entry_type not in (None, "market", "wait_pullback", "wait_breakout"):
        raise GeminiResponseError("Gemini javobida entry_type qiymati noto'g'ri")
    entry_note = parsed.get("entry_note")
    if entry_note is not None and not isinstance(entry_note, str):
        raise GeminiResponseError("Gemini javobida entry_note matn yoki null bo'lishi kerak")

    prices = {}
    for field in ("entry_price", "tp_price", "sl_price"):
        value = parsed.get(field)
        if value is not None:
            value = _finite_number(value, field)
            if value <= 0:
                raise GeminiResponseError(f"Gemini javobida {field} musbat son bo'lishi kerak")
        prices[field] = value

    return {
        "source": "gemini",
        "prompt_version": PROMPT_VERSION,
        "signal": signal,
        "confidence": confidence,
        "trend": parsed["trend"],
        "reasoning": parsed["reasoning"],
        "entry_type": entry_type,
        "entry_note": entry_note,
        **prices,
    }


def analyze_image(image_bytes: bytes, media_type: str = "image/png", market_context: dict | None = None,
                  settings: dict | None = None) -> dict:
    with _CLIENT_LOCK:
        key = settings["gemini_api_key"] if settings else GEMINI_API_KEY
        model = settings["gemini_model"] if settings else GEMINI_MODEL
        temporary_client = key != GEMINI_API_KEY
        if key == GEMINI_API_KEY:
            client = _get_client()
        elif not key.strip():
            raise RuntimeError("Gemini API kaliti sozlanmagan")
        else:
            client = genai.Client(api_key=key)
    parts = [types.Part.from_bytes(data=image_bytes, mime_type=media_type)]
    if market_context:
        parts.append(_format_context(market_context))
    parts.append("Shu chartni tahlil qil va JSON javob ber.")

    try:
        response = client.models.generate_content(
            model=model,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
    finally:
        if temporary_client:
            client.close()

    result = _validate_response(response.text)
    result["model"] = model
    return result
