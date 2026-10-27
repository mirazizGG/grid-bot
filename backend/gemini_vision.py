"""Gemini API (vision) orqali chart rasmini mustaqil tahlil qilish."""
import json
import math

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL

_client = genai.Client(api_key=GEMINI_API_KEY)

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


def analyze_image(image_bytes: bytes, media_type: str = "image/png", market_context: dict | None = None) -> dict:
    parts = [types.Part.from_bytes(data=image_bytes, mime_type=media_type)]
    if market_context:
        parts.append(_format_context(market_context))
    parts.append("Shu chartni tahlil qil va JSON javob ber.")

    response = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=parts,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.0,  # bir xil kirishda bir xil javob -- natijalar takrorlanuvchan bo'lsin
        ),
    )

    parsed = json.loads(response.text)
    signal = parsed.get("signal")
    try:
        confidence = float(parsed["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Gemini javobida ishonch qiymati noto'g'ri") from exc
    if signal not in ("Buy", "Sell", "Hold") or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Gemini javobida signal yoki ishonch qiymati noto'g'ri")

    return {
        "source": "gemini",
        "prompt_version": PROMPT_VERSION,
        "signal": signal,
        "confidence": confidence,
        "trend": parsed.get("trend", ""),
        "reasoning": parsed.get("reasoning", ""),
        "entry_type": parsed.get("entry_type"),
        "entry_note": parsed.get("entry_note"),
        "entry_price": parsed.get("entry_price"),
        "tp_price": parsed.get("tp_price"),
        "sl_price": parsed.get("sl_price"),
    }
