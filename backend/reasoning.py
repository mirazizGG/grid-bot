"""Custom model uchun qoidaga asoslangan (rule-based) matnli izoh generatori.

Custom model o'zi "fikrlab" yozmaydi (u statistik classifier, LLM emas) --
lekin foydalanuvchiga signal *nega* chiqqanini tushunarli qilish uchun, uning
qarorini olib kelgan haqiqiy indikator qiymatlaridan (RSI, MACD, ADX,
Stochastic, trend) Gemini'ning "trend"/"reasoning" formatiga o'xshash matn
yasaymiz. Bu aniqlikni oshirmaydi -- faqat shaffoflik uchun."""


def _trend_text(ctx: dict) -> str:
    close = ctx["last_close"]
    sma20, sma50, sma100 = ctx["sma_20"], ctx["sma_50"], ctx["sma_100"]

    if close > sma20 > sma50 > sma100:
        return "aniq yuqoriga trend (narx barcha SMA'lardan tepada)"
    if close < sma20 < sma50 < sma100:
        return "aniq pastga trend (narx barcha SMA'lardan pastda)"
    if close > sma50:
        return "yuqoriga moyillik, lekin aralash signal"
    if close < sma50:
        return "pastga moyillik, lekin aralash signal"
    return "aniq trend yo'q, flat holat"


def _reasoning_text(ctx: dict, signal: str) -> str:
    rsi = ctx["rsi_14"]
    macd, macd_sig = ctx["macd"], ctx["macd_signal"]
    adx = ctx["adx_14"]
    stoch_k = ctx["stoch_k"]

    parts = []

    if rsi >= 70:
        parts.append(f"RSI {rsi:.0f} -- overbought")
    elif rsi <= 30:
        parts.append(f"RSI {rsi:.0f} -- oversold")
    else:
        parts.append(f"RSI {rsi:.0f} -- neytral")

    if macd > macd_sig:
        parts.append("MACD signal chizig'idan yuqorida (bullish)")
    else:
        parts.append("MACD signal chizig'idan pastda (bearish)")

    if adx >= 25:
        parts.append(f"ADX {adx:.0f} -- kuchli trend")
    elif adx < 20:
        parts.append(f"ADX {adx:.0f} -- trend zaif/yo'q")

    if stoch_k >= 80:
        parts.append(f"Stochastic {stoch_k:.0f} -- overbought")
    elif stoch_k <= 20:
        parts.append(f"Stochastic {stoch_k:.0f} -- oversold")

    body = ", ".join(parts)

    if signal == "Buy":
        return f"{body}. Bu kombinatsiya statistik modelda ko'pincha Buy klassiga yaqin chiqadi."
    if signal == "Sell":
        return f"{body}. Bu kombinatsiya statistik modelda ko'pincha Sell klassiga yaqin chiqadi."
    return f"{body}. Ko'rsatkichlar aralash -- model aniq yo'nalish ko'rmadi, shuning uchun Hold."


def explain(market_context: dict, signal: str) -> dict:
    return {
        "trend": _trend_text(market_context),
        "reasoning": _reasoning_text(market_context, signal),
    }
