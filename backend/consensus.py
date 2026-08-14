"""Custom model va Gemini vision natijalarini birlashtirib yakuniy qaror chiqaradi.

Custom model test aniqligi past (~41%, tasodifiy daraja -- 33%) bo'lgani uchun u
yakuniy qarorni yolg'iz belgilamaydi -- faqat Gemini'ning fikrini tasdiqlaydi yoki
unga shubha uyg'otadi. Gemini asosiy manba hisoblanadi.

`training/backtest.py` orqali tekshirilgan: custom model ishonchi past bo'lganda
(<0.55) real expectancy nolga yaqin/manfiy, shuning uchun past ishonchli signal
Hold'ga tushiriladi -- kamroq, lekin sifatliroq signal berish strategiyasi.
"""
from config import VISION_AI_WEIGHT, CUSTOM_MODEL_WEIGHT, MIN_CUSTOM_MODEL_CONFIDENCE


def compute_consensus(custom: dict, vision_ai: dict) -> dict:
    custom_signal = custom["signal"]
    vision_signal = vision_ai["signal"]

    if vision_signal == "Hold":
        final_signal = "Hold"
        agreement = custom_signal == "Hold"
        note = "Gemini Hold deb hisobladi -- yakuniy qaror ham Hold."
    elif custom_signal == vision_signal:
        if custom["confidence"] < MIN_CUSTOM_MODEL_CONFIDENCE:
            final_signal = "Hold"
            agreement = False
            note = (
                f"Ikkala model {vision_signal} yo'nalishini ko'rsatdi, lekin custom model "
                f"ishonchi past ({custom['confidence']:.0%} < {MIN_CUSTOM_MODEL_CONFIDENCE:.0%}) "
                "-- backtest bo'yicha bu holatda signal ishonchsiz, shuning uchun Hold."
            )
        else:
            final_signal = vision_signal
            agreement = True
            note = "Ikkala model ham bir xil yo'nalishni ko'rsatdi va custom model ishonchi yetarli -- signal ishonchli."
    elif custom_signal == "Hold":
        final_signal = vision_signal
        agreement = False
        note = "Custom model Hold, Gemini esa signal berdi -- Gemini'ga tayanildi, ishonch o'rtacha."
    else:
        final_signal = "Hold"
        agreement = False
        note = "Ikkala model qarama-qarshi yo'nalish berdi (Buy vs Sell) -- xavfsizlik uchun Hold."

    if agreement:
        final_confidence = VISION_AI_WEIGHT * vision_ai["confidence"] + CUSTOM_MODEL_WEIGHT * custom["confidence"]
    elif final_signal == "Hold":
        final_confidence = 0.5
    else:
        final_confidence = vision_ai["confidence"] * 0.6  # kelishmaganda ishonchni pasaytiramiz

    entry_price = vision_ai.get("entry_price")
    tp_price = vision_ai.get("tp_price") if final_signal != "Hold" else None
    sl_price = vision_ai.get("sl_price") if final_signal != "Hold" else None
    entry_type = vision_ai.get("entry_type") if final_signal != "Hold" else None
    entry_note = vision_ai.get("entry_note") if final_signal != "Hold" else None

    return {
        "final_signal": final_signal,
        "final_confidence": round(final_confidence, 3),
        "agreement": agreement,
        "note": note,
        "entry_price": entry_price,
        "entry_type": entry_type,
        "entry_note": entry_note,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "custom_model": custom,
        "vision_ai": vision_ai,
    }
