"""Custom model va Gemini vision natijalarini birlashtirib yakuniy qaror chiqaradi.

Custom modelning tarixiy aniqligi mustaqil tasdiqlanmagan. U
yakuniy qarorni yolg'iz belgilamaydi -- Gemini'ning fikrini tasdiqlaydi yoki
unga shubha uyg'otadi. Gemini asosiy manba hisoblanadi.

Custom modelning avvalgi tajribadan olingan filtri qo'llanadi. Ushbu chegara
mustaqil foydalilikni yoki baholarning kalibrlanganligini tasdiqlamaydi.
"""
from config import (
    VISION_AI_WEIGHT,
    CUSTOM_MODEL_WEIGHT,
    MIN_CUSTOM_MODEL_CONFIDENCE,
    MIN_VISION_OVERRIDE_CONFIDENCE,
)
from config import get_settings


def compute_consensus(custom: dict, vision_ai: dict, settings: dict | None = None) -> dict:
    settings = settings or get_settings()
    min_custom = settings["min_custom_confidence"]
    min_override = settings["min_vision_override_confidence"]
    custom_signal = custom["signal"]
    vision_signal = vision_ai["signal"]

    if vision_signal == "Hold":
        final_signal = "Hold"
        agreement = custom_signal == "Hold"
        note = "Gemini Hold deb hisobladi -- yakuniy qaror ham Hold."
    elif custom_signal == vision_signal:
        if custom["confidence"] < min_custom:
            final_signal = "Hold"
            agreement = False
            note = (
                f"Ikkala model {vision_signal} yo'nalishini ko'rsatdi, lekin custom model "
                f"ishonchi past ({custom['confidence']:.0%} < {min_custom:.0%}) "
                "-- belgilangan filtr bo'yicha Hold."
            )
        else:
            final_signal = vision_signal
            agreement = True
            note = "Ikkala model bir xil yo'nalishni ko'rsatdi va custom model bahosi filtrdan o'tdi. Bu natija muvaffaqiyatni kafolatlamaydi."
    elif custom_signal == "Hold":
        final_signal = vision_signal
        agreement = False
        note = "Custom model Hold, Gemini esa signal berdi -- Gemini yo'nalishi olindi."
    elif vision_ai["confidence"] >= min_override:
        final_signal = vision_signal
        agreement = False
        note = (
            f"Custom model qarama-qarshi yo'nalish berdi (Sell/Buy ziddiyat), lekin Gemini ishonchi "
            f"yetarlicha yuqori ({vision_ai['confidence']:.0%} >= {min_override:.0%}) "
            "-- Gemini signaliga tayanildi, ehtiyot bo'ling."
        )
    else:
        final_signal = "Hold"
        agreement = False
        note = (
            "Ikkala model qarama-qarshi yo'nalish berdi (Buy vs Sell) va Gemini ishonchi ham "
            f"yetarli emas ({vision_ai['confidence']:.0%} < {min_override:.0%}) "
            "-- xavfsizlik uchun Hold."
        )

    if final_signal == "Hold":
        final_confidence = None  # Hold uchun ishonch ma'nosiz (avval qattiq 0.5 edi) -- dashboard "—" ko'rsatadi
    elif agreement:
        final_confidence = settings["vision_ai_weight"] * vision_ai["confidence"] + settings["custom_model_weight"] * custom["confidence"]
    else:
        final_confidence = vision_ai["confidence"] * 0.85  # kelishmagan, lekin Gemini ishonchi yuqori bo'lgani uchun kamroq pasaytiramiz

    entry_price = vision_ai.get("entry_price") if final_signal != "Hold" else None
    tp_price = vision_ai.get("tp_price") if final_signal != "Hold" else None
    sl_price = vision_ai.get("sl_price") if final_signal != "Hold" else None
    entry_type = vision_ai.get("entry_type") if final_signal != "Hold" else None
    entry_note = vision_ai.get("entry_note") if final_signal != "Hold" else None

    return {
        "final_signal": final_signal,
        "final_confidence": round(final_confidence, 3) if final_confidence is not None else None,
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
