"""Raqamli indikatorlar (RSI, MACD, ATR, SMA, Bollinger, Momentum) asosida
o'qitilgan GradientBoosting model orqali inference.

Eslatma: bu modelning test aniqligi ~36% (tasodifiy taxmindan -- 33.3%, 3 klass --
sal yuqori). Shuning uchun consensus'da past vazn bilan, faqat Gemini'ni
tasdiqlovchi ovoz sifatida ishlatiladi (backend/consensus.py).

Rasm-CNN asosidagi eski versiya custom_model_legacy_cnn.py'da saqlangan.
"""
import json
import os

import joblib

from config import MODEL_DIR
import mt5_context
import reasoning

_MODEL_PATH = os.path.join(MODEL_DIR, "numeric_model.joblib")
_META_PATH = os.path.join(MODEL_DIR, "numeric_model_meta.json")

_model = joblib.load(_MODEL_PATH)
with open(_META_PATH) as f:
    _meta = json.load(f)
_FEATURE_NAMES = _meta["feature_names"]


def analyze(market_context: dict | None) -> dict:
    """market_context -- mt5_context.get_context() natijasi. None bo'lsa
    (MT5 ulanmagan yoki symbol topilmagan) neytral Hold qaytaradi."""
    if market_context is None:
        return {
            "source": "custom_model",
            "signal": "Hold",
            "confidence": 0.34,
            "class_probs": {"Buy": 0.33, "Sell": 0.33, "Hold": 0.34},
            "note": "MT5 ulanmagani uchun raqamli indikatorlar hisoblanmadi",
            "trend": None,
            "reasoning": "MT5 ulanmagani uchun indikatorlarga asoslangan izoh hisoblanmadi.",
        }

    features = [mt5_context.to_feature_vector(market_context)]
    probs = _model.predict_proba(features)[0]
    class_probs = dict(zip(_model.classes_, probs))

    signal = max(class_probs, key=class_probs.get)
    confidence = float(class_probs[signal])

    # Custom model o'z alohida signaliga qarab TP/SL'ni ham hisoblab beradi
    # (yakuniy qaror Hold bo'lsa ham) -- shaffoflik uchun, "agar shu modelga
    # yolg'iz ishonilsa" degan ma'noda.
    own_levels = mt5_context.compute_trade_levels(market_context, signal)
    explanation = reasoning.explain(market_context, signal)

    return {
        "source": "custom_model",
        "signal": signal,
        "confidence": confidence,
        "class_probs": {k: float(v) for k, v in class_probs.items()},
        "entry_price": own_levels["entry_price"],
        "tp_price": own_levels["tp_price"],
        "sl_price": own_levels["sl_price"],
        "trend": explanation["trend"],
        "reasoning": explanation["reasoning"],
    }
