"""Raqamli indikatorlar (RSI, MACD, ATR, SMA, Bollinger, Momentum) asosida
o'qitilgan GradientBoosting model orqali inference.

Faol eski modelning o'qitish manbalari to'liq saqlanmagan; tarixiy aniqlik
mustaqil tasdiqlanmagan. Metadata bu cheklovni ochiq ko'rsatadi.

Rasm-CNN asosidagi eski versiya custom_model_legacy_cnn.py'da saqlangan.
"""
import json
import os
import hashlib
import math
import threading
from pathlib import Path

import joblib
import pandas as pd

from config import MODEL_DIR
import mt5_context
import reasoning

_MODEL_PATH = os.path.join(MODEL_DIR, "numeric_model.joblib")
_META_PATH = os.path.join(MODEL_DIR, "numeric_model_meta.json")

MODEL_LOCK = threading.RLock()
_snapshot_error = None


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(directory=None):
    """Load an internal local artifact pair; HTTP callers never supply a path."""
    directory = Path(directory or MODEL_DIR)
    model_path, meta_path = directory / "numeric_model.joblib", directory / "numeric_model_meta.json"
    if model_path.is_symlink() or meta_path.is_symlink():
        raise ValueError("Model fayli havola bo'lishi mumkin emas")
    with meta_path.open(encoding="utf-8") as source:
        meta = json.load(source)
    if not isinstance(meta, dict) or meta.get("model_sha256") != file_sha256(model_path):
        raise ValueError("Model SHA256 metadata bilan mos emas")
    if meta.get("feature_names") != mt5_context.FEATURE_NAMES or set(meta.get("classes", [])) != {"Buy", "Hold", "Sell"}:
        raise ValueError("Model xususiyatlari yoki klasslari joriy formatga mos emas")
    if meta.get("provenance_status") not in ("legacy_unverified", "verified_candidate"):
        raise ValueError("Model provenance holati qo'llab-quvvatlanmaydi")
    for key in ("accuracy", "majority_class_baseline_accuracy"):
        value = meta.get("validation", {}).get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("Model validation metrikasi yaroqsiz")
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise ValueError("Model faylini o'qib bo'lmadi") from exc
    if (list(getattr(model, "feature_names_in_", [])) != mt5_context.FEATURE_NAMES or
            getattr(model, "n_features_in_", None) != len(mt5_context.FEATURE_NAMES) or
            list(getattr(model, "classes_", [])) != meta["classes"]):
        raise ValueError("Model xususiyatlari yoki klasslari metadata bilan mos emas")
    probabilities = model.predict_proba(pd.DataFrame([[0.] * len(mt5_context.FEATURE_NAMES)], columns=mt5_context.FEATURE_NAMES))[0]
    if (len(probabilities) != 3 or not all(math.isfinite(float(p)) and 0 <= p <= 1 for p in probabilities) or
            not math.isclose(float(sum(probabilities)), 1., abs_tol=1e-6)):
        raise ValueError("Model ehtimollari yaroqsiz")
    if meta["model_sha256"] != file_sha256(model_path):
        raise ValueError("Model tekshirish vaqtida o'zgargan")
    return model, meta


def snapshot():
    with MODEL_LOCK:
        if _snapshot_error:
            raise RuntimeError(_snapshot_error)
        return _model, _meta


def install_snapshot(pair):
    global _model, _meta, _FEATURE_NAMES, _snapshot_error
    with MODEL_LOCK:
        _model, _meta = pair
        _FEATURE_NAMES = _meta["feature_names"]
        _snapshot_error = None


def disable_inference(reason):
    global _snapshot_error
    with MODEL_LOCK:
        _snapshot_error = reason


def metadata_warning(meta):
    messages = []
    if meta.get("provenance_status", "legacy_unverified") == "legacy_unverified":
        messages.append("Faol eski modelning o'qitish manbasi va tarixiy aniqligi mustaqil tasdiqlanmagan.")
    validation = meta.get("validation", {})
    accuracy, baseline = validation.get("accuracy"), validation.get("majority_class_baseline_accuracy")
    if isinstance(accuracy, (int, float)) and isinstance(baseline, (int, float)) and accuracy < baseline:
        messages.append("Validation aniqligi ko'pchilik klassini tanlash natijasidan past; sifat yaxshilanishi isbotlanmagan.")
    if meta.get("evaluation_limitations"):
        messages.append("Tarixiy davrlar oldin ko'rilgan; xarajatlar va yakuniy signal sifati mustaqil tasdiqlanmagan.")
    return " ".join(messages) or None


# A transaction interrupted between replacing the two files must be recovered
# before either artifact is loaded during server startup.
import model_control  # noqa: E402
model_control.recover_pending_transaction()
install_snapshot(load_model())


def provenance_status():
    _, meta = snapshot()
    status = meta.get("provenance_status", "legacy_unverified")
    return {"provenance_status": status,
            "warning": metadata_warning(meta), "model_sha256": meta["model_sha256"]}


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

    model, meta = snapshot()
    features = pd.DataFrame([mt5_context.to_feature_vector(market_context)], columns=meta["feature_names"])
    probs = model.predict_proba(features)[0]
    class_probs = dict(zip(model.classes_, probs))

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
        "model_sha256": meta["model_sha256"],
        "entry_price": own_levels["entry_price"],
        "tp_price": own_levels["tp_price"],
        "sl_price": own_levels["sl_price"],
        "trend": explanation["trend"],
        "reasoning": explanation["reasoning"],
    }
