"""Backend uchun umumiy sozlamalar."""
import os
import copy
import json
import math
import re
import tempfile
import threading
from pathlib import Path

from dotenv import load_dotenv
import mt5_clock

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_THIS_DIR, ".env"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3.5:4b")
QWEN_TIMEOUT = float(os.environ.get("QWEN_TIMEOUT", "120"))

MODEL_DIR = os.path.join(_THIS_DIR, "model")
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "chart_signal_model.onnx")
NORM_STATS_PATH = os.path.join(MODEL_DIR, "norm_stats.json")

DATA_DIR = Path(os.environ.get("GRID_BOT_DATA_DIR", _THIS_DIR)).resolve()
DB_PATH = str(DATA_DIR / "analyses.db")
UPLOAD_DIR = DATA_DIR / "uploads"
LOG_DIR = DATA_DIR / "logs" if os.environ.get("GRID_BOT_DATA_DIR") else Path(_THIS_DIR)

# Faol legacy modelning tarixiy aniqligi mustaqil tasdiqlanmagan.
# Consensus'da uning ovoziga past vazn beriladi -- Gemini asosiy manba.
CUSTOM_MODEL_WEIGHT = 0.25
VISION_AI_WEIGHT = 0.75

# Tarixiy tajribadan olingan amaldagi filtr. U testda tanlangan va xarajatlar
# hisoblanmagan; mustaqil foydalilik yoki kalibrlangan ehtimolni tasdiqlamaydi.
MIN_CUSTOM_MODEL_CONFIDENCE = 0.55

# Custom model va Gemini qarama-qarshi signal bersa (Buy vs Sell), Gemini
# ishonchi shu chegaradan yuqori bo'lsa Gemini signaliga tayaniladi (custom
# model ovozi shu holatda e'tiborga olinmaydi -- u ko'pincha xato bo'ladi).
# Past bo'lsa xavfsizlik uchun hamon Hold.
MIN_VISION_OVERRIDE_CONFIDENCE = 0.65

RUNTIME_DIR = DATA_DIR / "runtime"
SETTINGS_PATH = RUNTIME_DIR / "settings.json"
SETTINGS_LOCK = threading.RLock()
_DEFAULT_SETTINGS = {
    "analysis_enabled": True, "gemini_enabled": True, "qwen_enabled": True,
    "gemini_api_key": GEMINI_API_KEY, "gemini_model": GEMINI_MODEL,
    "qwen_model": QWEN_MODEL, "qwen_timeout": QWEN_TIMEOUT,
    "min_custom_confidence": MIN_CUSTOM_MODEL_CONFIDENCE,
    "min_vision_override_confidence": MIN_VISION_OVERRIDE_CONFIDENCE,
    "custom_model_weight": CUSTOM_MODEL_WEIGHT, "vision_ai_weight": VISION_AI_WEIGHT,
    "default_symbol": "XAUUSD", "default_timeframe": "M15",
    "mt5_clock_profile": mt5_clock.normalize_profile(),
}
_settings = copy.deepcopy(_DEFAULT_SETTINGS)
SETTINGS_WARNING = None


def _validate_settings(values):
    unknown = set(values) - set(_DEFAULT_SETTINGS)
    if unknown:
        raise ValueError("Noma'lum sozlama maydoni")
    values["mt5_clock_profile"] = mt5_clock.normalize_profile(values["mt5_clock_profile"])
    for key in ("analysis_enabled", "gemini_enabled", "qwen_enabled"):
        if type(values[key]) is not bool:
            raise ValueError(f"{key}: yoqish/o'chirish qiymati talab qilinadi")
    for key in ("min_custom_confidence", "min_vision_override_confidence", "custom_model_weight", "vision_ai_weight"):
        value = values[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key}: 0 va 1 orasidagi son talab qilinadi")
    if abs(values["custom_model_weight"] + values["vision_ai_weight"] - 1) > 1e-9:
        raise ValueError("Model vaznlari yig'indisi 1 bo'lishi kerak")
    timeout = values["qwen_timeout"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 5 <= timeout <= 600:
        raise ValueError("Qwen kutish muddati 5–600 soniya bo'lishi kerak")
    patterns = {"gemini_model": r"gemini-[A-Za-z0-9._-]{1,80}",
                "qwen_model": r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,120}",
                "default_symbol": r"[A-Z0-9._#-]{1,32}"}
    for key, pattern in patterns.items():
        if not isinstance(values[key], str) or not re.fullmatch(pattern, values[key]):
            raise ValueError(f"{key}: qiymat noto'g'ri")
    if values["default_timeframe"] not in ("M1", "M5", "M15", "M30", "H1", "H4", "D1"):
        raise ValueError("Vaqt oralig'i noto'g'ri")
    api_key = values["gemini_api_key"]
    if not isinstance(api_key, str) or len(api_key) > 512 or any(c.isspace() for c in api_key):
        raise ValueError("Gemini kaliti noto'g'ri formatda")


def get_settings(public=False):
    with SETTINGS_LOCK:
        values = copy.deepcopy(_settings)
    if public:
        values["api_key_configured"] = bool(values.pop("gemini_api_key"))
        values["warning"] = SETTINGS_WARNING
    return values


def save_settings(changes):
    """Validate and atomically persist first; leave live settings intact on failure."""
    global _settings, SETTINGS_WARNING
    if not isinstance(changes, dict):
        raise ValueError("Sozlamalar JSON obyekt bo'lishi kerak")
    if set(changes) - set(_DEFAULT_SETTINGS):
        raise ValueError("Noma'lum sozlama maydoni")
    with SETTINGS_LOCK:
        values = {**_settings, **changes}
        _validate_settings(values)
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=SETTINGS_PATH.parent,
                                             suffix=".tmp", delete=False) as output:
                name = output.name
                json.dump(values, output, allow_nan=False, indent=2)
                output.flush()
                os.fsync(output.fileno())
            os.replace(name, SETTINGS_PATH)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)
        _settings = values
        SETTINGS_WARNING = None
        return get_settings(public=True)


if SETTINGS_PATH.is_file():
    try:
        stored = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if not isinstance(stored, dict) or set(stored) - set(_DEFAULT_SETTINGS):
            raise ValueError("Invalid settings")
        loaded = {**_DEFAULT_SETTINGS, **stored}
        _validate_settings(loaded)
        _settings = loaded
    except (OSError, ValueError, TypeError):
        SETTINGS_WARNING = "Saqlangan sozlamalar o'qilmadi; boshlang'ich qiymatlar ishlatilmoqda."

# Import-time consumers use the persisted values on restart. Runtime updates
# are applied explicitly to each provider by the control API.
GEMINI_API_KEY = _settings["gemini_api_key"]
GEMINI_MODEL = _settings["gemini_model"]
QWEN_MODEL = _settings["qwen_model"]
QWEN_TIMEOUT = _settings["qwen_timeout"]
