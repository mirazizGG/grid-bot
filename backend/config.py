"""Backend uchun umumiy sozlamalar."""
import os

from dotenv import load_dotenv

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_THIS_DIR, ".env"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

MODEL_DIR = os.path.join(_THIS_DIR, "model")
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "chart_signal_model.onnx")
NORM_STATS_PATH = os.path.join(MODEL_DIR, "norm_stats.json")

DB_PATH = os.path.join(_THIS_DIR, "analyses.db")

# Custom model test aniqligi past (~41%, tasodifiy daraja -- 33%) bo'lgani
# uchun consensus'da uning ovoziga past vazn beriladi -- Gemini asosiy manba.
CUSTOM_MODEL_WEIGHT = 0.25
VISION_AI_WEIGHT = 0.75

# training/backtest.py orqali tekshirilgan: custom model ishonchi shu
# chegaradan past bo'lganda haqiqiy (out-of-sample) expectancy manfiy yoki
# nolga yaqin chiqadi; 0.55+ da esa ijobiy (+0.07R..+0.10R/savdo) natija
# berdi (74 ta savdo namunasida). Shuning uchun past ishonchli signal Hold'ga
# tushiriladi.
MIN_CUSTOM_MODEL_CONFIDENCE = 0.55
