"""Data pipeline uchun umumiy sozlamalar."""
import os

# Barcha yo'llar shu faylga nisbatan hisoblanadi (skript qayerdan ishga
# tushirilishidan qat'iy nazar to'g'ri ishlashi uchun).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

SYMBOL = "XAUUSD"
TIMEFRAME_NAME = "M15"

# MT5 dan qancha sham (bar) tortib olish
HISTORY_BARS = 60_000  # ~ several years of M15 data

# Sliding window: model bitta rasmda shuncha shamni ko'radi
WINDOW_SIZE = 80

# Windowlar orasidagi masofa (bar). 1 bo'lsa qo'shni windowlar deyarli bir xil
# (79/80 mos) bo'lib, model ularni yodlab olib overfit qiladi -- shuning uchun
# real xilma-xillik uchun stride WINDOW_SIZE ga yaqinroq tanlandi.
WINDOW_STRIDE = 20

# Label uchun kelajakka necha sham qarab chiqiladi
LOOKAHEAD_BARS = 24  # M15 da 6 soat

# ATR parametrlari
ATR_PERIOD = 14
TP_ATR_MULT = 1.5
SL_ATR_MULT = 1.0

# Chronological split nisbatlari
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
# qolgani (0.15) test

RAW_DATA_PATH = os.path.join(_THIS_DIR, "output", "ohlc_raw.parquet")
LABELED_DATA_PATH = os.path.join(_THIS_DIR, "output", "labeled_windows.parquet")
DATASET_DIR = os.path.join(_PROJECT_ROOT, "dataset")
