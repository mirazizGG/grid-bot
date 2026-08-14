"""MT5 terminalidan jonli narx va texnik indikatorlarni olib, tahlil uchun
'ground truth' raqamli kontekst tayyorlaydi -- Gemini rasmdan taxmin qilish
o'rniga, custom model esa piksel patternidan emas, shu haqiqiy raqamli
indikatorlardan xulosa chiqaradi.

`compute_raw_indicators` va `to_feature_vector` funksiyalari custom model
o'qitilganda (training/train_numeric_model.py) ham, ishlatilganda (bu yerda,
jonli) ham AYNAN bir xil ishlatiladi -- train/serve orasida farq bo'lmasligi
uchun bu muhim."""
import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

TIMEFRAME_MAP = {}
if mt5 is not None:
    TIMEFRAME_MAP = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }

ATR_PERIOD = 14
TP_ATR_MULT = 1.5
SL_ATR_MULT = 1.0

# custom model shu tartibda va shu nomlar bilan xususiyat vektorini kutadi
FEATURE_NAMES = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_pct",
    "price_vs_high30_atr",
    "price_vs_low30_atr",
    "sma20_dist_atr",
    "sma50_dist_atr",
    "sma100_dist_atr",
    "bb_percent_b",
    "momentum_10_atr",
    "adx_14",
    "stoch_k",
    "stoch_d",
    "hour_sin",
    "hour_cos",
]


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(period).mean().iloc[-1]
    avg_loss = pd.Series(loss).rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    return float(100 - (100 / (1 + avg_gain / avg_loss)))


def _macd(closes: np.ndarray, fast=12, slow=26, signal=9) -> tuple[float, float]:
    s = pd.Series(closes)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    return float(pd.Series(tr).rolling(period).mean().iloc[-1])


def _sma(closes: np.ndarray, period: int) -> float:
    return float(pd.Series(closes).rolling(period).mean().iloc[-1])


def _bollinger_percent_b(closes: np.ndarray, period: int = 20, num_std: float = 2.0) -> float:
    s = pd.Series(closes)
    ma = s.rolling(period).mean().iloc[-1]
    std = s.rolling(period).std().iloc[-1]
    upper, lower = ma + num_std * std, ma - num_std * std
    denom = upper - lower
    if not denom:
        return 0.5
    return float((closes[-1] - lower) / denom)


def _momentum(closes: np.ndarray, period: int = 10) -> float:
    if len(closes) <= period:
        return 0.0
    return float(closes[-1] - closes[-1 - period])


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Trend kuchini o'lchaydi (yo'nalishsiz, 0-100). Wilder smoothing (EMA, alpha=1/period)."""
    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = closes[:-1]
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - prev_close), np.abs(lows[1:] - prev_close)))

    alpha = 1.0 / period
    atr_s = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()

    atr_safe = atr_s.replace(0, np.nan)
    plus_di = 100 * plus_dm_s / atr_safe
    minus_di = 100 * minus_dm_s / atr_safe
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    return float(adx) if pd.notna(adx) else 0.0


def _stochastic(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14, smooth_d: int = 3) -> tuple[float, float]:
    h = pd.Series(highs).rolling(period).max()
    l = pd.Series(lows).rolling(period).min()
    denom = (h - l).replace(0, np.nan)
    percent_k = 100 * (pd.Series(closes) - l) / denom
    percent_d = percent_k.rolling(smooth_d).mean()
    k = percent_k.iloc[-1]
    d = percent_d.iloc[-1]
    return (float(k) if pd.notna(k) else 50.0, float(d) if pd.notna(d) else 50.0)


def compute_raw_indicators(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, bar_hour: float | None = None) -> dict:
    """Bar tarixi (oxirgi ~100 bar) asosida xom indikator qiymatlarini hisoblaydi.
    Training (tarixiy slice) va live (MT5'dan olingan oxirgi bar) uchun bir xil.
    `bar_hour` -- oxirgi barning soati (0-23, server vaqti), savdo sessiyasini
    kodlash uchun (Osiyo/London/Nyu-York momentumi farq qiladi)."""
    atr = _atr(highs, lows, closes, ATR_PERIOD)
    macd_line, macd_signal = _macd(closes)
    stoch_k, stoch_d = _stochastic(highs, lows, closes)

    if bar_hour is None:
        hour_sin, hour_cos = 0.0, 0.0
    else:
        angle = 2 * np.pi * bar_hour / 24
        hour_sin, hour_cos = float(np.sin(angle)), float(np.cos(angle))

    return {
        "last_close": float(closes[-1]),
        "rsi_14": round(_rsi(closes, 14), 2),
        "macd": round(macd_line, 5),
        "macd_signal": round(macd_signal, 5),
        "atr_14": round(atr, 5),
        "sma_20": round(_sma(closes, 20), 5),
        "sma_50": round(_sma(closes, 50), 5),
        "sma_100": round(_sma(closes, min(100, len(closes))), 5),
        "bb_percent_b": round(_bollinger_percent_b(closes, 20), 4),
        "momentum_10": round(_momentum(closes, 10), 5),
        "adx_14": round(_adx(highs, lows, closes, 14), 2),
        "stoch_k": round(stoch_k, 2),
        "stoch_d": round(stoch_d, 2),
        "hour_sin": round(hour_sin, 4),
        "hour_cos": round(hour_cos, 4),
        "recent_high_30bars": float(highs[-30:].max()),
        "recent_low_30bars": float(lows[-30:].min()),
    }


def to_feature_vector(raw: dict) -> list[float]:
    """`compute_raw_indicators` natijasini custom model kutadigan, ATR birligida
    normallashtirilgan (shkaladan mustaqil) xususiyat vektoriga aylantiradi."""
    atr = raw["atr_14"] or 1e-9
    close = raw["last_close"]
    return [
        raw["rsi_14"],
        raw["macd"],
        raw["macd_signal"],
        raw["macd"] - raw["macd_signal"],
        atr / close if close else 0.0,
        (raw["recent_high_30bars"] - close) / atr,
        (close - raw["recent_low_30bars"]) / atr,
        (close - raw["sma_20"]) / atr,
        (close - raw["sma_50"]) / atr,
        (close - raw["sma_100"]) / atr,
        raw["bb_percent_b"],
        raw["momentum_10"] / atr,
        raw["adx_14"],
        raw["stoch_k"],
        raw["stoch_d"],
        raw["hour_sin"],
        raw["hour_cos"],
    ]


def get_context(symbol: str, timeframe_name: str, n_bars: int = 100) -> dict | None:
    """MT5'dan jonli narx + indikatorlarni oladi. MT5 mavjud bo'lmasa yoki
    symbol topilmasa None qaytaradi -- chaqiruvchi shunda eski (rasmdan
    taxmin qilingan) rejimga qaytadi."""
    if mt5 is None or timeframe_name not in TIMEFRAME_MAP:
        return None

    if not mt5.initialize():
        return None
    try:
        if not mt5.symbol_select(symbol, True):
            return None

        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_MAP[timeframe_name], 0, n_bars)
        tick = mt5.symbol_info_tick(symbol)
        if rates is None or len(rates) < 55 or tick is None:
            return None

        closes = rates["close"].astype(np.float64)
        highs = rates["high"].astype(np.float64)
        lows = rates["low"].astype(np.float64)
        bar_hour = pd.to_datetime(rates["time"][-1], unit="s").hour

        raw = compute_raw_indicators(highs, lows, closes, bar_hour=bar_hour)
        raw.update({
            "symbol": symbol,
            "timeframe": timeframe_name,
            "bid": float(tick.bid),
            "ask": float(tick.ask),
        })
        return raw
    finally:
        mt5.shutdown()


def compute_trade_levels(context: dict, signal: str) -> dict:
    """TP/SL'ni ATR asosida deterministik hisoblaydi -- Gemini yoki boshqa
    modelning narx taxminiga tayanmaydi, shu bilan hallucinatsiya xavfini
    yo'qotadi."""
    atr = context["atr_14"]
    if signal == "Buy":
        entry = context["ask"]
        tp = entry + TP_ATR_MULT * atr
        sl = entry - SL_ATR_MULT * atr
    elif signal == "Sell":
        entry = context["bid"]
        tp = entry - TP_ATR_MULT * atr
        sl = entry + SL_ATR_MULT * atr
    else:
        return {"entry_price": None, "tp_price": None, "sl_price": None}

    return {
        "entry_price": round(entry, 5),
        "tp_price": round(tp, 5),
        "sl_price": round(sl, 5),
    }
