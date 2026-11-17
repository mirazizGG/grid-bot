"""MT5 terminalidan jonli narx va texnik indikatorlarni olib, tahlil uchun
'ground truth' raqamli kontekst tayyorlaydi -- Gemini rasmdan taxmin qilish
o'rniga, custom model esa piksel patternidan emas, shu haqiqiy raqamli
indikatorlardan xulosa chiqaradi.

`compute_raw_indicators` va `to_feature_vector` funksiyalari custom model
o'qitilganda (training/train_numeric_model.py) ham, ishlatilganda (bu yerda,
jonli) ham AYNAN bir xil ishlatiladi -- train/serve orasida farq bo'lmasligi
uchun bu muhim."""
import time
import threading
import math
from types import SimpleNamespace

import bridge_client
import config
import mt5_clock

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


class MT5ReadError(ValueError):
    """Safe MT5 connection/transport failure for callers requesting diagnostics."""


TIMEFRAME_MAP = {}
MT5_LOCK = threading.RLock()
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
MAX_QUOTE_AGE_SECONDS = 60
TIMEFRAME_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800,
                     "H1": 3600, "H4": 14400, "D1": 86400}
# No broker session calendar is available. Allow weekends/holiday closures,
# but do not accept an arbitrarily old cache just because quotes resumed.
MAX_HISTORY_SESSION_GAP_SECONDS = 7 * 86400
# training/backtest.py orqali tekshirilgan: 2:1 nisbat (ishonch>=0.55 bilan
# birga) out-of-sample'da eng barqaror ijobiy expectancy berdi (+0.095R/savdo).
TP_ATR_MULT = 2.0
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


N_CONTEXT_CANDLES = 20
SWING_K = 3  # swing nuqta: har ikki tomonida K ta bardan balandroq/pastroq
# joriy timeframe'dan yuqori timeframe'lar (Gemini'ga katta rasm kontekstini beradi)
HIGHER_TIMEFRAMES = {
    "M1": ["M15", "H1"], "M5": ["M15", "H1"], "M15": ["H1", "H4"], "M30": ["H1", "H4"],
    "H1": ["H4", "D1"], "H4": ["D1"], "D1": [],
}


def find_swings(highs: np.ndarray, lows: np.ndarray, k: int = SWING_K) -> dict:
    """Oxirgi 2 ta swing high va 2 ta swing low (eng yangisi birinchi) va struktura
    (HH/HL = yuqoriga, LH/LL = pastga) ni topadi. Oxirgi k bar tasdiqlanmagan,
    shuning uchun hisobga olinmaydi."""
    n = len(highs)
    swing_highs, swing_lows = [], []
    for i in range(n - k - 1, k - 1, -1):
        if highs[i] == highs[i - k:i + k + 1].max() and len(swing_highs) < 2:
            swing_highs.append((n - 1 - i, round(float(highs[i]), 5)))  # (necha bar oldin, narx)
        if lows[i] == lows[i - k:i + k + 1].min() and len(swing_lows) < 2:
            swing_lows.append((n - 1 - i, round(float(lows[i]), 5)))
        if len(swing_highs) == 2 and len(swing_lows) == 2:
            break

    structure = None
    if len(swing_highs) == 2 and len(swing_lows) == 2:
        hh = swing_highs[0][1] > swing_highs[1][1]
        hl = swing_lows[0][1] > swing_lows[1][1]
        if hh and hl:
            structure = "HH/HL (yuqoriga struktura)"
        elif not hh and not hl:
            structure = "LH/LL (pastga struktura)"
        else:
            structure = "aralash (HH/LL yoki LH/HL) -- aniq struktura yo'q"
    return {"swing_highs": swing_highs, "swing_lows": swing_lows, "structure": structure}


def trend_summary(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> dict | None:
    """Yuqori timeframe uchun qisqa trend xulosasi."""
    if len(closes) < 55:
        return None
    sma20, sma50 = _sma(closes, 20), _sma(closes, 50)
    adx = _adx(highs, lows, closes, 14)
    if adx < 20:
        direction = "yon (trend yo'q)"
    else:
        direction = "yuqoriga" if sma20 > sma50 and closes[-1] > sma20 else "pastga" if sma20 < sma50 and closes[-1] < sma20 else "noaniq"
    return {"direction": direction, "adx": round(adx, 1), "rsi": round(_rsi(closes, 14), 1)}


def _clock_profile(value=None):
    return mt5_clock.normalize_profile(config.get_settings()["mt5_clock_profile"] if value is None else value)


def get_context(symbol: str, timeframe_name: str, n_bars: int = 100, *, clock_profile=None) -> dict | None:
    with MT5_LOCK:
        try:
            profile = _clock_profile(clock_profile)
        except mt5_clock.ClockProfileError:
            return None
        if bridge_client.enabled():
            try:
                if timeframe_name not in TIMEFRAME_SECONDS or type(n_bars) is not int or not 55 <= n_bars <= 1000:
                    return None
                result = bridge_client.request("context", {"symbol": symbol, "timeframe": timeframe_name,
                                                           "n_bars": n_bars, "clock_profile": profile})
                if result is None:
                    return None
                if not isinstance(result, dict) or result.get("symbol") != symbol or result.get("timeframe") != timeframe_name:
                    raise bridge_client.BridgeError("Host bridge konteksti so'ralgan bozorga mos emas")
                quote = _bridge_quote(result, profile)
                if quote is None or not all(math.isfinite(float(value)) for value in to_feature_vector(result)) or result["atr_14"] <= 0:
                    raise bridge_client.BridgeError("Host bridge konteksti yaroqsiz yoki eskirgan")
                return {**result, **quote}
            except (bridge_client.BridgeError, ValueError, TypeError, KeyError, OverflowError) as exc:
                bridge_client.record_error(str(exc))
                return None
        try:
            return _get_context(symbol, timeframe_name, n_bars, clock_profile=profile)
        except mt5_clock.ClockProfileError:
            return None


def _bridge_quote(result, clock_profile=None):
    if not isinstance(result, dict):
        return None
    profile = mt5_clock.normalize_profile(clock_profile)
    if mt5_clock.normalize_profile(result.get("clock_profile")) != profile:
        raise bridge_client.BridgeError("MT5 javobining vaqt sozlamalari so'rovga mos emas")
    if result.get("timestamp_basis", "utc") != "utc":
        raise bridge_client.BridgeError("MT5 javobining vaqti UTC sifatida tasdiqlanmagan")
    digits = result.get("digits")
    if type(digits) is not int or not 0 <= digits <= 12:
        return None
    # The host has already converted to UTC. Applying the server offset here
    # again would silently select a different quote/history interval.
    quote = _checked_quote(SimpleNamespace(bid=result["bid"], ask=result["ask"], time=result["tick_time"]),
                           SimpleNamespace(digits=digits))
    if quote is None:
        return None
    raw_time = float(result.get("raw_tick_time", quote["tick_time"]))
    if (not math.isfinite(raw_time) or
            abs(raw_time - quote["tick_time"] - mt5_clock.offset_seconds(profile)) > 0.001):
        raise bridge_client.BridgeError("MT5 javobining UTC va asl vaqti bir-biriga mos emas")
    return {**quote, "clock_profile": profile, "raw_tick_time": raw_time}


def _checked_quote(tick, info, *, clock_profile=None) -> dict | None:
    if tick is None or info is None:
        return None
    tick_msc = getattr(tick, "time_msc", 0)
    bid, ask = float(tick.bid), float(tick.ask)
    profile = mt5_clock.normalize_profile(clock_profile)
    raw_tick_time = float(tick_msc) / 1000 if tick_msc else float(tick.time)
    tick_time = raw_tick_time - mt5_clock.offset_seconds(profile)
    age = time.time() - tick_time
    if (not all(math.isfinite(x) for x in (bid, ask, tick_time)) or
            bid <= 0 or ask < bid or age < -5 or age > MAX_QUOTE_AGE_SECONDS):
        return None
    return {"bid": bid, "ask": ask, "digits": int(info.digits),
            "tick_time": tick_time, "quote_checked_at": time.time(),
            "raw_tick_time": raw_tick_time, "timestamp_basis": "utc", "clock_profile": profile}


def _initialization_error() -> MT5ReadError:
    # Call immediately after initialize() fails, while MT5_LOCK is held. Never
    # expose the native description: terminals can include sensitive values.
    try:
        failure = mt5.last_error()
    except Exception:
        failure = None
    code = failure[0] if isinstance(failure, (tuple, list)) and failure else None
    if type(code) is not int or not -(2 ** 31) <= code < 2 ** 31:
        code = None
    if code == -6:
        return MT5ReadError("MT5 hisobiga kirish tasdiqlanmadi (kod -6). Terminalda hisobingizga kiring va shu hisobning broker serverini tekshiring.")
    suffix = f"kod {code}" if code is not None else "kod noma'lum"
    return MT5ReadError(f"MT5 terminaliga ulanib bo'lmadi ({suffix}). Terminalni oching va ulanishini tekshiring.")


def get_quote(symbol: str, *, raise_on_error: bool = False, clock_profile=None) -> dict | None:
    """Read a fresh quote; optionally report safe connection/transport failures.

    Native quote freshness/price rejection continues to return None.
    """
    with MT5_LOCK:
        try:
            profile = _clock_profile(clock_profile)
        except mt5_clock.ClockProfileError as exc:
            if raise_on_error:
                raise MT5ReadError(str(exc)) from exc
            return None
        if bridge_client.enabled():
            try:
                result = bridge_client.request("quote", {"symbol": symbol, "clock_profile": profile}, timeout=3)
                if result is None:
                    return None
                quote = _bridge_quote(result, profile)
                if quote is None:
                    raise bridge_client.BridgeError("Host bridge narxi yaroqsiz yoki eskirgan")
                return quote
            except bridge_client.BridgeError as exc:
                bridge_client.record_error(str(exc))
                if raise_on_error:
                    raise MT5ReadError(str(exc)) from exc
                return None
            except (ValueError, TypeError, KeyError, OverflowError):
                message = "Host bridge narxi yaroqsiz yoki eskirgan"
                bridge_client.record_error(message)
                if raise_on_error:
                    raise MT5ReadError(message) from None
                return None
        if mt5 is None:
            if raise_on_error:
                raise MT5ReadError("MetaTrader5 moduli mavjud emas. Windows MT5 yordamchisi va uning Python kutubxonalarini tekshiring.")
            return None
        if not mt5.initialize():
            failure = _initialization_error()
            if raise_on_error:
                raise failure
            return None
        try:
            mt5_clock.validate_server(mt5, profile)
            if not mt5.symbol_select(symbol, True):
                return None
            quote = _checked_quote(mt5.symbol_info_tick(symbol), mt5.symbol_info(symbol), clock_profile=profile)
            mt5_clock.validate_server(mt5, profile)
            return quote
        except mt5_clock.ClockProfileError as exc:
            if raise_on_error:
                raise MT5ReadError(str(exc)) from exc
            return None
        finally:
            mt5.shutdown()


def _checked_closed_rates(rates, timeframe_name: str, tick_time: float):
    """Validate a snapshot containing the forming bar, then return closed bars.

    A recent quote alone cannot establish that the terminal's candle cache is
    current. Its forming bar must cover that quote's time. Completed bars may
    precede a weekend/holiday, so bounded session gaps are allowed rather than
    requiring the preceding bar to be exactly one timeframe old.
    """
    if rates is None or len(rates) < 56:  # 55 closed bars plus the forming bar
        return None
    seconds = TIMEFRAME_SECONDS[timeframe_name]
    times = rates["time"].astype(np.float64)
    if not np.all(np.isfinite(times)) or np.any(times <= 0):
        return None
    gaps = np.diff(times)
    if (np.any(gaps <= 0) or
            np.any(gaps > max(MAX_HISTORY_SESSION_GAP_SECONDS, seconds)) or
            times[-1] > tick_time + 5 or tick_time - times[-1] > seconds + 5):
        return None
    return rates[:-1]


def _get_context(symbol: str, timeframe_name: str, n_bars: int = 100, *, clock_profile=None) -> dict | None:
    """MT5'dan jonli narx + indikatorlarni oladi. MT5 mavjud bo'lmasa yoki
    symbol topilmasa None qaytaradi -- chaqiruvchi shunda eski (rasmdan
    taxmin qilingan) rejimga qaytadi."""
    if mt5 is None or timeframe_name not in TIMEFRAME_MAP:
        return None

    if not mt5.initialize():
        return None
    try:
        profile = _clock_profile(clock_profile)
        mt5_clock.validate_server(mt5, profile)
        if not mt5.symbol_select(symbol, True):
            return None

        # Read one snapshot including the forming bar to verify cache freshness;
        # the forming bar is excluded before computing any model indicators.
        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_MAP[timeframe_name], 0, n_bars + 1)
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        quote = _checked_quote(tick, info, clock_profile=profile)
        if quote is None:
            return None
        rates = _checked_closed_rates(mt5_clock.normalize_rates(rates, profile), timeframe_name, quote["tick_time"])
        if rates is None:
            return None

        closes = rates["close"].astype(np.float64)
        highs = rates["high"].astype(np.float64)
        lows = rates["low"].astype(np.float64)
        # Existing training data encoded broker-wall hours. Correct UTC cache
        # checks and exported timestamps without changing those model inputs.
        raw_bar_time = int(rates["time"][-1]) + mt5_clock.offset_seconds(profile)
        bar_hour = pd.to_datetime(raw_bar_time, unit="s").hour

        raw = compute_raw_indicators(highs, lows, closes, bar_hour=bar_hour)
        raw.update({
            "symbol": symbol,
            "timeframe": timeframe_name,
            **quote,
            "bar_time": int(rates["time"][-1]),
            "raw_bar_time": raw_bar_time,
            "feature_time_basis": "broker_wall",
            "server_utc_offset_hours": profile["offset_minutes"] / 60,
        })

        # Faqat Gemini uchun qo'shimcha kontekst (custom model xususiyatlariga kirmaydi --
        # train/serve parity buzilmasligi uchun compute_raw_indicators'ga qo'shilmagan)
        n = N_CONTEXT_CANDLES
        raw["candles"] = [
            [round(float(o), 5), round(float(h), 5), round(float(l), 5), round(float(c), 5), int(v)]
            for o, h, l, c, v in zip(rates["open"][-n:], rates["high"][-n:], rates["low"][-n:],
                                     rates["close"][-n:], rates["tick_volume"][-n:])
        ]
        raw["swings"] = find_swings(highs, lows)
        higher = {}
        for tf_name in HIGHER_TIMEFRAMES.get(timeframe_name, []):
            hr = mt5.copy_rates_from_pos(symbol, TIMEFRAME_MAP[tf_name], 0, n_bars + 1)
            hr = _checked_closed_rates(mt5_clock.normalize_rates(hr, profile), tf_name, quote["tick_time"])
            if hr is not None:
                summary = trend_summary(hr["high"].astype(np.float64), hr["low"].astype(np.float64),
                                        hr["close"].astype(np.float64))
                if summary:
                    higher[tf_name] = summary
        raw["higher_tf"] = higher
        mt5_clock.validate_server(mt5, profile)
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

    digits = context.get("digits", 5)
    return {
        "entry_price": round(entry, digits),
        "tp_price": round(tp, digits),
        "sl_price": round(sl, digits),
    }
