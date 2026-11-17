"""Explicit MT5 source-clock conversion and a read-only verification command.

Positive offsets mean native live tick/bar timestamps are ahead of UTC. History
query/result conventions must be verified separately; no offset is inferred.
"""
import argparse
import datetime as dt
import importlib
import json
import math
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import numpy as np

UTC = dt.timezone.utc
EPOCH = dt.datetime(1970, 1, 1, tzinfo=UTC)
MAX_TICKS = 150_000
HISTORY_MODES = {"utc", "server", "utc_query_server_result", "unverified"}
DEFAULT_PROFILE = {"version": 1, "server": "", "offset_minutes": 0, "history_mode": "utc"}
TICK_DTYPE = [("time_msc", "i8"), ("bid", "f8"), ("ask", "f8")]


class ClockProfileError(ValueError):
    def __init__(self, message, sdk_error_code=None):
        super().__init__(message)
        self.sdk_error_code = sdk_error_code if type(sdk_error_code) is int and -(2 ** 31) <= sdk_error_code < 2 ** 31 else None
        self.checks = {}


def normalize_profile(value=None):
    if value is None:
        return dict(DEFAULT_PROFILE)
    if not isinstance(value, dict) or set(value) - set(DEFAULT_PROFILE):
        raise ClockProfileError("MT5 vaqt sozlamasi noto'g'ri")
    profile = {**DEFAULT_PROFILE, **value}
    if type(profile["version"]) is not int or profile["version"] != 1:
        raise ClockProfileError("MT5 vaqt sozlamasi versiyasi qo'llanmaydi")
    offset = profile["offset_minutes"]
    if type(offset) is not int or not -840 <= offset <= 840:
        raise ClockProfileError("MT5 vaqt farqi -840 va 840 orasidagi butun minut bo'lishi kerak")
    server = profile["server"]
    if (not isinstance(server, str) or len(server) > 128 or server != server.strip() or
            any(not 32 <= ord(char) <= 126 for char in server) or (offset and not server)):
        raise ClockProfileError("MT5 vaqt sozlamasida aniq broker serveri talab qilinadi")
    if offset and "history_mode" not in value:
        profile["history_mode"] = "unverified"
    if not isinstance(profile["history_mode"], str) or profile["history_mode"] not in HISTORY_MODES:
        raise ClockProfileError("MT5 tarixining vaqt usuli noto'g'ri")
    return profile


def validate_server(api, profile):
    profile = normalize_profile(profile)
    if not profile["server"]:
        return profile
    try:
        account = api.account_info()
        server = getattr(account, "server", None)
    except Exception:
        raise ClockProfileError("MT5 broker serverini tekshirib bo'lmadi") from None
    if not isinstance(server, str) or server != profile["server"]:
        raise ClockProfileError("MT5 broker serveri vaqt sozlamasiga mos emas")
    return profile


def offset_seconds(profile):
    return normalize_profile(profile)["offset_minutes"] * 60


def normalize_rates(rates, profile):
    shift = offset_seconds(profile)
    if rates is None:
        return None
    if not isinstance(rates, np.ndarray) or rates.ndim != 1 or not rates.dtype.names or "time" not in rates.dtype.names:
        raise ClockProfileError("MT5 sham vaqtlari noto'g'ri")
    result = rates.copy()
    values = result["time"]
    if values.ndim != 1 or values.dtype.kind not in "iu" or np.any(values <= 0) or np.any(values > np.iinfo(np.int64).max):
        raise ClockProfileError("MT5 sham vaqtlari noto'g'ri")
    normalized = values.astype(np.int64) - shift
    if np.any(normalized <= 0):
        raise ClockProfileError("MT5 sham vaqtlari noto'g'ri")
    result["time"] = normalized
    return result


def _utc_microseconds(value):
    if not isinstance(value, dt.datetime) or value.tzinfo is None or value.utcoffset() != dt.timedelta(0):
        raise ClockProfileError("MT5 tarix so'rovi aniq UTC vaqtini talab qiladi")
    delta = value - EPOCH
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def _bounds(start, end):
    start_us, end_us = _utc_microseconds(start), _utc_microseconds(end)
    if start_us < 0 or not 0 < end_us - start_us <= 300_000_000:
        raise ClockProfileError("MT5 tarix oralig'i 5 daqiqadan oshmasligi kerak")
    return start_us, end_us


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _price(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ClockProfileError("MT5 tarix narxlari noto'g'ri")
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ClockProfileError("MT5 tarix narxlari noto'g'ri") from None
    if not math.isfinite(value):
        raise ClockProfileError("MT5 tarix narxlari noto'g'ri")
    return value


def _normalize_ticks(rows, start_us, end_us, result_shift):
    if rows is None:
        return None
    if not isinstance(rows, (list, tuple, np.ndarray)) or len(rows) > MAX_TICKS:
        raise ClockProfileError("MT5 tarix javobi hajmi yoki formati noto'g'ri")
    checked = []
    try:
        for row in rows:
            timestamp, bid, ask = row["time_msc"], _price(row["bid"]), _price(row["ask"])
            if not _integer(timestamp) or not 0 < timestamp <= np.iinfo(np.int64).max:
                raise ClockProfileError("MT5 tick vaqti noto'g'ri")
            timestamp = int(timestamp) - result_shift * 1000
            if timestamp <= 0 or timestamp > np.iinfo(np.int64).max or bid <= 0 or ask < bid:
                raise ClockProfileError("MT5 tick vaqti yoki narxlari noto'g'ri")
            # The native API can round datetime bounds to seconds. Keep only
            # ticks inside the exact requested interval, including its end.
            if start_us <= timestamp * 1000 <= end_us:
                checked.append((timestamp, bid, ask))
    except ClockProfileError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        raise ClockProfileError("MT5 tarix javobi noto'g'ri") from None
    return np.array(checked, dtype=TICK_DTYPE)


def copy_ticks_utc(api, symbol, start, end, flags, profile):
    profile = normalize_profile(profile)
    if profile["history_mode"] == "unverified":
        raise ClockProfileError("MT5 tarixining vaqt usuli hali tasdiqlanmagan")
    start_us, end_us = _bounds(start, end)
    validate_server(api, profile)
    shift = offset_seconds(profile)
    query_shift = shift if profile["history_mode"] == "server" else 0
    result_shift = 0 if profile["history_mode"] == "utc" else shift
    try:
        rows = api.copy_ticks_range(symbol, start + dt.timedelta(seconds=query_shift),
                                    end + dt.timedelta(seconds=query_shift), flags)
    except Exception:
        raise ClockProfileError("MT5 tarixini o'qib bo'lmadi") from None
    validate_server(api, profile)
    return _normalize_ticks(rows, start_us, end_us, result_shift)


class NativeMT5Proxy:
    def __init__(self, api, profile):
        self.api = api
        self.profile = normalize_profile(profile)
        self.COPY_TICKS_INFO = getattr(api, "COPY_TICKS_INFO", 1)

    def initialize(self):
        try:
            initialized = self.api.initialize()
        except Exception:
            code = _sdk_code(self.api)
            self._failed_initialization_cleanup()
            raise ClockProfileError("MT5 terminaliga ulanib bo'lmadi", code) from None
        if not initialized:
            self._failed_initialization_cleanup()
            return False
        try:
            validate_server(self.api, self.profile)
        except ClockProfileError:
            self._failed_initialization_cleanup()
            raise
        return True

    def _failed_initialization_cleanup(self):
        try:
            self.api.shutdown()
        except Exception:
            pass

    def shutdown(self):
        return self.api.shutdown()

    def copy_ticks_range(self, symbol, start, end, flags):
        return copy_ticks_utc(self.api, symbol, start, end, flags, self.profile)


def _sdk_code(api):
    try:
        value = api.last_error()[0]
        return value if type(value) is int and -(2 ** 31) <= value < 2 ** 31 else None
    except Exception:
        return None


def _quote_fingerprint(tick, shift, now):
    if tick is None:
        raise ClockProfileError("MT5 yangi narx bermadi")
    seconds, milliseconds = getattr(tick, "time", None), getattr(tick, "time_msc", None)
    if not _integer(seconds) or not _integer(milliseconds) or milliseconds // 1000 != seconds:
        raise ClockProfileError("MT5 tick sekund va millisekund vaqtlari mos emas")
    bid, ask = _price(getattr(tick, "bid", None)), _price(getattr(tick, "ask", None))
    corrected = int(milliseconds) - shift * 1000
    age = now - corrected / 1000
    if corrected <= 0 or bid <= 0 or ask < bid or not -5 <= age <= 60:
        raise ClockProfileError("MT5 narxi berilgan vaqt farqi bilan yangi emas")
    return corrected, bid, ask


def _observed_time(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value) if abs(int(value)) <= np.iinfo(np.int64).max else None
    value = float(value)
    return round(value, 3) if math.isfinite(value) and abs(value) <= np.iinfo(np.int64).max else None


def _quote_observation(tick, shift, now):
    seconds = _observed_time(getattr(tick, "time", None))
    milliseconds = _observed_time(getattr(tick, "time_msc", None))
    age = _observed_time(now - milliseconds / 1000 + shift) if milliseconds is not None else None
    return {"raw_time": seconds, "raw_time_msc": milliseconds,
            "os_epoch": _observed_time(now), "corrected_age": age}


def _history_observation(rows):
    try:
        count = len(rows) if isinstance(rows, (list, tuple, np.ndarray)) else None
        last = _observed_time(rows[-1]["time_msc"]) if count else None
    except (KeyError, TypeError, ValueError, IndexError):
        count, last = None, None
    return {"count": count, "last_raw_ms": last}


def _bars_current(api, symbol, profile, now, checks=None):
    for name, seconds in (("M1", 60), ("M15", 900), ("H1", 3600)):
        try:
            rates = api.copy_rates_from_pos(symbol, getattr(api, "TIMEFRAME_" + name), 0, 1)
        except Exception:
            raise ClockProfileError("MT5 sham tarixini o'qib bo'lmadi") from None
        if checks is not None:
            try:
                raw = _observed_time(rates["time"][-1]) if rates is not None and len(rates) else None
            except (KeyError, TypeError, ValueError, IndexError):
                raw = None
            checks.setdefault("bar_observations", {})[name] = {
                "raw_time": raw,
                "corrected_age": _observed_time(now - raw + offset_seconds(profile)) if raw is not None else None,
            }
        normalized = normalize_rates(rates, profile)
        if normalized is None or len(normalized) != 1 or not -5 <= now - int(normalized["time"][0]) <= seconds + 5:
            raise ClockProfileError("MT5 joriy shamlari berilgan vaqt farqiga mos emas")


def _fingerprints(rows):
    return set() if rows is None else {(int(row["time_msc"]), float(row["bid"]), float(row["ask"])) for row in rows}


def _diagnose(api, server, offset, terminal_path, symbol="XAUUSD"):
    profile = normalize_profile({"server": server, "offset_minutes": offset, "history_mode": "unverified"})
    if not profile["server"]:
        raise ClockProfileError("Tekshirish uchun aniq broker serveri talab qilinadi")
    initialized = False
    checks = {"advancing_quotes": 0}
    try:
        try:
            initialized = bool(api.initialize(str(terminal_path), timeout=5000))
        except Exception:
            raise ClockProfileError("MT5 terminaliga ulanib bo'lmadi", _sdk_code(api)) from None
        if not initialized:
            raise ClockProfileError("MT5 terminaliga ulanib bo'lmadi", _sdk_code(api))
        validate_server(api, profile)
        try:
            selected = api.symbol_select(symbol, True)
        except Exception:
            raise ClockProfileError("MT5 vositasini tanlab bo'lmadi", _sdk_code(api)) from None
        if not selected:
            raise ClockProfileError("MT5 vositasini tanlab bo'lmadi", _sdk_code(api))
        shift = offset_seconds(profile)
        deadline = time.monotonic() + 10
        captured = []
        while time.monotonic() < deadline and len(captured) < 3:
            try:
                tick = api.symbol_info_tick(symbol)
            except Exception:
                raise ClockProfileError("MT5 narxini o'qib bo'lmadi", _sdk_code(api)) from None
            now = time.time()
            checks["last_quote"] = _quote_observation(tick, shift, now)
            fingerprint = _quote_fingerprint(tick, shift, now)
            if captured and fingerprint[0] < captured[-1][0]:
                raise ClockProfileError("MT5 tick vaqti orqaga qaytdi")
            if not captured or fingerprint[0] > captured[-1][0]:
                captured.append(fingerprint)
                checks["advancing_quotes"] = len(captured)
            if len(captured) < 3:
                time.sleep(.2)
        if len(captured) < 3:
            raise ClockProfileError("MT5 narx vaqti yetarlicha yangilanmadi")
        _bars_current(api, symbol, profile, time.time(), checks)
        start = EPOCH + dt.timedelta(milliseconds=captured[0][0] - 1000)
        end = EPOCH + dt.timedelta(milliseconds=captured[-1][0] + 1000)
        start_us, end_us = _bounds(start, end)
        reference = set(captured)
        deadline = time.monotonic() + 12
        counts, winner, confirmations = {}, None, 0
        while time.monotonic() < deadline:
            windows = {}
            for key, query_shift in (("utc", 0), ("server", shift)):
                if key == "server" and not shift:
                    windows[key] = windows["utc"]
                    continue
                try:
                    windows[key] = api.copy_ticks_range(symbol, start + dt.timedelta(seconds=query_shift),
                                                        end + dt.timedelta(seconds=query_shift),
                                                        getattr(api, "COPY_TICKS_ALL", api.COPY_TICKS_INFO))
                except Exception:
                    raise ClockProfileError("MT5 tarixini tekshirib bo'lmadi", _sdk_code(api)) from None
                checks.setdefault("history_windows", {})[key] = _history_observation(windows[key])
            modes = ("utc",) if not shift else ("utc", "server", "utc_query_server_result")
            counts = {}
            checks["history_matches"] = counts
            for mode in modes:
                rows = windows["server" if mode == "server" else "utc"]
                matches = reference & _fingerprints(_normalize_ticks(rows, start_us, end_us, 0 if mode == "utc" else shift))
                counts[mode] = len({item[0] for item in matches})
            qualified = [mode for mode, count in counts.items() if count >= 2]
            if len(qualified) > 1:
                raise ClockProfileError("MT5 tarixining vaqt usuli noaniq; sozlama qo'llanmadi")
            selected = qualified[0] if qualified else None
            confirmations = confirmations + 1 if selected and selected == winner else 1 if selected else 0
            winner = selected
            if winner and confirmations >= 2:
                break
            time.sleep(.25)
        if not winner or confirmations < 2:
            raise ClockProfileError("MT5 tarixining vaqt usuli tasdiqlanmadi")
        _bars_current(api, symbol, profile, time.time(), checks)
        validate_server(api, profile)
        checks["forming_bars"] = ["M1", "M15", "H1"]
        return {"verified": True, "applied": False,
                "profile": {**profile, "history_mode": winner},
                "checks": checks}
    except ClockProfileError as exc:
        exc.checks = checks
        raise
    except Exception:
        failure = ClockProfileError("MT5 vaqtini tekshirish tugamadi; sozlama qo'llanmadi", _sdk_code(api))
        failure.checks = checks
        raise failure from None
    finally:
        if initialized:
            try:
                api.shutdown()
            except Exception:
                pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def _apply_profile(base_url, profile):
    parsed = urlparse(base_url)
    try:
        port = parsed.port
    except ValueError:
        raise ClockProfileError("Sozlama manzili mahalliy HTTP manzil bo'lishi kerak") from None
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.username or parsed.password or
            parsed.path not in ("", "/") or parsed.query or parsed.fragment or port is None or not 1 <= port <= 65535):
        raise ClockProfileError("Sozlama manzili http://127.0.0.1:port bo'lishi kerak")
    url = base_url.rstrip("/") + "/control/settings"
    request = Request(url,
                      data=json.dumps({"mt5_clock_profile": normalize_profile(profile)}, allow_nan=False).encode("utf-8"),
                      headers={"Content-Type": "application/json", "X-Grid-Bot-Request": "1"}, method="POST")
    try:
        opener = build_opener(ProxyHandler({}), _NoRedirect())
        for operation in (request, Request(url, method="GET")):
            with opener.open(operation, timeout=5) as response:
                raw = response.read(65537)
            if len(raw) > 65536:
                raise ClockProfileError("Sozlama serverining javobi noto'g'ri")
            result = json.loads(raw)
            if (not isinstance(result, dict) or not isinstance(result.get("mt5_clock_profile"), dict) or
                    normalize_profile(result["mt5_clock_profile"]) != normalize_profile(profile)):
                raise ClockProfileError("Sozlama saqlangani tasdiqlanmadi")
    except (HTTPError, URLError, OSError, ValueError, TypeError):
        raise ClockProfileError("Tasdiqlangan vaqt sozlamasini saqlab bo'lmadi") from None


def main(argv=None):
    class Parser(argparse.ArgumentParser):
        def error(self, message):
            raise ClockProfileError("Buyruq argumentlari noto'g'ri")

    parser = Parser(description=__doc__)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--server", required=True)
    parser.add_argument("--offset-minutes", required=True, type=int)
    parser.add_argument("--terminal-path", required=True)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--apply-url")
    report = {"verified": False, "applied": False}
    try:
        args = parser.parse_args(argv)
        terminal = Path(args.terminal_path)
        if (not args.diagnose or not terminal.is_absolute() or not terminal.is_file() or
                terminal.name.lower() not in ("terminal64.exe", "terminal.exe") or
                not re.fullmatch(r"[A-Za-z0-9._#-]{1,32}", args.symbol)):
            raise ClockProfileError("Aniq MT5 terminal yo'li va vositasi talab qilinadi")
        normalize_profile({"server": args.server, "offset_minutes": args.offset_minutes})
        try:
            api = importlib.import_module("MetaTrader5")
        except ImportError:
            raise ClockProfileError("Windows MetaTrader5 moduli mavjud emas") from None
        report = _diagnose(api, args.server, args.offset_minutes, terminal, args.symbol)
        if args.apply_url:
            _apply_profile(args.apply_url, report["profile"])
            report["applied"] = True
        print(json.dumps(report, allow_nan=False, separators=(",", ":")))
        return 0
    except ClockProfileError as exc:
        report["error"] = str(exc)
        if exc.checks:
            report["checks"] = exc.checks
        if exc.sdk_error_code is not None:
            report["sdk_error_code"] = exc.sdk_error_code
    except Exception:
        report["error"] = "MT5 vaqtini tekshirish tugamadi; sozlama qo'llanmadi"
    print(json.dumps(report, allow_nan=False, separators=(",", ":")))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
