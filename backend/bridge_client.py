"""Authenticated JSON file-spool RPC to the fixed Windows host bridge.

No sockets, pickle, arbitrary file paths or automatic retries are used. A timed
out request's id is never reused, so late replies cannot satisfy later calls.
"""
import datetime as dt
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time
import uuid

import numpy as np

MAX_BYTES = 20 * 1024 * 1024
MAX_PENDING = 64
MAX_AGE_SECONDS = 30
OPS = {"health", "context", "quote", "ticks", "runtime", "capabilities", "service", "logs",
       "ollama_tags", "ollama_chat", "stop"}


class BridgeError(RuntimeError):
    pass


def canonical(message):
    return json.dumps(message, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sign(message, token):
    return hmac.new(token, canonical({key: value for key, value in message.items() if key != "hmac"}), hashlib.sha256).hexdigest()


def _no_link(path):
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if path.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise BridgeError("Host bridge fayl havolasi taqiqlangan")
    return path


class BridgeClient:
    def __init__(self, directory, token_file=None):
        # The configured mount root itself may be an existing workspace alias;
        # all spool children must be real files/directories below its target.
        self.directory = Path(directory).resolve()
        self.token_file = Path(token_file).absolute() if token_file else self.directory / "token"
        self._lock = threading.Lock()
        self.last_error = None
        self.last_contact = None

    def _error(self, message):
        with self._lock:
            self.last_error = str(message)

    def status(self):
        with self._lock:
            return {"enabled": True, "transport": "host_bridge", "last_error": self.last_error,
                    "last_contact": self.last_contact}

    def _folder(self, name):
        if not self.directory.is_dir():
            raise BridgeError("Host bridge papkasi mavjud emas")
        path = _no_link(self.directory / name)
        path.mkdir(exist_ok=True)
        if path.resolve().parent != self.directory:
            raise BridgeError("Host bridge yo'li papkadan tashqariga chiqdi")
        return path

    def _token(self):
        path = _no_link(self.token_file)
        if path.stat().st_size > 128:
            raise BridgeError("Host bridge token formati noto'g'ri")
        token = path.read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{64}", token):
            raise BridgeError("Host bridge token formati noto'g'ri")
        return bytes.fromhex(token)

    def request(self, op, args=None, timeout=None):
        if op not in OPS or not isinstance(args if args is not None else {}, dict):
            raise BridgeError("Host bridge operatsiyasi noto'g'ri")
        args = {} if args is None else args
        if timeout is None:
            if op == "ollama_chat":
                inference_timeout = args.get("timeout", 120)
                if type(inference_timeout) not in (int, float) or not math.isfinite(inference_timeout) or not 5 <= inference_timeout <= 600:
                    raise BridgeError("Host bridge inference kutish muddati noto'g'ri")
                timeout = inference_timeout + 5
            else:
                timeout = 15.
        maximum = 630. if op == "ollama_chat" else 30.
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= maximum:
            raise BridgeError("Host bridge kutish muddati noto'g'ri")
        request_path = response_path = temporary = None
        try:
            token = self._token()
            requests, responses = self._folder("requests"), self._folder("responses")
            if sum(1 for _ in requests.glob("*.json")) >= MAX_PENDING:
                raise BridgeError("Host bridge so'rov navbati band")
            identifier = uuid.uuid4().hex
            created = time.time()
            message = {"id": identifier, "created_at": created, "op": op, "args": args}
            message["hmac"] = sign(message, token)
            encoded = canonical(message)
            if len(encoded) > MAX_BYTES:
                raise BridgeError("Host bridge so'rovi juda katta")
            request_path, response_path = requests / f"{identifier}.json", responses / f"{identifier}.json"
            temporary = requests / f"{identifier}.tmp"
            with temporary.open("xb") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, request_path)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if response_path.exists() or response_path.is_symlink():
                    _no_link(response_path)
                    if response_path.stat().st_size > MAX_BYTES:
                        raise BridgeError("Host bridge javobi juda katta")
                    with response_path.open("rb") as source:
                        payload = source.read(MAX_BYTES + 1)
                    if len(payload) > MAX_BYTES:
                        raise BridgeError("Host bridge javobi juda katta")
                    response = json.loads(payload)
                    if not isinstance(response, dict) or set(response) != {"id", "created_at", "ok", "result", "error", "hmac"}:
                        raise BridgeError("Host bridge javobi noto'g'ri")
                    signature = response["hmac"]
                    if not isinstance(signature, str) or not hmac.compare_digest(signature, sign(response, token)):
                        raise BridgeError("Host bridge javobi imzosi mos emas")
                    timestamp = response["created_at"]
                    if (response["id"] != identifier or type(response["ok"]) is not bool or
                            isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or
                            timestamp < created or time.time() - timestamp > MAX_AGE_SECONDS or timestamp > time.time() + 5):
                        raise BridgeError("Host bridge javobi eskirgan yoki boshqa so'rovga tegishli")
                    if not response["ok"]:
                        error = response.get("error")
                        raise BridgeError("Host bridge: " + (error[:1000] if isinstance(error, str) else "operatsiya bajarilmadi"))
                    with self._lock:
                        self.last_error, self.last_contact = None, time.time()
                    return response["result"]
                time.sleep(min(.025, max(0, deadline - time.monotonic())))
            raise BridgeError("Host bridge javob bermadi (offline yoki band)")
        except BridgeError as exc:
            self._error(exc)
            raise
        except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as exc:
            message = "Host bridge fayllari yoki JSON javobini o'qib bo'lmadi"
            self._error(message)
            raise BridgeError(message) from exc
        finally:
            for path in (temporary, request_path, response_path):
                if path is not None:
                    try:
                        # Never follow a response link while cleaning up.
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass


_CLIENT_LOCK = threading.Lock()
_client = None
_client_key = None


def enabled():
    return bool(os.environ.get("GRID_BOT_BRIDGE_DIR", "").strip())


def get_client():
    global _client, _client_key
    directory = os.environ.get("GRID_BOT_BRIDGE_DIR", "").strip()
    if not directory:
        raise BridgeError("Host bridge sozlanmagan")
    token_file = os.environ.get("GRID_BOT_BRIDGE_TOKEN_FILE", "").strip() or None
    key = (directory, token_file)
    with _CLIENT_LOCK:
        if _client_key != key:
            try:
                _client = BridgeClient(directory, token_file)
            except (OSError, ValueError, TypeError) as exc:
                raise BridgeError("Host bridge konfiguratsiyasidagi yo'l noto'g'ri") from exc
            _client_key = key
        return _client


def request(op, args=None, timeout=None):
    return get_client().request(op, args, timeout)


def record_error(message):
    if enabled():
        try:
            get_client()._error(message)
        except BridgeError:
            pass


def status():
    if not enabled():
        return {"enabled": False, "transport": "native", "last_error": None, "last_contact": None}
    try:
        return get_client().status()
    except BridgeError as exc:
        return {"enabled": True, "transport": "host_bridge", "last_error": str(exc), "last_contact": None}


class MT5Proxy:
    """Only the read-only subset used by tick evaluation; never order actions."""
    COPY_TICKS_INFO = 1

    def initialize(self):
        result = request("health", {}, timeout=3)
        return bool(isinstance(result, dict) and (result.get("mt5_ready") is True or result.get("status") == "ready"))

    def shutdown(self):
        # Each host operation owns its own native MT5 session/lock.
        return None

    def copy_ticks_range(self, symbol, start, end, flags):
        def epoch_ms(value):
            if not isinstance(value, dt.datetime) or value.tzinfo is None:
                raise BridgeError("Tick so'rovi UTC vaqtini ko'rsatishi kerak")
            return int(value.timestamp() * 1000)
        start_ms, end_ms = epoch_ms(start), epoch_ms(end)
        if end_ms <= start_ms or end_ms - start_ms > 300_000:
            raise BridgeError("Tick oralig'i 5 daqiqadan oshmasligi kerak")
        rows = request("ticks", {"symbol": symbol, "start_ms": start_ms, "end_ms": end_ms}, timeout=10)
        if rows is None:
            return None
        if not isinstance(rows, list) or len(rows) > 250_000:
            raise BridgeError("Host bridge tick javobi noto'g'ri")
        checked = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"time_msc", "bid", "ask"}:
                raise BridgeError("Host bridge tick formati noto'g'ri")
            timestamp, bid, ask = row["time_msc"], row["bid"], row["ask"]
            if (type(timestamp) is not int or not start_ms <= timestamp <= end_ms or
                    type(bid) not in (int, float) or type(ask) not in (int, float) or
                    not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask < bid):
                raise BridgeError("Host bridge tick qiymatlari yaroqsiz")
            checked.append((timestamp, bid, ask))
        return np.array(checked, dtype=[("time_msc", "i8"), ("bid", "f8"), ("ask", "f8")])
