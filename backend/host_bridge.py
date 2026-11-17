"""Authenticated file-spool bridge for the Windows MT5/Ollama host.

No listening socket, trading/order API, arbitrary command or caller-supplied
filesystem path is exposed. Run with --directory <shared-spool>, --check/--stop.
"""
import argparse
import base64
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
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# The Windows helper must never recurse into its own container transport.
if __name__ == "__main__":
    os.environ.pop("GRID_BOT_BRIDGE_DIR", None)
    os.environ.pop("GRID_BOT_BRIDGE_TOKEN_FILE", None)

import config
import control_runtime
import mt5_context
import psutil

MAX_BYTES = 20 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ID_PATTERN = re.compile(r"[0-9a-f]{32}")
_SCHEMA = {"type": "object", "properties": {
    "signal": {"type": "string", "enum": ["Buy", "Sell", "Hold"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "trend": {"type": "string"}, "reasoning": {"type": "string"}},
    "required": ["signal", "confidence", "trend", "reasoning"]}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def signed(body, token):
    return {**body, "hmac": hmac.new(token, canonical(body), hashlib.sha256).hexdigest()}


def authenticated(message, token):
    if not isinstance(message, dict) or not isinstance(message.get("hmac"), str):
        return False
    body = {key: value for key, value in message.items() if key != "hmac"}
    return hmac.compare_digest(message["hmac"], hmac.new(token, canonical(body), hashlib.sha256).hexdigest())


def checked_child(base, path):
    path = Path(path).absolute()
    relative = path.relative_to(base)
    current = base
    for component in relative.parts:
        current = current / component
        if current.exists() or current.is_symlink():
            info = current.lstat()
            if current.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                raise ValueError("Bridge links are not allowed")
    if path.resolve() != path:
        raise ValueError("Bridge path escaped its directory")
    return path


def read_json(base, path):
    with checked_child(base, path).open("rb") as source:
        data = source.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("Bridge message is too large")
    return json.loads(data)


def write_json(base, path, value):
    payload = canonical(value)
    if len(payload) > MAX_BYTES:
        raise ValueError("Bridge response is too large")
    path = checked_child(base, path)
    temporary = checked_child(base, path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp"))
    try:
        with temporary.open("xb") as output:
            output.write(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def symbol(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z0-9._#-]{1,32}", value):
        raise ValueError("Invalid symbol")
    return value


def exact_args(args, required=(), optional=()):
    if not isinstance(args, dict) or not set(required).issubset(args) or set(args) - set(required) - set(optional):
        raise ValueError("Invalid operation arguments")


def ollama_request(path, payload=None, timeout=2):
    parsed = urlparse(config.OLLAMA_URL)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or
            parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("Host Ollama must use a loopback URL")
    request = Request(config.OLLAMA_URL.rstrip("/") + path,
        data=canonical(payload) if payload is not None else None,
        headers={"Content-Type": "application/json"}, method="POST" if payload is not None else "GET")
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Ollama response is too large")
    return json.loads(raw)


def checked_chat(args):
    exact_args(args, ("payload", "timeout"))
    timeout = args["timeout"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 5 <= timeout <= 600:
        raise ValueError("Invalid Ollama timeout")
    payload = args["payload"]
    exact_args(payload, ("model", "messages", "stream", "think", "format", "options", "keep_alive"))
    if not isinstance(payload["model"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,120}", payload["model"]):
        raise ValueError("Invalid Ollama model")
    messages = payload["messages"]
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("One image message is required")
    message = messages[0]
    exact_args(message, ("role", "content", "images"))
    if message["role"] != "user" or not isinstance(message["content"], str) or len(message["content"].encode("utf-8")) > 128 * 1024:
        raise ValueError("Invalid Ollama prompt")
    images = message["images"]
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise ValueError("One encoded image is required")
    if len(base64.b64decode(images[0], validate=True)) > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large")
    if (payload["stream"] is not False or payload["think"] is not False or payload["keep_alive"] != 0 or
            payload["options"] != {"temperature": 0, "num_ctx": 4096, "num_predict": 400} or payload["format"] != _SCHEMA):
        raise ValueError("Only bounded non-streaming image inference is allowed")
    return payload, timeout


class HostBridge:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        if not self.directory.is_dir():
            raise ValueError("Bridge directory must be created by setup")
        token_path = checked_child(self.directory, self.directory / "token")
        token_text = token_path.read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{64}", token_text):
            raise ValueError("Bridge token must contain 64 hexadecimal characters")
        self.token = bytes.fromhex(token_text)
        for name in ("requests", "responses", "working", "seen"):
            checked_child(self.directory, self.directory / name).mkdir(exist_ok=True)
        self.stopping = threading.Event()
        self.slots = threading.BoundedSemaphore(4)
        self.qwen_slot = threading.BoundedSemaphore(1)

    def dispatch(self, operation, args):
        if operation in ("health", "runtime", "capabilities", "ollama_tags", "stop"):
            exact_args(args)
            if operation == "health":
                return {"status": "ok", "protocol": 1, "mt5_ready": mt5_context.mt5 is not None}
            if operation == "runtime":
                return control_runtime.runtime_snapshot()
            if operation == "capabilities":
                caps = control_runtime.service_capabilities()
                return {name: caps[name] for name in ("ollama", "mt5")}
            if operation == "ollama_tags":
                return ollama_request("/api/tags")
            self.stopping.set()
            return {"status": "stopping"}
        if operation == "quote":
            exact_args(args, ("symbol",))
            return mt5_context.get_quote(symbol(args["symbol"]))
        if operation == "context":
            exact_args(args, ("symbol", "timeframe"), ("n_bars",))
            count = args.get("n_bars", 100)
            if args["timeframe"] not in mt5_context.TIMEFRAME_SECONDS or type(count) is not int or not 55 <= count <= 1000:
                raise ValueError("Invalid timeframe or candle count")
            return mt5_context.get_context(symbol(args["symbol"]), args["timeframe"], count)
        if operation == "ticks":
            exact_args(args, ("symbol", "start_ms", "end_ms"))
            selected = symbol(args["symbol"])
            start, end = args["start_ms"], args["end_ms"]
            if type(start) is not int or type(end) is not int or start < 0 or not 0 < end - start <= 300000:
                raise ValueError("Tick interval must be at most five minutes")
            start_date = dt.datetime.fromtimestamp(start / 1000, dt.timezone.utc)
            end_date = dt.datetime.fromtimestamp(end / 1000, dt.timezone.utc)
            with mt5_context.MT5_LOCK:
                mt5 = mt5_context.mt5
                if mt5 is None or not mt5.initialize():
                    return None
                try:
                    if not mt5.symbol_select(selected, True):
                        return None
                    ticks = mt5.copy_ticks_range(selected, start_date, end_date, mt5.COPY_TICKS_INFO)
                    if ticks is None:
                        return None
                    if len(ticks) > 150000:
                        raise ValueError("Tick response exceeds bounded capacity")
                    return [{"time_msc": int(tick["time_msc"]), "bid": float(tick["bid"]), "ask": float(tick["ask"])} for tick in ticks]
                finally:
                    mt5.shutdown()
        if operation == "service":
            exact_args(args, ("service", "action"))
            allowed = {"ollama": ("start", "stop", "restart", "check"), "mt5": ("check",)}
            if args["service"] not in allowed or args["action"] not in allowed[args["service"]]:
                raise ValueError("Unsupported host service action")
            return control_runtime.service_action(args["service"], args["action"])
        if operation == "logs":
            exact_args(args, ("source",), ("lines",))
            return control_runtime.read_logs(args["source"], args.get("lines", 100))
        if operation == "ollama_chat":
            payload, timeout = checked_chat(args)
            if not self.qwen_slot.acquire(blocking=False):
                raise ValueError("Host image inference is busy; retry later")
            try:
                return ollama_request("/api/chat", payload, timeout)
            finally:
                self.qwen_slot.release()
        raise ValueError("Unsupported bridge operation")

    def process_file(self, path):
        identifier = path.stem
        if not ID_PATTERN.fullmatch(identifier) or path.suffix != ".json":
            return False
        claimed = self.directory / "working" / path.name
        try:
            checked_child(self.directory, path)
            checked_child(self.directory, claimed)
            os.replace(path, claimed)  # claim before reading or executing
            request = read_json(self.directory, claimed)
            created = request.get("created_at") if isinstance(request, dict) else None
            if (not authenticated(request, self.token) or set(request) != {"id", "created_at", "op", "args", "hmac"} or
                    request["id"] != identifier or type(created) not in (int, float) or not math.isfinite(created) or
                    not -5 <= time.time() - created <= 30):
                return False
            marker = checked_child(self.directory, self.directory / "seen" / identifier)
            try:
                with marker.open("x"):
                    pass
            except FileExistsError:
                return False
            try:
                result = self.dispatch(request["op"], request["args"])
                body = {"id": identifier, "created_at": time.time(), "ok": True, "result": result, "error": None}
                canonical(body)
            except Exception as exc:
                message = str(exc) if isinstance(exc, ValueError) else "Host operation failed: " + type(exc).__name__
                body = {"id": identifier, "created_at": time.time(), "ok": False, "result": None,
                        "error": control_runtime._redact(message)[:500]}
            write_json(self.directory, self.directory / "responses" / path.name, signed(body, self.token))
            return True
        except (OSError, ValueError, TypeError, RecursionError):
            return False
        finally:
            try:
                checked_child(self.directory, claimed).unlink(missing_ok=True)
            except (OSError, ValueError):
                pass

    def cleanup(self):
        cutoff = time.time() - 700
        for name in ("responses", "working", "seen"):
            directory = checked_child(self.directory, self.directory / name)
            for path in directory.iterdir():
                try:
                    if checked_child(self.directory, path).is_file() and path.stat().st_mtime < cutoff:
                        path.unlink()
                except (OSError, ValueError):
                    continue

    def run(self):
        lock_path = checked_child(self.directory, self.directory / "bridge.lock")
        with lock_path.open("a+b") as lock:
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            pid_path = self.directory / "bridge.pid.json"
            process = psutil.Process()
            write_json(self.directory, pid_path, {"pid": process.pid, "create_time": process.create_time(), "directory": str(self.directory)})
            try:
                with ThreadPoolExecutor(max_workers=4, thread_name_prefix="grid-host") as pool:
                    next_cleanup = 0
                    while not self.stopping.is_set():
                        if time.monotonic() >= next_cleanup:
                            self.cleanup()
                            next_cleanup = time.monotonic() + 30
                        with os.scandir(checked_child(self.directory, self.directory / "requests")) as entries:
                            for index, entry in enumerate(entries):
                                if index >= 512 or self.stopping.is_set():
                                    break
                                if not entry.name.endswith(".json") or not self.slots.acquire(blocking=False):
                                    continue
                                def work(path=Path(entry.path)):
                                    try:
                                        self.process_file(path)
                                    finally:
                                        self.slots.release()
                                pool.submit(work)
                        self.stopping.wait(0.1)
            finally:
                checked_child(self.directory, pid_path).unlink(missing_ok=True)


def control_request(bridge, operation, timeout):
    identifier = uuid.uuid4().hex
    path = bridge.directory / "requests" / (identifier + ".json")
    response_path = bridge.directory / "responses" / path.name
    created = time.time()
    write_json(bridge.directory, path, signed({"id": identifier, "created_at": created, "op": operation, "args": {}}, bridge.token))
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if response_path.exists():
                response = read_json(bridge.directory, response_path)
                if (authenticated(response, bridge.token) and response.get("id") == identifier and
                        created <= response.get("created_at", 0) <= time.time() + 5):
                    return bool(response.get("ok"))
                return False
            time.sleep(0.1)
        return False
    finally:
        for target in (path, response_path):
            checked_child(bridge.directory, target).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    bridge = HostBridge(args.directory)
    if args.check or args.stop:
        raise SystemExit(0 if control_request(bridge, "stop" if args.stop else "health", 10 if args.stop else 5) else 1)
    bridge.run()


if __name__ == "__main__":
    main()
