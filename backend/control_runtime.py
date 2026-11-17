"""Local telemetry and narrowly scoped, application-owned service controls."""
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import psutil

import config
import mt5_context
import bridge_client

HERE = Path(__file__).resolve().parent
# The configured workspace itself may be a Windows junction (Desktop -> X:).
# Trust that configured base once; checks below still reject redirected children.
RUNTIME_DIR = Path(config.RUNTIME_DIR).resolve()
OWNER_FILE = RUNTIME_DIR / "ollama-owner.json"
LOG_BASE = Path(config.LOG_DIR).resolve()
LOG_PATHS = {"server": LOG_BASE / "server_log.txt", "error": LOG_BASE / "server_error.log",
             "ollama": RUNTIME_DIR / "ollama.log"}
MAX_LOG_BYTES = 256 * 1024
_service_lock = threading.RLock()
_gpu_lock = threading.Lock()
_gpu_cache = None
_gpu_checked_at = 0.0
_backend_server = None
_backend_restart_requested = False


def _hidden_flags():
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _executable(name):
    if name == "ollama":
        candidates = [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"]
    else:
        candidates = [Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"]
    found = shutil.which(name)
    if found:
        candidates.append(Path(found))
    return next((str(path.resolve()) for path in candidates if path.is_file()), None)


def _number(value, multiplier=1):
    try:
        number = float(value.strip()) * multiplier
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def _gpu_snapshot():
    global _gpu_cache, _gpu_checked_at
    with _gpu_lock:
        if _gpu_cache is not None and time.monotonic() - _gpu_checked_at < 10:
            return _gpu_cache
        result = {"gpus": []}
        executable = _executable("nvidia-smi")
        if executable is None:
            result["gpu_error"] = "NVIDIA telemetry mavjud emas"
        else:
            try:
                response = subprocess.run(
                    [executable, "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True,
                    timeout=2, check=True, creationflags=_hidden_flags(), shell=False,
                )
                for line in response.stdout.splitlines()[:128]:
                    fields = line.split(",")
                    if len(fields) != 6:
                        continue
                    result["gpus"].append({"index": fields[0].strip(), "name": fields[1].strip(),
                        "utilization_percent": _number(fields[2]),
                        "memory_used_bytes": _number(fields[3], 1024 * 1024),
                        "memory_total_bytes": _number(fields[4], 1024 * 1024),
                        "temperature_c": _number(fields[5])})
            except (OSError, subprocess.SubprocessError):
                result["gpu_error"] = "NVIDIA telemetry javob bermadi"
        _gpu_cache, _gpu_checked_at = result, time.monotonic()
        return result


def runtime_snapshot():
    if bridge_client.enabled():
        process = psutil.Process(os.getpid())
        local_process = {"pid": process.pid, "uptime_seconds": max(0, time.time() - process.create_time()),
                         "rss_bytes": process.memory_info().rss}
        try:
            host = bridge_client.request("runtime", {}, timeout=4)
            return {**host, "scope": "windows_host", "host_process": host.get("process"), "process": local_process}
        except bridge_client.BridgeError:
            return {"scope": "windows_host", "cpu_percent": None, "ram": {}, "disk": {}, "gpus": [],
                    "process": local_process, "warning": "Windows MT5 ko'prigi bilan aloqa yo'q; host ko'rsatkichlari mavjud emas"}
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(HERE.anchor))
    process = psutil.Process(os.getpid())
    # API calls may use different pool threads; an explicit short sample avoids
    # psutil's first-call zero for each thread being presented as CPU idleness.
    return {"cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram": {"used_bytes": memory.used, "total_bytes": memory.total, "percent": memory.percent},
            "disk": {"used_bytes": disk.used, "total_bytes": disk.total, "percent": disk.percent},
            "process": {"pid": process.pid, "uptime_seconds": max(0, time.time() - process.create_time()),
                        "rss_bytes": process.memory_info().rss}, **_gpu_snapshot()}


def _redact(text):
    # Also remove the configured key verbatim; providers do not all use a
    # recognizable prefix. Never include the settings or environment themselves.
    secrets = [getattr(config, "GEMINI_API_KEY", "")]
    getter = getattr(config, "get_settings", None)
    if getter:
        secrets.append(getter().get("gemini_api_key", ""))
    for secret in secrets:
        if isinstance(secret, str) and secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"AIza[0-9A-Za-z_-]{20,}|\bsk-[0-9A-Za-z_-]{12,}", "[REDACTED]", text)
    text = re.sub(r'''(?i)(["']?authorization["']?\s*[:=]\s*["']?)(?:bearer|basic)\s+[^"'\s,;]+''', r"\1[REDACTED]", text)
    text = re.sub(r'''(?i)(["']?(?:[a-z_]*api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|secret)["']?\s*[:=]\s*)(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s&,;]+)''',
                  r"\1[REDACTED]", text)
    return text


def read_logs(source, lines=100):
    if source not in LOG_PATHS:
        raise ValueError("Noma'lum log manbasi")
    if isinstance(lines, bool) or not isinstance(lines, int) or not 1 <= lines <= 500:
        raise ValueError("Log qatorlari 1 dan 500 gacha bo'lishi kerak")
    if source == "ollama" and bridge_client.enabled():
        return bridge_client.request("logs", {"source": source, "lines": lines}, timeout=3)
    path = LOG_PATHS[source]
    if path.resolve() != path.absolute():
        raise ValueError("Log uchun yo'naltirilgan faylga ruxsat berilmagan")
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - MAX_LOG_BYTES))
            raw = stream.read(MAX_LOG_BYTES)
        text = raw.decode("utf-8", errors="replace")
        if size > MAX_LOG_BYTES:
            text = text.partition("\n")[2]  # skip a potentially partial secret/line
        return {"source": source, "lines": _redact(text).splitlines()[-lines:], "available": True}
    except FileNotFoundError:
        return {"source": source, "lines": [], "available": False}


def _ollama_endpoint():
    parsed = urlparse(config.OLLAMA_URL)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.path not in ("", "/"):
        return None
    return parsed.hostname, parsed.port or 80


def _listening(endpoint):
    if endpoint is None:
        return False
    try:
        with socket.create_connection(endpoint, timeout=0.25):
            return True
    except OSError:
        return False


def _owned_ollama():
    """Verify persisted process identity so backend restarts preserve ownership."""
    try:
        if OWNER_FILE.resolve() != OWNER_FILE.absolute():
            return None
        owner = json.loads(OWNER_FILE.read_text(encoding="utf-8"))
        process = psutil.Process(int(owner["pid"]))
        if (abs(process.create_time() - owner["create_time"]) > 0.01 or
                Path(process.exe()).resolve() != Path(owner["executable"]).resolve()):
            return None
        command = process.cmdline()
        return process if len(command) == 2 and command[1] == "serve" and process.is_running() else None
    except (OSError, ValueError, KeyError, TypeError, psutil.Error):
        return None


def _supervised():
    if _backend_server is None or os.environ.get("GRID_BOT_SUPERVISED") != "1":
        return False
    try:
        parent_pid = int(os.environ.get("GRID_BOT_SUPERVISOR_PID", "0"))
        return parent_pid == os.getppid() and psutil.pid_exists(parent_pid)
    except ValueError:
        return False


def _capability(enabled, reason=""):
    return {"enabled": bool(enabled), "reason": "" if enabled else reason}


def service_capabilities():
    if bridge_client.enabled():
        try:
            capabilities = bridge_client.request("capabilities", {}, timeout=2)
        except bridge_client.BridgeError:
            unavailable = _capability(False, "Windows MT5 ko'prigi bilan aloqa yo'q")
            capabilities = {"ollama": {name: dict(unavailable) for name in ("start", "stop", "restart")},
                            "mt5": {"check": dict(unavailable)}}
        capabilities["backend"] = {"restart": _capability(_supervised() and not _backend_restart_requested,
                                    "Backend nazoratchisi mavjud emas yoki restart bajarilmoqda")}
        return capabilities
    with _service_lock:
        endpoint = _ollama_endpoint()
        owned = _owned_ollama()
        occupied = _listening(endpoint)
        local = endpoint is not None
        external_reason = "Ollama boshqa dastur tomonidan ishga tushirilgan; bu panel uni to'xtatmaydi"
        executable = _executable("ollama")
        return {"ollama": {
                    "start": _capability(local and not occupied and owned is None and executable,
                        "Ollama ishlayapti yoki lokal bajariluvchi fayl/URL mavjud emas"),
                    "stop": _capability(owned is not None, external_reason),
                    "restart": _capability(owned is not None and local and executable, external_reason)},
                "mt5": {"check": _capability(mt5_context.mt5 is not None, "MetaTrader5 Python paketi mavjud emas")},
                "backend": {"restart": _capability(_supervised() and not _backend_restart_requested,
                    "Backend start_server.pyw nazoratchisi orqali ishga tushirilmagan yoki restart bajarilmoqda")}}


def _start_ollama():
    endpoint = _ollama_endpoint()
    executable = _executable("ollama")
    if endpoint is None or executable is None:
        raise ValueError("Lokal Ollama URL yoki dastur topilmadi")
    if _owned_ollama() is not None or _listening(endpoint):
        raise ValueError("Ollama porti band; mavjud jarayon o'zgartirilmadi")
    if RUNTIME_DIR.resolve() != RUNTIME_DIR.absolute():
        raise ValueError("Runtime uchun yo'naltirilgan papkaga ruxsat berilmagan")
    RUNTIME_DIR.mkdir(exist_ok=True)
    environment = os.environ.copy()
    host, port = endpoint
    environment["OLLAMA_HOST"] = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    with LOG_PATHS["ollama"].open("ab") as log:
        child = subprocess.Popen([executable, "serve"], cwd=str(Path(executable).parent),
            env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            creationflags=_hidden_flags(), close_fds=True, shell=False)
    try:
        process = psutil.Process(child.pid)
        owner = {"pid": child.pid, "create_time": process.create_time(), "executable": executable}
        temporary = OWNER_FILE.with_suffix(".tmp")
        temporary.write_text(json.dumps(owner), encoding="utf-8")
        temporary.replace(OWNER_FILE)
    except Exception:
        child.terminate()
        child.wait(timeout=5)
        raise
    return {"service": "ollama", "status": "starting", "managed": True, "pid": child.pid}


def _stop_ollama():
    process = _owned_ollama()
    if process is None:
        raise ValueError("Faqat ushbu panel ishga tushirgan Ollama to'xtatiladi")
    # Only the verified owned process and its descendants may be terminated.
    processes = process.children(recursive=True) + [process]
    for child in reversed(processes):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=2)
    if alive:
        raise RuntimeError("Ollama jarayoni to'xtamadi")
    OWNER_FILE.unlink(missing_ok=True)
    return {"service": "ollama", "status": "stopped", "managed": False}


def service_action(service, action):
    global _backend_restart_requested
    allowed = {"ollama": ("start", "stop", "restart", "check"), "mt5": ("check",), "backend": ("restart", "check")}
    if service not in allowed or action not in allowed[service]:
        raise ValueError("Xizmat yoki amal qo'llab-quvvatlanmaydi")
    if service != "backend" and bridge_client.enabled():
        return bridge_client.request("service", {"service": service, "action": action}, timeout=15)
    with _service_lock:
        if service == "ollama":
            if action == "check":
                return {"service": service, "status": "running" if _listening(_ollama_endpoint()) else "unavailable",
                        "managed": _owned_ollama() is not None}
            if action == "start":
                return _start_ollama()
            if action == "stop":
                return _stop_ollama()
            _stop_ollama()
            return _start_ollama()
        if service == "mt5":
            with mt5_context.MT5_LOCK:
                mt5 = mt5_context.mt5
                if mt5 is None or not mt5.initialize():
                    return {"service": service, "status": "unavailable", "connected": False}
                try:
                    terminal = mt5.terminal_info()
                    connected = bool(terminal and terminal.connected)
                    return {"service": service, "status": "ready" if connected else "disconnected", "connected": connected}
                finally:
                    mt5.shutdown()
        if action == "check":
            return {"service": service, "status": "running", "managed": _supervised()}
        if not _supervised() or _backend_restart_requested:
            raise ValueError("Backend restarti uchun faol nazoratchi talab qilinadi")
        _backend_restart_requested = True
        timer = threading.Timer(1.0, lambda: setattr(_backend_server, "should_exit", True))
        timer.daemon = True
        timer.start()
        return {"service": service, "status": "restarting", "managed": True}


def run_supervised_server():
    """Launcher child: graceful Uvicorn shutdown with an explicit restart code."""
    global _backend_server
    import uvicorn
    _backend_server = uvicorn.Server(uvicorn.Config("main:app", host=os.environ.get("GRID_BOT_BIND_HOST", "127.0.0.1"),
                                                   port=8811, timeout_graceful_shutdown=15, proxy_headers=False))
    _backend_server.run()
    sys.exit(75 if _backend_restart_requested else 0)
