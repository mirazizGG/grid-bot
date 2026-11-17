"""Control-panel contracts; every process, GPU and MT5 operation is mocked."""
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import control_runtime as runtime


@pytest.fixture(autouse=True)
def isolate_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(runtime, "OWNER_FILE", tmp_path / "runtime" / "owner.json")
    monkeypatch.setattr(runtime, "LOG_PATHS", {"server": tmp_path / "server.log", "error": tmp_path / "error.log",
                                              "ollama": tmp_path / "runtime" / "ollama.log"})
    monkeypatch.setattr(runtime, "_gpu_cache", None)
    monkeypatch.setattr(runtime, "_backend_server", None)
    monkeypatch.setattr(runtime, "_backend_restart_requested", False)


def test_telemetry_contract_and_fixed_gpu_command(monkeypatch):
    monkeypatch.setattr(runtime.psutil, "cpu_percent", lambda interval: 12)
    monkeypatch.setattr(runtime.psutil, "virtual_memory", lambda: SimpleNamespace(used=100, total=400, percent=25))
    monkeypatch.setattr(runtime.psutil, "disk_usage", lambda path: SimpleNamespace(used=20, total=100, percent=20))
    monkeypatch.setattr(runtime.psutil, "Process", lambda pid: SimpleNamespace(pid=pid, create_time=lambda: 100,
                         memory_info=lambda: SimpleNamespace(rss=500)))
    monkeypatch.setattr(runtime.time, "time", lambda: 160)
    monkeypatch.setattr(runtime, "_executable", lambda name: "nvidia-smi.exe")
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="0, Test GPU, 70, 100, 8000, 55\n")
    monkeypatch.setattr(runtime.subprocess, "run", run)
    result = runtime.runtime_snapshot()
    assert result["cpu_percent"] == 12
    assert result["ram"] == {"used_bytes": 100, "total_bytes": 400, "percent": 25}
    assert result["process"]["uptime_seconds"] == 60
    assert result["gpus"][0]["memory_used_bytes"] == 100 * 1024 * 1024
    assert calls[0][1]["timeout"] == 2 and calls[0][1]["shell"] is False
    assert calls[0][0][2] == "--format=csv,noheader,nounits"
    runtime._gpu_snapshot()
    assert len(calls) == 1  # polling does not spawn nvidia-smi every time


def test_gpu_timeout_is_bounded_unavailability(monkeypatch):
    monkeypatch.setattr(runtime, "_executable", lambda name: "nvidia-smi.exe")
    def timeout(*args, **kwargs):
        raise runtime.subprocess.TimeoutExpired("nvidia-smi", 2)
    monkeypatch.setattr(runtime.subprocess, "run", timeout)
    assert runtime._gpu_snapshot()["gpus"] == []
    assert "gpu_error" in runtime._gpu_snapshot()


def test_configured_workspace_alias_is_canonicalized_but_child_links_rejected(monkeypatch, tmp_path):
    alias = tmp_path / "desktop-alias" / "backend" / "runtime"
    canonical = tmp_path / "actual-project" / "backend" / "runtime"
    canonical.mkdir(parents=True)
    (canonical / "ollama.log").write_text("owned process log\n", encoding="utf-8")
    original_resolve = Path.resolve
    redirect_child = [False]

    def resolve(path, *args, **kwargs):
        if path.is_relative_to(alias):
            return canonical / path.relative_to(alias)
        if redirect_child[0] and path == canonical / "ollama.log":
            return tmp_path / "outside-secret.log"
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(runtime.config, "RUNTIME_DIR", alias)
    spec = importlib.util.spec_from_file_location("control_runtime_alias_probe", runtime.__file__)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    assert probe.RUNTIME_DIR == canonical
    assert probe.OWNER_FILE.resolve() == probe.OWNER_FILE.absolute()
    assert probe.read_logs("ollama")["lines"] == ["owned process log"]
    redirect_child[0] = True
    with pytest.raises(ValueError, match="yo'naltirilgan"):
        probe.read_logs("ollama")


def test_log_redaction_bounds_and_allowlist(monkeypatch):
    monkeypatch.setattr(runtime.config, "GEMINI_API_KEY", "configured-private-key")
    monkeypatch.setattr(runtime.config, "get_settings", lambda: {"gemini_api_key": "changed-private-key"})
    runtime.LOG_PATHS["server"].write_text('old\nGEMINI_API_KEY=configured-private-key\n'
        '{"Authorization": "Bearer hidden-bearer", "password": "hidden-password"}\n'
        'GET /?token=hidden-token&x=1\n', encoding="utf-8")
    result = runtime.read_logs("server", 3)
    assert result["available"] and len(result["lines"]) == 3
    assert all(secret not in str(result) for secret in ("configured-private-key", "hidden-bearer", "hidden-password", "hidden-token"))
    assert runtime.read_logs("error")["available"] is False
    with pytest.raises(ValueError):
        runtime.read_logs("../.env")
    for count in (0, 501, True):
        with pytest.raises(ValueError):
            runtime.read_logs("server", count)
    monkeypatch.setattr(runtime, "MAX_LOG_BYTES", 12)
    runtime.LOG_PATHS["server"].write_bytes(b"secret-partial-line\nlast\n")
    assert runtime.read_logs("server")["lines"] == ["last"]
    assert runtime._redact("changed-private-key") == "[REDACTED]"


def test_external_ollama_cannot_be_stopped_or_restarted(monkeypatch):
    monkeypatch.setattr(runtime, "_owned_ollama", lambda: None)
    monkeypatch.setattr(runtime, "_listening", lambda endpoint: True)
    monkeypatch.setattr(runtime, "_executable", lambda name: "ollama.exe")
    caps = runtime.service_capabilities()["ollama"]
    assert not any(item["enabled"] for item in caps.values())
    for action in ("start", "stop", "restart"):
        with pytest.raises(ValueError):
            runtime.service_action("ollama", action)


def test_ollama_start_persists_owned_identity_with_hidden_fixed_command(monkeypatch):
    monkeypatch.setattr(runtime, "_owned_ollama", lambda: None)
    monkeypatch.setattr(runtime, "_listening", lambda endpoint: False)
    monkeypatch.setattr(runtime, "_executable", lambda name: str(Path("C:/test/ollama.exe")))
    monkeypatch.setattr(runtime, "_ollama_endpoint", lambda: ("127.0.0.1", 11434))
    calls = []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=1234)
    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    monkeypatch.setattr(runtime.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 200))
    result = runtime.service_action("ollama", "start")
    assert result["status"] == "starting" and result["managed"]
    assert json.loads(runtime.OWNER_FILE.read_text())["create_time"] == 200
    assert calls[0][0][1:] == ["serve"]
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["env"]["OLLAMA_HOST"] == "127.0.0.1:11434"


def test_reused_pid_is_never_owned(monkeypatch):
    runtime.RUNTIME_DIR.mkdir()
    runtime.OWNER_FILE.write_text(json.dumps({"pid": 12, "create_time": 100, "executable": "ollama.exe"}))
    monkeypatch.setattr(runtime.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 101))
    assert runtime._owned_ollama() is None


def test_owned_stop_terminates_only_owned_descendants(monkeypatch):
    terminated = []
    child = SimpleNamespace(terminate=lambda: terminated.append("child"))
    parent = SimpleNamespace(terminate=lambda: terminated.append("parent"), children=lambda recursive: [child])
    monkeypatch.setattr(runtime, "_owned_ollama", lambda: parent)
    monkeypatch.setattr(runtime.psutil, "wait_procs", lambda processes, timeout: (processes, []))
    assert runtime.service_action("ollama", "stop")["status"] == "stopped"
    assert set(terminated) == {"child", "parent"}


def test_mt5_check_uses_shared_connection_without_orders(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.mt5_context, "mt5", SimpleNamespace(
        initialize=lambda: calls.append("initialize") or True,
        terminal_info=lambda: SimpleNamespace(connected=True), shutdown=lambda: calls.append("shutdown")))
    assert runtime.service_action("mt5", "check")["connected"]
    assert calls == ["initialize", "shutdown"]
    with pytest.raises(ValueError):
        runtime.service_action("mt5", "stop")


def test_restart_requires_supervisor_and_defers_graceful_exit(monkeypatch):
    with pytest.raises(ValueError):
        runtime.service_action("backend", "restart")
    server = SimpleNamespace(should_exit=False)
    monkeypatch.setattr(runtime, "_backend_server", server)
    monkeypatch.setattr(runtime, "_supervised", lambda: True)
    timers = []
    def timer(delay, callback):
        timers.append((delay, callback))
        return SimpleNamespace(start=lambda: None)
    monkeypatch.setattr(runtime.threading, "Timer", timer)
    assert runtime.service_action("backend", "restart")["status"] == "restarting"
    assert not server.should_exit and timers[0][0] == 1
    timers[0][1]()
    assert server.should_exit
    with pytest.raises(ValueError):
        runtime.service_action("backend", "restart")


def test_launcher_restarts_only_on_explicit_child_exit(monkeypatch, tmp_path):
    loader = importlib.machinery.SourceFileLoader("control_test_launcher", str(Path(__file__).with_name("start_server.pyw")))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    launcher = importlib.util.module_from_spec(spec)
    loader.exec_module(launcher)
    monkeypatch.setattr(launcher, "LOG", tmp_path / "launcher.log")
    monkeypatch.setattr(launcher, "listening", lambda port: False)
    exits, calls = iter([75, 0]), []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(wait=lambda: next(exits))
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    launcher.main()
    assert len(calls) == 2
    assert calls[0][1]["env"]["GRID_BOT_SUPERVISED"] == "1"
    assert calls[0][1]["shell"] is False


@pytest.mark.parametrize("restart,exit_code", [(False, 0), (True, 75)])
def test_supervised_child_uses_graceful_server_exit(monkeypatch, restart, exit_code):
    configs = []
    monkeypatch.setattr(runtime, "_backend_restart_requested", restart)
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(
        Config=lambda *args, **kwargs: configs.append((args, kwargs)) or "config",
        Server=lambda config: SimpleNamespace(run=lambda: None)))
    with pytest.raises(SystemExit) as exc:
        runtime.run_supervised_server()
    assert exc.value.code == exit_code
    assert configs[0][1]["host"] == "127.0.0.1"
    assert configs[0][1]["timeout_graceful_shutdown"] == 15
