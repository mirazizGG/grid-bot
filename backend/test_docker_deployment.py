"""Docker boundaries and host integration, without running host services."""
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import bridge_client
import control_api
import control_runtime
import docker_entrypoint
import main
import network_guard
import ollama_vision
from test_core import local_client


def test_docker_only_trusts_explicit_gateway_and_keeps_origin_checks(monkeypatch, tmp_path):
    route = tmp_path / "route"
    route.write_text("Iface Destination Gateway Flags\neth0 00000000 010011AC 0003\n")
    monkeypatch.setattr(network_guard, "ROUTE_FILE", route)
    monkeypatch.delenv("GRID_BOT_CONTAINER", raising=False)
    monkeypatch.delenv("GRID_BOT_TRUST_DOCKER_GATEWAY", raising=False)
    assert network_guard.trusted_client("127.0.0.1")
    assert not network_guard.trusted_client("172.17.0.1")
    monkeypatch.setenv("GRID_BOT_CONTAINER", "1")
    monkeypatch.setenv("GRID_BOT_TRUST_DOCKER_GATEWAY", "1")
    assert network_guard.trusted_client("172.17.0.1")
    assert not network_guard.trusted_client("172.17.0.2")
    from fastapi.testclient import TestClient
    client = TestClient(main.app, base_url="http://127.0.0.1:8811", client=("172.17.0.1", 40000))
    assert client.get("/health").status_code == 200
    assert client.get("/health", headers={"Origin": "https://external.example"}).status_code == 403
    assert client.get("/health", headers={"Host": "external.example"}).status_code == 403
    remote = TestClient(main.app, base_url="http://127.0.0.1:8811", client=("192.0.2.1", 40000))
    assert remote.get("/health", headers={"X-Forwarded-For": "127.0.0.1"}).status_code == 403


@pytest.mark.parametrize("contents", ["", "not a route", "header\neth0 00000000 ZZZZ 0003", "header\neth0 00000000 08080808 0003"])
def test_missing_or_public_gateway_fails_closed(monkeypatch, tmp_path, contents):
    route = tmp_path / "route"
    route.write_text(contents)
    monkeypatch.setattr(network_guard, "ROUTE_FILE", route)
    monkeypatch.setenv("GRID_BOT_CONTAINER", "1")
    monkeypatch.setenv("GRID_BOT_TRUST_DOCKER_GATEWAY", "1")
    assert not network_guard.trusted_client("192.0.2.1")


def test_model_seed_preserves_existing_pair_and_refuses_partial_storage(tmp_path):
    source, target = tmp_path / "seed", tmp_path / "model"
    source.mkdir()
    for name in docker_entrypoint.FILENAMES:
        (source / name).write_bytes(b"seed")
    docker_entrypoint.seed_model(source, target)
    (target / docker_entrypoint.FILENAMES[0]).write_bytes(b"active-changed")
    docker_entrypoint.seed_model(source, target)
    assert (target / docker_entrypoint.FILENAMES[0]).read_bytes() == b"active-changed"
    (target / docker_entrypoint.FILENAMES[1]).unlink()
    with pytest.raises(RuntimeError, match="to'liq emas"):
        docker_entrypoint.seed_model(source, target)


def test_failed_seed_copy_never_installs_half_pair(tmp_path):
    source, target = tmp_path / "seed", tmp_path / "model"
    source.mkdir()
    (source / docker_entrypoint.FILENAMES[0]).write_bytes(b"seed")
    with pytest.raises(FileNotFoundError):
        docker_entrypoint.seed_model(source, target)
    assert list(target.iterdir()) == []


def test_docker_metrics_use_windows_host_and_backend_process(monkeypatch):
    monkeypatch.setattr(bridge_client, "enabled", lambda: True)
    monkeypatch.setattr(bridge_client, "request", lambda *a, **kw: {
        "cpu_percent": 23, "ram": {"percent": 40}, "disk": {"percent": 30}, "gpus": [], "process": {"pid": 9876}})
    result = control_runtime.runtime_snapshot()
    assert result["scope"] == "windows_host" and result["cpu_percent"] == 23
    assert result["host_process"]["pid"] == 9876 and result["process"]["pid"] == os.getpid()


def test_missing_host_does_not_present_container_metrics_as_windows(monkeypatch):
    monkeypatch.setattr(bridge_client, "enabled", lambda: True)
    def fail(*a, **kw):
        raise bridge_client.BridgeError("offline")
    monkeypatch.setattr(bridge_client, "request", fail)
    result = control_runtime.runtime_snapshot()
    assert result["scope"] == "windows_host" and result["cpu_percent"] is None
    assert result["ram"] == {} and result["warning"]
    caps = control_runtime.service_capabilities()
    assert not caps["mt5"]["check"]["enabled"] and not caps["ollama"]["stop"]["enabled"]


def test_host_controls_and_ollama_logs_use_scoped_bridge(monkeypatch):
    monkeypatch.setattr(bridge_client, "enabled", lambda: True)
    calls = []
    def request(op, args, **kw):
        calls.append((op, args))
        return {"ok": True}
    monkeypatch.setattr(bridge_client, "request", request)
    assert control_runtime.service_action("ollama", "stop") == {"ok": True}
    assert control_runtime.read_logs("ollama", 10) == {"ok": True}
    assert calls == [("service", {"service": "ollama", "action": "stop"}),
                     ("logs", {"source": "ollama", "lines": 10})]
    with pytest.raises(ValueError):
        control_runtime.service_action("shell", "run")


def test_qwen_inference_crosses_bridge_without_host_network_port(monkeypatch):
    monkeypatch.setattr(bridge_client, "enabled", lambda: True)
    calls = []
    def request(op, args, **kw):
        calls.append((op, args, kw))
        return {"message": {"content": json.dumps({"signal": "Buy", "confidence": 0.7,
                                                   "trend": "up", "reasoning": "test"})}}
    monkeypatch.setattr(bridge_client, "request", request)
    monkeypatch.setattr(ollama_vision, "urlopen", lambda *a, **kw: pytest.fail("must use authenticated file bridge"))
    result = ollama_vision.analyze_image(b"image", model="test:4b", timeout=60)
    assert result["signal"] == "Buy"
    assert calls[0][0] == "ollama_chat" and calls[0][1]["timeout"] == 60
    assert calls[0][2]["timeout"] == 90


def test_docker_services_model_tags_use_host_bridge(monkeypatch):
    monkeypatch.setattr(bridge_client, "enabled", lambda: True)
    monkeypatch.setattr(main.mt5_context, "get_quote", lambda *a: None)
    monkeypatch.setattr(main, "urlopen", lambda *a, **kw: pytest.fail("container loopback is not Windows Ollama"))
    selected = main.config.get_settings()["qwen_model"]
    monkeypatch.setattr(bridge_client, "request", lambda *a, **kw: {"models": [{"name": selected}]})
    result = local_client().get("/services")
    assert result.status_code == 200 and result.json()["qwen"]["status"] == "ready"


def test_compose_limits_exposure_and_preserves_application_state():
    import yaml
    root = Path(__file__).resolve().parents[1]
    if not (root / "compose.yaml").exists():
        pytest.skip("deployment manifests are outside the runtime image")
    definition = yaml.safe_load((root / "compose.yaml").read_text())
    app = definition["services"]["app"]
    assert app["ports"] == ["127.0.0.1:${GRID_BOT_HTTP_PORT:-8811}:8811"]
    assert app["read_only"] is True and app["cap_drop"] == ["ALL"]
    assert len(app["volumes"]) == 4
    assert not any("docker.sock" in value for value in app["volumes"])
    assert "privileged" not in app
