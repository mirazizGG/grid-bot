"""Host bridge tests: disposable spool, fake MT5, no real service actions."""
import base64
import datetime as dt
import io
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import host_bridge as host


@pytest.fixture
def bridge(tmp_path):
    (tmp_path / "token").write_text("12" * 32, encoding="ascii")
    return host.HostBridge(tmp_path)


def request(bridge, operation="health", args=None, identifier=None, created=None):
    identifier = identifier or uuid.uuid4().hex
    body = {"id": identifier, "created_at": time.time() if created is None else created,
            "op": operation, "args": {} if args is None else args}
    path = bridge.directory / "requests" / (identifier + ".json")
    host.write_json(bridge.directory, path, host.signed(body, bridge.token))
    return path


def response(bridge, path):
    result = host.read_json(bridge.directory, bridge.directory / "responses" / path.name)
    assert host.authenticated(result, bridge.token)
    return result


def test_signed_health_and_replay_rejection(bridge, monkeypatch):
    calls = []
    monkeypatch.setattr(bridge, "dispatch", lambda operation, args: calls.append(operation) or {"status": "ok"})
    path = request(bridge)
    assert bridge.process_file(path)
    assert response(bridge, path)["ok"]
    request(bridge, identifier=path.stem)
    assert not bridge.process_file(path)
    assert calls == ["health"]
    # Replay markers survive a helper restart.
    replacement = host.HostBridge(bridge.directory)
    request(replacement, identifier=path.stem)
    assert not replacement.process_file(path)


@pytest.mark.parametrize("difference", [-31, 6])
def test_expired_or_future_requests_do_not_execute(bridge, monkeypatch, difference):
    monkeypatch.setattr(bridge, "dispatch", lambda *args: pytest.fail("invalid request executed"))
    path = request(bridge, created=time.time() + difference)
    assert not bridge.process_file(path)
    assert not (bridge.directory / "responses" / path.name).exists()


def test_tampered_body_or_mismatched_id_does_not_execute(bridge, monkeypatch):
    monkeypatch.setattr(bridge, "dispatch", lambda *args: pytest.fail("invalid request executed"))
    path = request(bridge)
    document = host.read_json(bridge.directory, path)
    document["op"] = "stop"
    host.write_json(bridge.directory, path, document)
    assert not bridge.process_file(path)
    path = request(bridge)
    wrong_path = path.with_name(uuid.uuid4().hex + ".json")
    path.rename(wrong_path)
    assert not bridge.process_file(wrong_path)


@pytest.mark.parametrize("operation,args", [("order_send", {}), ("login", {}), ("shell", {"command": "whoami"}),
    ("service", {"service": "backend", "action": "restart"}),
    ("service", {"service": "mt5", "action": "stop"}),
    ("quote", {"symbol": "EURUSD", "path": "outside"})])
def test_allowlist_rejects_unsupported_operations(bridge, operation, args):
    path = request(bridge, operation, args)
    assert bridge.process_file(path)
    assert response(bridge, path)["ok"] is False


def test_context_and_quote_use_existing_host_logic(bridge, monkeypatch):
    calls = []
    monkeypatch.setattr(host.mt5_context, "get_context", lambda *args: calls.append(args) or {"symbol": "EURUSD"})
    monkeypatch.setattr(host.mt5_context, "get_quote", lambda name: {"bid": 1.1, "ask": 1.2})
    assert bridge.dispatch("context", {"symbol": "EURUSD", "timeframe": "M15"}) == {"symbol": "EURUSD"}
    assert calls == [("EURUSD", "M15", 100)]
    assert bridge.dispatch("quote", {"symbol": "EURUSD"})["bid"] == 1.1
    for arguments in ({"symbol": "../bad", "timeframe": "M15"},
                      {"symbol": "EURUSD", "timeframe": "M15", "n_bars": 1001},
                      {"symbol": "EURUSD", "timeframe": "W1"}):
        with pytest.raises(ValueError):
            bridge.dispatch("context", arguments)


def test_ticks_preserve_utc_milliseconds_and_cleanup_connection(bridge, monkeypatch):
    calls = []
    def copy(symbol, start, end, flags):
        calls.append((symbol, start, end, flags))
        return [{"time_msc": 1700000000123, "bid": 1.1, "ask": 1.2}]
    fake = SimpleNamespace(initialize=lambda: True, symbol_select=lambda *args: True,
        shutdown=lambda: calls.append("shutdown"), copy_ticks_range=copy, COPY_TICKS_INFO=7)
    monkeypatch.setattr(host.mt5_context, "mt5", fake)
    args = {"symbol": "EURUSD", "start_ms": 1700000000000, "end_ms": 1700000300000}
    assert bridge.dispatch("ticks", args) == [{"time_msc": 1700000000123, "bid": 1.1, "ask": 1.2}]
    assert calls[0][1].tzinfo == dt.timezone.utc and calls[0][2].tzinfo == dt.timezone.utc
    assert calls[-1] == "shutdown"
    with pytest.raises(ValueError):
        bridge.dispatch("ticks", {**args, "end_ms": args["end_ms"] + 1})
    monkeypatch.setattr(fake, "copy_ticks_range", lambda *args: (_ for _ in ()).throw(RuntimeError("fake")))
    with pytest.raises(RuntimeError):
        bridge.dispatch("ticks", args)
    assert calls[-1] == "shutdown"


def chat_args():
    return {"timeout": 120, "payload": {"model": "qwen3.5:4b", "messages": [{"role": "user", "content": "chart",
        "images": [base64.b64encode(b"fake-image").decode()]}], "stream": False, "think": False,
        "format": host._SCHEMA, "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 400}, "keep_alive": 0}}


def test_ollama_proxy_is_fixed_loopback_and_bounded(bridge, monkeypatch):
    calls = []
    def open_url(request, timeout):
        calls.append((request.full_url, json.loads(request.data), timeout))
        return io.BytesIO(b'{"message":{"content":"{}"}}')
    monkeypatch.setattr(host, "urlopen", open_url)
    monkeypatch.setattr(host.config, "OLLAMA_URL", "http://127.0.0.1:11434")
    assert bridge.dispatch("ollama_chat", chat_args())["message"]["content"] == "{}"
    assert calls[0][0] == "http://127.0.0.1:11434/api/chat" and calls[0][2] == 120
    arguments = chat_args()
    arguments["payload"]["options"]["num_predict"] = 999999
    with pytest.raises(ValueError):
        bridge.dispatch("ollama_chat", arguments)
    monkeypatch.setattr(host.config, "OLLAMA_URL", "http://example.com")
    with pytest.raises(ValueError):
        bridge.dispatch("ollama_tags", {})
    assert len(calls) == 1


def test_slow_qwen_leaves_other_operations_available(bridge, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def slow(*args):
        entered.set()
        assert release.wait(5)
        return {"message": {"content": "{}"}}
    monkeypatch.setattr(host, "ollama_request", slow)
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(bridge.dispatch, "ollama_chat", chat_args())
        assert entered.wait(5)
        try:
            assert pool.submit(bridge.dispatch, "health", {}).result(timeout=1)["status"] == "ok"
            with pytest.raises(ValueError, match="busy"):
                bridge.dispatch("ollama_chat", chat_args())
        finally:
            release.set()
        running.result(timeout=5)


def test_host_capabilities_exclude_backend_and_controls_are_scoped(bridge, monkeypatch):
    monkeypatch.setattr(host.control_runtime, "service_capabilities", lambda: {"ollama": {}, "mt5": {}, "backend": {"restart": {"enabled": True}}})
    monkeypatch.setattr(host.control_runtime, "service_action", lambda service, action: {"service": service, "status": action})
    assert set(bridge.dispatch("capabilities", {})) == {"ollama", "mt5"}
    assert bridge.dispatch("service", {"service": "ollama", "action": "start"})["status"] == "start"


def test_oversized_message_and_linked_paths_are_rejected(bridge, monkeypatch):
    path = request(bridge)
    monkeypatch.setattr(host, "MAX_BYTES", 8)
    assert not bridge.process_file(path)
    with pytest.raises(ValueError):
        host.checked_child(bridge.directory, bridge.directory.parent / "outside")
    original = host.Path.is_symlink
    monkeypatch.setattr(host.Path, "is_symlink", lambda path: path == bridge.directory / "token" or original(path))
    with pytest.raises(ValueError, match="links"):
        host.HostBridge(bridge.directory)


def test_helper_round_trip_check_and_stop_without_services(bridge, monkeypatch):
    monkeypatch.setattr(host.mt5_context, "mt5", None)
    worker = threading.Thread(target=bridge.run)
    worker.start()
    try:
        assert host.control_request(bridge, "health", 5)
        assert (bridge.directory / "bridge.pid.json").exists()
        assert host.control_request(bridge, "stop", 5)
    finally:
        bridge.stopping.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert not (bridge.directory / "bridge.pid.json").exists()
