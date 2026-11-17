"""Spool RPC tests never contact MT5, a host service or a paid API."""
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

import bridge_client
import evaluator
import mt5_context


TOKEN = bytes(range(32))


@pytest.fixture
def spool(tmp_path, monkeypatch):
    (tmp_path / "token").write_text(TOKEN.hex())
    (tmp_path / "requests").mkdir()
    (tmp_path / "responses").mkdir()
    monkeypatch.setenv("GRID_BOT_BRIDGE_DIR", str(tmp_path))
    monkeypatch.delenv("GRID_BOT_BRIDGE_TOKEN_FILE", raising=False)
    return tmp_path


def reply_once(directory, result=None, mutate=None):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        files = list((directory / "requests").glob("*.json"))
        if files:
            request = json.loads(files[0].read_text(encoding="utf-8"))
            assert request["hmac"] == bridge_client.sign(request, TOKEN)
            response = {"id": request["id"], "created_at": time.time(), "ok": True,
                        "result": result, "error": None}
            if mutate:
                mutate(response)
            response["hmac"] = bridge_client.sign(response, TOKEN)
            temporary = directory / "responses" / (request["id"] + ".tmp")
            temporary.write_bytes(bridge_client.canonical(response))
            temporary.replace(directory / "responses" / (request["id"] + ".json"))
            return request
        time.sleep(.002)
    raise AssertionError("No bridge request appeared")


def test_signed_round_trip_and_cleanup(spool):
    client = bridge_client.get_client()
    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = executor.submit(reply_once, spool, {"mt5_ready": True})
        assert client.request("health", {}, timeout=1) == {"mt5_ready": True}
        sent = worker.result()
    assert sent["op"] == "health" and len(sent["id"]) == 32
    assert client.status()["last_error"] is None and client.status()["last_contact"]
    assert not list((spool / "requests").iterdir())
    assert not list((spool / "responses").iterdir())


def test_real_host_spool_protocol_with_fake_native_ticks(spool, monkeypatch):
    import host_bridge
    host = host_bridge.HostBridge(spool)
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    milliseconds = int(start.timestamp() * 1000)
    calls = []
    ticks = np.array([(milliseconds + 1000, 1., 1.1)], dtype=[("time_msc", "i8"), ("bid", "f8"), ("ask", "f8")])
    fake = SimpleNamespace(initialize=lambda: calls.append("initialize") or True,
                           shutdown=lambda: calls.append("shutdown"), symbol_select=lambda *args: True,
                           COPY_TICKS_INFO=1, copy_ticks_range=lambda *args: ticks)
    monkeypatch.setattr(host_bridge.mt5_context, "mt5", fake)
    def process_one():
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            files = list((host.directory / "requests").glob("*.json"))
            if files:
                return host.process_file(files[0])
            time.sleep(.002)
        raise AssertionError("No host request")
    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = executor.submit(process_one)
        received = bridge_client.MT5Proxy().copy_ticks_range("EURUSD", start, start + dt.timedelta(seconds=10), 1)
        assert worker.result() is True
    np.testing.assert_array_equal(received, ticks)
    assert calls == ["initialize", "shutdown"]


@pytest.mark.parametrize("failure", ["id", "old", "future", "error"])
def test_rejects_wrong_id_stale_future_and_remote_failure(spool, failure):
    def mutate(response):
        if failure == "id":
            response["id"] = "0" * 32
        elif failure == "old":
            response["created_at"] -= 60
        elif failure == "future":
            response["created_at"] += 60
        else:
            response.update(ok=False, result=None, error="MT5 offline")
    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = executor.submit(reply_once, spool, {}, mutate)
        with pytest.raises(bridge_client.BridgeError):
            bridge_client.request("health", {}, timeout=1)
        worker.result()
    assert bridge_client.status()["last_error"]


def test_wrong_token_signature_is_rejected(spool):
    def corrupt_signature():
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            files = list((spool / "requests").glob("*.json"))
            if files:
                request = json.loads(files[0].read_text())
                response = {"id": request["id"], "created_at": time.time(), "ok": True, "result": {}, "error": None}
                response["hmac"] = bridge_client.sign(response, b"wrong-token")
                (spool / "responses" / files[0].name).write_bytes(bridge_client.canonical(response))
                return
            time.sleep(.002)
        raise AssertionError("No request")
    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = executor.submit(corrupt_signature)
        with pytest.raises(bridge_client.BridgeError, match="imzosi"):
            bridge_client.request("health", {}, timeout=1)
        worker.result()


def test_timeout_has_no_retry_and_late_id_is_not_reused(spool, monkeypatch):
    identifiers = []
    original = bridge_client.os.replace
    def track(source, target):
        if Path(target).parent == spool / "requests":
            identifiers.append(Path(target).stem)
        original(source, target)
    monkeypatch.setattr(bridge_client.os, "replace", track)
    with pytest.raises(bridge_client.BridgeError, match="javob bermadi"):
        bridge_client.request("health", {}, timeout=.03)
    assert len(identifiers) == 1
    late = {"id": identifiers[0], "created_at": time.time(), "ok": True, "result": "stale", "error": None}
    late["hmac"] = bridge_client.sign(late, TOKEN)
    (spool / "responses" / f"{identifiers[0]}.json").write_bytes(bridge_client.canonical(late))
    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = executor.submit(reply_once, spool, "new-result")
        assert bridge_client.request("health", {}, timeout=1) == "new-result"
        worker.result()
    assert len(identifiers) == 2 and identifiers[0] != identifiers[1]
    assert not list((spool / "requests").glob("*.json"))


def test_invalid_token_and_full_queue_fail_safely(spool):
    (spool / "token").write_text("invalid")
    with pytest.raises(bridge_client.BridgeError, match="token"):
        bridge_client.request("health", {})
    (spool / "token").write_text(TOKEN.hex())
    for i in range(bridge_client.MAX_PENDING):
        (spool / "requests" / f"{i}.json").write_text("{}")
    with pytest.raises(bridge_client.BridgeError, match="navbati"):
        bridge_client.request("health", {})


def test_spool_child_link_cannot_escape_mount(spool, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (spool / "requests").rmdir()
    try:
        (spool / "requests").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("This Windows account cannot create symbolic links")
    with pytest.raises(bridge_client.BridgeError, match="havolasi"):
        bridge_client.request("health", {}, timeout=.05)
    assert not list(outside.iterdir())


def test_oversized_request_never_enters_queue(spool, monkeypatch):
    monkeypatch.setattr(bridge_client, "MAX_BYTES", 1024)
    with pytest.raises(bridge_client.BridgeError, match="juda katta"):
        bridge_client.request("logs", {"source": "x" * 2000}, timeout=.05)
    assert not list((spool / "requests").iterdir())


def test_bridge_context_keeps_native_indicator_features(spool, monkeypatch):
    closes = np.linspace(100., 110., 100)
    context = mt5_context.compute_raw_indicators(closes + 1, closes - 1, closes, bar_hour=12)
    context.update(symbol="XAUUSD", timeframe="M15", bid=110., ask=110.1, digits=2, tick_time=time.time())
    expected = mt5_context.to_feature_vector(context)
    monkeypatch.setattr(bridge_client, "request", lambda op, args, **kwargs: dict(context))
    monkeypatch.setattr(mt5_context, "mt5", None)
    result = mt5_context.get_context("XAUUSD", "M15")
    assert mt5_context.to_feature_vector(result) == expected
    assert result["ask"] == 110.1
    context["tick_time"] -= 100
    assert mt5_context.get_context("XAUUSD", "M15") is None
    assert "eskirgan" in bridge_client.status()["last_error"]


def test_bridge_offline_returns_no_context_quote_or_tick_outcome(spool, monkeypatch):
    def offline(*args, **kwargs):
        raise bridge_client.BridgeError("Host offline")
    monkeypatch.setattr(bridge_client, "request", offline)
    assert mt5_context.get_context("XAUUSD", "M15") is None
    assert mt5_context.get_quote("XAUUSD") is None
    start = dt.datetime(2026, 1, 1)
    assert evaluator._tick_path_outcome("XAUUSD", "Buy", 2, 1, start, start + dt.timedelta(seconds=10)) == "NO_DATA"
    with pytest.raises(RuntimeError, match="host bridge"):
        evaluator.evaluate_pending()


def test_tick_proxy_preserves_bid_ask_and_five_minute_bound(spool, monkeypatch):
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    start_ms = int(start.timestamp() * 1000)
    sent = []
    def ticks(op, args, **kwargs):
        sent.append((op, args))
        return [{"time_msc": start_ms + 1000, "bid": .9, "ask": 1.1},
                {"time_msc": start_ms + 2000, "bid": 1.9, "ask": 2.1}]
    monkeypatch.setattr(bridge_client, "request", ticks)
    assert evaluator._tick_path_outcome("EURUSD", "Sell", 1., 2., start.replace(tzinfo=None),
                                        (start + dt.timedelta(seconds=10)).replace(tzinfo=None)) == "SL"
    assert sent[0][1] == {"symbol": "EURUSD", "start_ms": start_ms, "end_ms": start_ms + 10000}
    with pytest.raises(bridge_client.BridgeError, match="5 daqiqadan"):
        bridge_client.MT5Proxy().copy_ticks_range("EURUSD", start, start + dt.timedelta(minutes=6), 1)
