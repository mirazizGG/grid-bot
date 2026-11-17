"""Control actions use temporary storage and mocked processes/providers."""
import io
import json
import threading
import time

import pytest

import config
import control_api
import control_runtime
import gemini_vision
import main
import model_control
import ollama_vision
from test_core import local_client, _optional_qwen_record


@pytest.fixture
def panel(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_settings", dict(config._DEFAULT_SETTINGS))
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    queue = control_api.JobQueue(tmp_path / "jobs.json")
    monkeypatch.setattr(control_api, "jobs", queue)
    monkeypatch.setattr(control_api, "BACKUP_DIR", tmp_path / "backups")
    monkeypatch.setattr(control_api, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(main.mt5_context, "get_quote", lambda *a, **kw: None)
    monkeypatch.setattr(main, "urlopen", lambda *a, **kw: io.BytesIO(b'{"models": []}'))
    monkeypatch.setattr(control_runtime, "runtime_snapshot", lambda: {"cpu_percent": 12, "gpus": []})
    monkeypatch.setattr(control_runtime, "service_capabilities", lambda: {
        "ollama": {"start": {"enabled": True}, "stop": {"enabled": False, "reason": "external"}},
        "backend": {"restart": {"enabled": False, "reason": "not supervised"}},
    })
    monkeypatch.setattr(model_control, "registry", lambda: {"active": {"available": True}, "candidate": {"available": False}})
    original = (gemini_vision.GEMINI_API_KEY, gemini_vision.GEMINI_MODEL, gemini_vision._client,
                ollama_vision.QWEN_MODEL, ollama_vision.QWEN_TIMEOUT)
    yield local_client(), queue, tmp_path
    queue.pool.shutdown(wait=True)
    gemini_vision.configure(original[0], original[1])
    gemini_vision._client = original[2]
    ollama_vision.configure(original[3], original[4])


def finished(queue):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        rows = queue.snapshot()
        if rows and all(j["status"] not in ("queued", "running") for j in rows):
            return rows
        threading.Event().wait(0.005)
    pytest.fail("control job did not finish")


def test_overview_and_settings_never_return_credentials(panel):
    client, _, path = panel
    key = 'test-secret-"quoted"-value'
    response = client.post("/control/settings", json={"gemini_api_key": key, "default_symbol": "EURUSD"})
    assert response.status_code == 200
    assert response.json()["api_key_configured"] is True
    assert "gemini_api_key" not in response.json()
    assert key not in response.text
    assert gemini_vision.GEMINI_API_KEY == key
    assert json.loads((path / "settings.json").read_text())["gemini_api_key"] == key
    overview = client.get("/control/overview")
    assert overview.status_code == 200
    assert overview.json()["settings"]["default_symbol"] == "EURUSD"
    assert "gemini_api_key" not in overview.json()["settings"]
    assert overview.json()["system"]["cpu_percent"] == 12
    assert client.get("/control/settings").json()["api_key_configured"] is True


def test_overview_exposes_mt5_connection_reason_and_clears_it_after_recovery(panel, monkeypatch):
    client, _, _ = panel
    error = "MT5 hisobiga kirish tasdiqlanmadi (kod -6)."
    connected = False
    def quote(symbol, *, raise_on_error=False):
        assert raise_on_error is True
        if not connected:
            raise main.mt5_context.MT5ReadError(error)
        return {"bid": 1., "ask": 1.01, "digits": 2, "tick_time": time.time()}
    monkeypatch.setattr(main.mt5_context, "get_quote", quote)
    monkeypatch.setattr(main.bridge_client, "enabled", lambda: False)
    monkeypatch.setattr(gemini_vision, "_get_client", lambda: pytest.fail("overview must not perform paid inference"))
    failure = client.get("/control/overview")
    assert failure.status_code == 200
    assert failure.json()["services"]["mt5"]["status"] == "unavailable"
    assert failure.json()["services"]["mt5"]["error"] == error
    connected = True
    ready = client.get("/control/overview")
    assert ready.status_code == 200
    assert ready.json()["services"]["mt5"]["status"] == "ready"
    assert ready.json()["services"]["mt5"]["error"] is None


@pytest.mark.parametrize("changes", [{"unknown": True}, {"analysis_enabled": 1}, {"qwen_timeout": 0},
                                     {"custom_model_weight": 0.9}, {"min_custom_confidence": "0.5"},
                                     {"default_symbol": "../bad"}, {"gemini_api_key": "bad\nkey"},
                                     {"gemini_model": "bad/model"}, {"default_timeframe": "W1"}])
def test_invalid_settings_are_atomic(panel, changes):
    client, _, path = panel
    before = client.get("/control/settings").json()
    assert client.post("/control/settings", json=changes).status_code == 400
    assert client.get("/control/settings").json() == before
    assert not (path / "settings.json").exists()


def test_settings_disk_failure_keeps_live_values(panel, monkeypatch):
    client, _, _ = panel
    before = config.get_settings()
    monkeypatch.setattr(config.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    assert client.post("/control/settings", json={"analysis_enabled": False}).status_code == 503
    assert config.get_settings() == before


def test_pause_and_provider_flags_block_work_before_inference(panel, monkeypatch, tmp_path):
    client, _, _ = panel
    monkeypatch.setattr(main.mt5_context, "get_context", lambda *a: pytest.fail("paused analysis must not call MT5"))
    image = io.BytesIO()
    main.Image.new("RGB", (2, 2)).save(image, "PNG")
    assert client.post("/control/settings", json={"analysis_enabled": False}).status_code == 200
    assert client.post("/analyze", files={"file": ("a.png", image.getvalue(), "image/png")}).status_code == 503
    assert client.post("/control/settings", json={"analysis_enabled": True, "gemini_enabled": False}).status_code == 200
    assert client.post("/analyze", files={"file": ("a.png", image.getvalue(), "image/png")}).status_code == 503
    assert client.post("/control/settings", json={"qwen_enabled": False}).status_code == 200
    assert client.post("/history/1/qwen").status_code == 409


def test_service_capabilities_and_job_allowlists(panel, monkeypatch):
    client, queue, _ = panel
    calls = []
    monkeypatch.setattr(control_runtime, "service_action", lambda s, a: calls.append((s, a)) or {"status": "started"})
    assert client.post("/control/services/ollama/stop").status_code == 409
    assert client.post("/control/services/shell/execute").status_code == 400
    assert client.post("/control/jobs/arbitrary-command").status_code == 400
    assert client.post("/control/models/arbitrary-path").status_code == 400
    assert client.post("/control/services/ollama/start").status_code == 202
    assert finished(queue)[0]["status"] == "succeeded"
    assert calls == [("ollama", "start")]


def test_job_errors_and_results_redact_secrets(panel, monkeypatch):
    client, queue, path = panel
    key = 'new-"secret"-token'
    config.save_settings({"gemini_api_key": key})
    monkeypatch.setattr(control_api.evaluator, "evaluate_pending", lambda: {"message": "result " + key})
    assert client.post("/control/jobs/evaluate").status_code == 202
    rows = finished(queue)
    assert key not in json.dumps(rows)
    assert "[REDACTED]" in rows[0]["result"]["message"]
    def fail():
        raise RuntimeError("failed with " + key)
    monkeypatch.setattr(control_api.evaluator, "evaluate_pending", fail)
    client.post("/control/jobs/evaluate")
    rows = finished(queue)
    assert rows[0]["status"] == "failed" and key not in rows[0]["error"]
    assert key not in (path / "jobs.json").read_text()


def test_queue_deduplicates_and_recovers_interrupted_jobs(tmp_path):
    queue = control_api.JobQueue(tmp_path / "jobs.json", workers=1, capacity=1)
    started, release = threading.Event(), threading.Event()
    def blocked():
        started.set()
        assert release.wait(5)
        return {"ok": True}
    try:
        first = queue.submit("evaluate", blocked)
        assert started.wait(5)
        assert queue.submit("evaluate", blocked)["id"] == first["id"]
        with pytest.raises(ValueError):
            queue.submit("backup", blocked)
        recovered = control_api.JobQueue(queue.path)
        try:
            assert recovered.snapshot()[0]["status"] == "failed"
        finally:
            recovered.pool.shutdown()
    finally:
        release.set()
        queue.pool.shutdown(wait=True)
    assert queue.snapshot()[0]["status"] == "succeeded"


def test_backups_downloads_logs_and_csv_are_bounded_and_local(panel, monkeypatch, tmp_path):
    client, queue, path = panel
    backup_dir = path / "backups"
    backup_dir.mkdir()
    (backup_dir / "safe.zip").write_bytes(b"zip-data")
    assert client.get("/control/backups").json()["backups"][0]["download_url"] == "/control/backups/safe.zip"
    assert client.get("/control/backups/safe.zip").content == b"zip-data"
    assert client.get("/control/backups/..evil.zip").status_code == 404
    assert client.get("/control/logs?source=arbitrary").status_code == 400
    assert client.get("/control/logs?lines=10000").status_code == 422
    row = _optional_qwen_record(monkeypatch, tmp_path)
    exported = client.get("/control/history/export")
    assert exported.status_code == 200 and "EURUSD" in exported.text
    assert "attachment" in exported.headers["content-disposition"]
    assert client.get("/history?symbol=EURUSD&signal=Buy&limit=1").headers["x-total-count"] == "1"
    assert client.get("/history?symbol=USDJPY").json() == []
    assert client.get("/history?offset=-1").status_code == 422
    assert client.get("/history?limit=10000").status_code == 422
    assert client.get("/history?signal=invalid").status_code == 422
    assert client.get("/history?symbol=bad/path").status_code == 400


def test_mutating_control_routes_require_local_request_header(panel):
    client, _, _ = panel
    assert client.post("/control/settings", json={"analysis_enabled": False},
                       headers={"X-Grid-Bot-Request": ""}).status_code == 403
    assert client.post("/control/jobs/backup", headers={"Origin": "https://external.example"}).status_code == 403


def test_restart_waits_for_jobs_and_prevents_new_work(tmp_path):
    queue = control_api.JobQueue(tmp_path / "jobs.json", workers=1)
    started, release = threading.Event(), threading.Event()
    def blocked():
        started.set()
        assert release.wait(5)
    try:
        queue.submit("evaluate", blocked)
        assert started.wait(5)
        with pytest.raises(ValueError, match="tugashini kuting"):
            queue.submit("backend:restart", lambda: None)
        release.set()
        finished(queue)
        queue.submit("backend:restart", lambda: {"status": "restarting"})
        finished(queue)
        with pytest.raises(ValueError, match="qayta ishga tushmoqda"):
            queue.submit("backup", lambda: None)
    finally:
        release.set()
        queue.pool.shutdown(wait=True)


def test_failed_restart_reopens_queue(panel):
    _, queue, _ = panel
    def fail():
        raise RuntimeError("not managed")
    queue.submit("backend:restart", fail)
    assert finished(queue)[0]["status"] == "failed"
    assert not queue.restarting
    queue.submit("backup", lambda: {"ok": True})
    assert finished(queue)[0]["status"] == "succeeded"


def test_runtime_confidence_changes_are_used_immediately(panel):
    client, _, _ = panel
    custom = {"signal": "Buy", "confidence": 0.6}
    vision = {"signal": "Buy", "confidence": 0.8}
    assert main.compute_consensus(custom, vision)["final_signal"] == "Buy"
    assert client.post("/control/settings", json={"min_custom_confidence": 0.7}).status_code == 200
    assert main.compute_consensus(custom, vision)["final_signal"] == "Hold"


def test_qwen_uses_the_same_settings_for_lease_and_inference(panel, monkeypatch, tmp_path):
    client, _, _ = panel
    row = _optional_qwen_record(monkeypatch, tmp_path)
    selected = config.get_settings()
    real_claim = main.claim_qwen_analysis
    calls = []
    def claim(record_id, model, lease):
        calls.append((model, lease))
        value = real_claim(record_id, model, lease)
        # Simulate an admin changing the provider after the request was accepted.
        ollama_vision.configure("new-model:latest", 600)
        return value
    def infer(image, context, *, model, timeout):
        calls.append((model, timeout))
        return {"signal": "Sell", "confidence": 0.7, "model": model}
    monkeypatch.setattr(main, "claim_qwen_analysis", claim)
    monkeypatch.setattr(ollama_vision, "analyze_image", infer)
    assert client.post(f"/history/{row.id}/qwen").status_code == 200
    assert calls == [(selected["qwen_model"], selected["qwen_timeout"] + 30),
                     (selected["qwen_model"], selected["qwen_timeout"])]
