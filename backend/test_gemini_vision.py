"""Gemini response boundaries; no provider requests or production database access."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import httpx
from google.genai import errors

import gemini_vision


@pytest.fixture
def valid_response():
    return {
        "signal": "Buy", "confidence": 0.8,
        "trend": "yuqoriga", "reasoning": "Test tahlili.",
        "entry_type": "market", "entry_note": "Hozir kirish mumkin.",
        "entry_price": 1.1, "tp_price": 1.2, "sl_price": 1,
    }


def set_response(monkeypatch, text):
    client = SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda **kwargs: SimpleNamespace(text=text)))
    monkeypatch.setattr(gemini_vision, "_client", client)


def test_valid_provider_response(monkeypatch, valid_response):
    set_response(monkeypatch, json.dumps(valid_response))
    result = gemini_vision.analyze_image(b"image")
    assert result == {**valid_response, "source": "gemini",
                      "prompt_version": gemini_vision.PROMPT_VERSION, "model": gemini_vision.GEMINI_MODEL}


def test_request_snapshot_keeps_model_and_key_across_settings_change(monkeypatch, valid_response):
    calls = []
    old_client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: calls.append(kw["model"]) or
                                                       SimpleNamespace(text=json.dumps(valid_response))),
                                 close=lambda: calls.append("closed"))
    monkeypatch.setattr(gemini_vision, "GEMINI_API_KEY", "new-key")
    monkeypatch.setattr(gemini_vision, "GEMINI_MODEL", "gemini-new")
    monkeypatch.setattr(gemini_vision.genai, "Client", lambda **kw: calls.append(kw["api_key"]) or old_client)
    result = gemini_vision.analyze_image(b"image", settings={"gemini_api_key": "accepted-key", "gemini_model": "gemini-accepted"})
    assert result["model"] == "gemini-accepted"
    assert calls == ["accepted-key", "gemini-accepted", "closed"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("entry_type", [None, "market", "wait_pullback", "wait_breakout"])
def test_nullable_fields_and_entry_types(monkeypatch, valid_response, entry_type):
    valid_response.update(entry_type=entry_type, entry_note=None,
                          entry_price=None, tp_price=None, sl_price=None)
    set_response(monkeypatch, json.dumps(valid_response))
    result = gemini_vision.analyze_image(b"image")
    assert result["entry_type"] == entry_type and result["entry_note"] is None
    assert all(result[field] is None for field in ("entry_price", "tp_price", "sl_price"))


def test_missing_optional_fields(monkeypatch, valid_response):
    for field in ("entry_type", "entry_note", "entry_price", "tp_price", "sl_price"):
        valid_response.pop(field)
    set_response(monkeypatch, json.dumps(valid_response))
    result = gemini_vision.analyze_image(b"image")
    assert result["entry_type"] is None and result["entry_price"] is None


@pytest.mark.parametrize("text", [None, "", "not JSON", "[]", "null", "42", "true", '"text"'])
def test_rejects_invalid_json_or_non_object(monkeypatch, text):
    set_response(monkeypatch, text)
    with pytest.raises(ValueError, match="Gemini"):
        gemini_vision.analyze_image(b"image")


@pytest.mark.parametrize("field", ["signal", "confidence", "trend", "reasoning"])
def test_rejects_missing_required_fields(monkeypatch, valid_response, field):
    valid_response.pop(field)
    set_response(monkeypatch, json.dumps(valid_response))
    with pytest.raises(ValueError, match="Gemini"):
        gemini_vision.analyze_image(b"image")


@pytest.mark.parametrize("field", ["entry_price", "tp_price", "sl_price"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), float("-inf"),
                                   True, "1.23", {}, [], 10 ** 400])
def test_rejects_invalid_prices(monkeypatch, valid_response, field, value):
    valid_response[field] = value
    set_response(monkeypatch, json.dumps(valid_response))
    with pytest.raises(ValueError, match=field):
        gemini_vision.analyze_image(b"image")


def test_rejects_overflowing_json_exponent(monkeypatch, valid_response):
    text = json.dumps(valid_response).replace('"entry_price": 1.1', '"entry_price": 1e309')
    set_response(monkeypatch, text)
    with pytest.raises(ValueError, match="entry_price"):
        gemini_vision.analyze_image(b"image")


@pytest.mark.parametrize("value", [None, True, "0.8", -0.1, 1.1, float("nan"), float("inf"), [], {}, 10 ** 400])
def test_rejects_invalid_confidence(monkeypatch, valid_response, value):
    valid_response["confidence"] = value
    set_response(monkeypatch, json.dumps(valid_response))
    with pytest.raises(ValueError, match="Gemini"):
        gemini_vision.analyze_image(b"image")


@pytest.mark.parametrize("field,value", [
    ("signal", "BUY"), ("signal", []), ("signal", None),
    ("trend", None), ("trend", []), ("reasoning", {}), ("reasoning", 123),
    ("entry_type", "later"), ("entry_type", []), ("entry_type", False),
    ("entry_note", {}), ("entry_note", 123),
])
def test_rejects_invalid_text_and_entry_types(monkeypatch, valid_response, field, value):
    valid_response[field] = value
    set_response(monkeypatch, json.dumps(valid_response))
    with pytest.raises(ValueError, match="Gemini"):
        gemini_vision.analyze_image(b"image")


def test_missing_key_fails_only_on_analysis(monkeypatch):
    monkeypatch.setattr(gemini_vision, "_client", None)
    monkeypatch.setattr(gemini_vision, "GEMINI_API_KEY", " ")
    monkeypatch.setattr(gemini_vision.genai, "Client", lambda **kwargs: pytest.fail("SDK must not be initialized"))
    with pytest.raises(RuntimeError, match="kaliti sozlanmagan"):
        gemini_vision.analyze_image(b"image")


def test_client_is_created_lazily_and_reused(monkeypatch, valid_response):
    calls = []
    client = SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda **kwargs: SimpleNamespace(text=json.dumps(valid_response))))
    monkeypatch.setattr(gemini_vision, "_client", None)
    monkeypatch.setattr(gemini_vision, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(gemini_vision.genai, "Client", lambda **kwargs: calls.append(kwargs) or client)
    assert not calls
    gemini_vision.analyze_image(b"image")
    gemini_vision.analyze_image(b"image")
    assert calls == [{"api_key": "test-key"}]


def test_server_import_without_credentials():
    script = textwrap.dedent("""
        import io
        import sys
        from types import ModuleType
        from google import genai

        def unexpected_client(**kwargs):
            raise AssertionError("Import must not create a Gemini client")
        genai.Client = unexpected_client
        # Import the real server while preventing all database initialization.
        database = ModuleType("db")
        database.Analysis = object
        database.SessionLocal = None
        database.save_analysis = None
        database.claim_qwen_analysis = None
        database.patch_analysis_result = None
        sys.modules["db"] = database
        import config
        assert config.GEMINI_API_KEY == ""
        import main
        import gemini_vision
        from fastapi.testclient import TestClient
        main.mt5_context.get_quote = lambda *args, **kwargs: None
        main.urlopen = lambda *args, **kwargs: io.BytesIO(b'{"models": []}')
        assert main.app is not None
        assert gemini_vision._client is None
        with TestClient(main.app, base_url="http://127.0.0.1:8811",
                        client=("127.0.0.1", 50000)) as client:
            assert client.get("/health").json() == {"status": "ok"}
            response = client.get("/services")
            assert response.status_code == 200
            assert response.json()["gemini"]["status"] == "not_configured"
            assert response.json()["mt5"]["status"] == "unavailable"
        assert gemini_vision._client is None
        print("server import succeeded without credentials")
    """)
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parent,
                            env={**os.environ, "GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "server import succeeded without credentials" in result.stdout


def offline_api_error(code, status="INVALID_ARGUMENT", reason=None):
    details = [] if reason is None else [{
        "@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason,
        "metadata": {"api_key": "private-secret", "url": "https://private.invalid/?key=private-secret"},
    }]
    return errors.APIError(code, {"error": {"code": code, "status": status,
        "message": "private-secret provider message", "details": details}})


@pytest.mark.parametrize("code,kind", [(400, "request"), (401, "authentication"), (402, "billing"),
    (403, "permission"), (404, "model"), (429, "quota"), (500, "service"), (503, "service")])
def test_safe_sdk_http_errors_have_fixed_messages_without_private_payload(code, kind):
    result = gemini_vision.safe_error(offline_api_error(code))
    assert result["kind"] == kind and result["code"] == code
    assert f"HTTP {code}" in result["message"]
    assert set(result) == {"kind", "code", "message"}
    assert "private" not in json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("reason,kind", [("API_KEY_INVALID", "authentication"), ("BILLING_DISABLED", "billing"),
    ("CONSUMER_INVALID", "project"), ("SERVICE_DISABLED", "project"), ("PROJECT_DELETED", "project"),
    ("API_KEY_SERVICE_BLOCKED", "permission"), ("API_KEY_HTTP_REFERRER_BLOCKED", "permission")])
def test_safe_error_uses_only_allowlisted_errorinfo_reasons(reason, kind):
    result = gemini_vision.safe_error(offline_api_error(403, reason=reason))
    assert result["kind"] == kind and result["code"] == 403
    assert "private" not in json.dumps(result, allow_nan=False)


def test_quota_does_not_claim_daily_or_per_minute_limit_from_raw_details():
    exc = offline_api_error(429, status="RESOURCE_EXHAUSTED")
    exc.details["error"]["message"] = "private-secret daily limit resets tomorrow after billing"
    exc.details["error"]["details"] = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [{"description": "private-secret daily limit"}]}]
    result = gemini_vision.safe_error(exc)
    assert result["kind"] == "quota"
    assert all(word not in result["message"] for word in ("daily", "tomorrow", "kunlik", "ertaga", "private-secret"))


@pytest.mark.parametrize("payload", [None, "private-secret", ["private-secret"],
    {"error": "private-secret"}, {"error": {"details": "private-secret"}},
    {"error": {"details": [{"reason": "BILLING_DISABLED", "message": "private-secret"}]}},
    {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.LocalizedMessage", "reason": "BILLING_DISABLED"}]}},
    {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": ["private-secret"]}]}},
    {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID private-secret"}]}},
])
def test_malformed_or_unrecognized_provider_details_cannot_escape(payload):
    exc = offline_api_error(403)
    exc.details = payload
    exc.status = "private-secret status"
    result = gemini_vision.safe_error(exc)
    assert result["kind"] == "permission"
    assert "private" not in json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("code", [True, False, "403", 403.0, None, 200, 399, 600, 10 ** 400, {}])
def test_untrusted_http_code_is_not_rendered_or_serialized(code):
    exc = offline_api_error(403)
    exc.code, exc.status, exc.details = code, "private-secret", "private-secret"
    result = gemini_vision.safe_error(exc)
    assert result["code"] is None and result["kind"] == "unknown"
    assert "HTTP" not in result["message"] and "private" not in json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("status,kind", [("UNAUTHENTICATED", "authentication"), ("PERMISSION_DENIED", "permission"),
    ("RESOURCE_EXHAUSTED", "quota"), ("NOT_FOUND", "model"), ("FAILED_PRECONDITION", "request")])
def test_allowlisted_rpc_status_is_a_safe_fallback_without_a_valid_http_code(status, kind):
    exc = offline_api_error(400, status=status)
    exc.code = None
    result = gemini_vision.safe_error(exc)
    assert result["kind"] == kind and result["code"] is None


@pytest.mark.parametrize("exception,kind", [
    (httpx.ReadTimeout("private-secret", request=httpx.Request("GET", "https://private.invalid/?key=private-secret")), "timeout"),
    (httpx.ConnectTimeout("private-secret"), "timeout"),
    (httpx.ConnectError("private-secret"), "network"),
    (httpx.ProxyError("private-secret"), "network"),
    (RuntimeError("private-secret"), "unknown"),
    (ValueError("private-secret response text"), "unknown"),
])
def test_transport_and_unknown_errors_never_expose_messages_or_request_urls(exception, kind):
    result = gemini_vision.safe_error(exception)
    assert result["kind"] == kind and result["code"] is None
    assert "private" not in json.dumps(result, allow_nan=False)


def test_only_typed_local_validation_failure_is_reported_as_response_error():
    with pytest.raises(gemini_vision.GeminiResponseError) as failure:
        gemini_vision._validate_response('{"confidence":"private-secret"}')
    result = gemini_vision.safe_error(failure.value)
    assert result["kind"] == "response" and result["code"] is None
    assert "private" not in json.dumps(result, allow_nan=False)
    assert gemini_vision.safe_error(ValueError("Gemini javobi private-secret"))["kind"] == "unknown"


def test_reason_scan_is_bounded_and_never_parses_exception_text():
    exc = offline_api_error(400)
    exc.message = "API_KEY_INVALID private-secret"
    exc.details["error"]["details"] = [{}] * 16 + [{
        "@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}]
    assert gemini_vision.safe_error(exc)["kind"] == "request"

    class UnsafeString(Exception):
        def __str__(self):
            pytest.fail("Raw exception text must not be inspected")

    assert gemini_vision.safe_error(UnsafeString())["kind"] == "unknown"
