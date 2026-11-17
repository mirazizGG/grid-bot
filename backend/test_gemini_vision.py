"""Gemini response boundaries; no provider requests or production database access."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

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
