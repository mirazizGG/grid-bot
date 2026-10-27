"""Focused regression tests. Uses an isolated temporary SQLite database."""
import datetime as dt
import io
import json
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import config

_temp_db = tempfile.TemporaryDirectory()
config.DB_PATH = str(__import__("pathlib").Path(_temp_db.name) / "test.db")

import db
import evaluator
import gemini_vision
import main
import mt5_context
import ollama_vision
from consensus import compute_consensus
from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def close_test_database():
    yield
    db.engine.dispose()
    _temp_db.cleanup()


def test_consensus_thresholds_and_genuine_hold():
    vision = {"signal": "Buy", "confidence": 0.65}
    assert compute_consensus({"signal": "Sell", "confidence": 0.8}, vision)["final_signal"] == "Buy"
    assert compute_consensus({"signal": "Buy", "confidence": 0.54}, vision)["final_signal"] == "Hold"
    assert compute_consensus({"signal": "Hold", "confidence": 0.4}, vision)["final_signal"] == "Buy"


def test_missing_mt5_forces_hold(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(main.mt5_context, "get_context", lambda *args: None)
    monkeypatch.setattr(main.custom_model, "analyze", lambda ctx: {"signal": "Hold", "confidence": 0.34})
    monkeypatch.setattr(main.gemini_vision, "analyze_image", lambda *a, **kw: {
        "signal": "Buy", "confidence": 0.9, "entry_price": 1, "tp_price": 2, "sl_price": 0.5})
    monkeypatch.setattr(main.ollama_vision, "analyze_image", lambda *a: {"signal": "Sell", "confidence": 0.8})
    monkeypatch.setattr(main, "save_analysis", lambda path, result: SimpleNamespace(to_dict=lambda: result))
    image = io.BytesIO()
    Image.new("RGB", (2, 2)).save(image, "PNG")
    response = TestClient(main.app).post("/analyze", files={"file": ("test.png", image.getvalue(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["final_signal"] == "Hold"
    assert result["entry_price"] is None and result["tp_price"] is None
    assert result["vision_ai"]["signal"] == "Buy"
    assert result["qwen_vision"]["signal"] == "Sell"
    assert result["warnings"]


def test_broker_precision():
    levels = mt5_context.compute_trade_levels({"ask": 1.08421, "bid": 1.08419, "atr_14": 0.0003, "digits": 5}, "Buy")
    assert levels == {"entry_price": 1.08421, "tp_price": 1.08481, "sl_price": 1.08391}


def test_qwen_response_validation_and_payload(monkeypatch):
    sent = []
    def fake_urlopen(request, timeout):
        sent.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"message": {"content": json.dumps({
            "signal": "Sell", "confidence": 0.6, "trend": "pastga", "reasoning": "test"})}}).encode())
    monkeypatch.setattr(ollama_vision, "urlopen", fake_urlopen)
    result = ollama_vision.analyze_image(b"image")
    assert result["signal"] == "Sell"
    assert sent[0]["messages"][0]["images"] == ["aW1hZ2U="]
    assert sent[0]["think"] is False and sent[0]["keep_alive"] == 0


@pytest.mark.parametrize("data", [{"signal": "BUY", "confidence": 0.7}, {"signal": "Buy", "confidence": 3}, {"signal": "Buy", "confidence": "nan"}])
def test_gemini_rejects_invalid_signal_and_confidence(monkeypatch, data):
    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: SimpleNamespace(text=json.dumps(data))))
    monkeypatch.setattr(gemini_vision, "_client", fake_client)
    with pytest.raises(ValueError):
        gemini_vision.analyze_image(b"image")


def test_qwen_failure_does_not_change_final_signal(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "UPLOAD_DIR", str(tmp_path))
    context = {"symbol": "EURUSD", "bid": 1.0, "ask": 1.1, "atr_14": 0.1, "digits": 2}
    monkeypatch.setattr(main.mt5_context, "get_context", lambda *args: context)
    monkeypatch.setattr(main.mt5_context, "get_quote", lambda *args: {"bid": 1.0, "ask": 1.1, "digits": 2})
    monkeypatch.setattr(main.custom_model, "analyze", lambda ctx: {"signal": "Buy", "confidence": 0.8})
    monkeypatch.setattr(main.gemini_vision, "analyze_image", lambda *a, **kw: {"signal": "Buy", "confidence": 0.8, "entry_type": "market"})
    def unavailable(*args):
        raise RuntimeError("Ollama unavailable")
    monkeypatch.setattr(main.ollama_vision, "analyze_image", unavailable)
    monkeypatch.setattr(main, "save_analysis", lambda path, result: SimpleNamespace(to_dict=lambda: result))
    image = io.BytesIO()
    Image.new("RGB", (2, 2)).save(image, "PNG")
    response = TestClient(main.app).post("/analyze", files={"file": ("test.png", image.getvalue(), "image/png")})
    assert response.status_code == 200
    assert response.json()["final_signal"] == "Buy"
    assert response.json()["qwen_vision"]["error"] == "Ollama unavailable"


def test_evaluation_uses_utc_complete_bars_and_rejects_gaps(monkeypatch):
    now = dt.datetime.utcnow().replace(second=0, microsecond=0)
    signal = now - dt.timedelta(minutes=4, seconds=30)
    first = signal.replace(second=0) + dt.timedelta(minutes=1)
    epoch = lambda t: int(t.replace(tzinfo=dt.timezone.utc).timestamp())
    dtype = [("time", "i8"), ("high", "f8"), ("low", "f8")]
    # First complete minute is benign; the next minute is missing, and a later
    # bar reaches TP. It must not be accepted because SL could occur in the gap.
    bars = np.array([(epoch(first), 1.1, 0.9), (epoch(first) + 120, 3.0, 0.9)], dtype=dtype)
    calls = []
    ticks = np.array([(int((epoch(signal) + 1) * 1000), 1.0, 1.1)], dtype=[("time_msc", "i8"), ("bid", "f8"), ("ask", "f8")])
    fake = SimpleNamespace(initialize=lambda: True, shutdown=lambda: None, TIMEFRAME_M1=1, COPY_TICKS_INFO=1,
                           copy_ticks_range=lambda *a: ticks,
                           copy_rates_range=lambda symbol, tf, start, end: calls.append((start, end)) or bars)
    monkeypatch.setattr(evaluator, "mt5", fake)
    session = db.SessionLocal()
    row = db.Analysis(timestamp=signal, final_signal="Buy", entry_price=1.0, tp_price=2.0, sl_price=0.5,
                      raw_result=json.dumps({"market_context": {"symbol": "EURUSD", "server_utc_offset_hours": None}, "evaluation_plan": "market"}))
    session.add(row)
    session.commit()
    row_id = row.id
    session.close()
    result = evaluator.evaluate_pending(horizon_hours=1)
    assert calls[0][0] == epoch(first)
    assert result["by_outcome"]["NO_DATA"] == 1
    session = db.SessionLocal()
    saved = session.get(db.Analysis, row_id)
    assert saved.outcome_version == 3
    session.close()


def test_invalid_horizon():
    with pytest.raises(ValueError):
        evaluator.evaluate_pending(float("nan"))


def test_first_partial_minute_checks_ticks_in_order(monkeypatch):
    start = dt.datetime(2026, 1, 1, 10, 0, 20)
    epoch_ms = int(start.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    dtype = [("time_msc", "i8"), ("bid", "f8"), ("ask", "f8")]
    ticks = np.array([(epoch_ms - 1000, 2.1, 2.2), (epoch_ms + 1000, 0.4, 0.5),
                      (epoch_ms + 2000, 2.2, 2.3)], dtype=dtype)
    monkeypatch.setattr(evaluator, "mt5", SimpleNamespace(COPY_TICKS_INFO=1, copy_ticks_range=lambda *a: ticks))
    assert evaluator._partial_minute_outcome("XAUUSD", "Buy", 2, 0.5, start, start + dt.timedelta(seconds=40)) == "SL"
    monkeypatch.setattr(evaluator, "mt5", SimpleNamespace(COPY_TICKS_INFO=1, copy_ticks_range=lambda *a: None))
    assert evaluator._partial_minute_outcome("XAUUSD", "Buy", 2, 0.5, start, start + dt.timedelta(seconds=40)) == "NO_DATA"


def test_current_candle_is_excluded_from_model_context(monkeypatch):
    dtype = [("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"), ("tick_volume", "i8")]
    bars = np.array([(1700000000 + i * 900, 100 + i, 102 + i, 99 + i, 101 + i, 1) for i in range(100)], dtype=dtype)
    positions = []
    fake = SimpleNamespace(initialize=lambda: True, shutdown=lambda: None, symbol_select=lambda *a: True,
                           symbol_info_tick=lambda *a: SimpleNamespace(bid=200, ask=201, time=time.time()),
                           symbol_info=lambda *a: SimpleNamespace(digits=2),
                           copy_rates_from_pos=lambda symbol, tf, position, count: positions.append(position) or bars)
    monkeypatch.setattr(mt5_context, "mt5", fake)
    monkeypatch.setattr(mt5_context, "TIMEFRAME_MAP", {"M15": 1, "H1": 2, "H4": 3})
    context = mt5_context.get_context("XAUUSD", "M15")
    assert positions == [1, 1, 1]
    assert context["ask"] == 201 and context["digits"] == 2


def test_quote_rejects_stale_and_invalid_prices():
    info = SimpleNamespace(digits=5)
    assert mt5_context._checked_quote(SimpleNamespace(bid=1, ask=1.1, time=time.time() - 100), info) is None
    assert mt5_context._checked_quote(SimpleNamespace(bid=1, ask=0.9, time=time.time()), info) is None
    assert mt5_context._checked_quote(SimpleNamespace(bid=1, ask=1.1, time=time.time()), info) is not None


@pytest.mark.parametrize("entry_type,new_ask,expected", [("wait_pullback", 1.1, "Hold"),
                                                         ("market", 1.2, "Hold"),
                                                         ("market", 1.11, "Buy")])
def test_rechecks_quote_and_conditional_entry(monkeypatch, tmp_path, entry_type, new_ask, expected):
    monkeypatch.setattr(main, "UPLOAD_DIR", str(tmp_path))
    ctx = {"symbol": "EURUSD", "bid": 1.0, "ask": 1.1, "atr_14": 0.1, "digits": 2}
    monkeypatch.setattr(main.mt5_context, "get_context", lambda *a: ctx)
    monkeypatch.setattr(main.mt5_context, "get_quote", lambda *a: {"bid": new_ask - 0.1, "ask": new_ask, "digits": 2})
    monkeypatch.setattr(main.custom_model, "analyze", lambda *a: {"signal": "Buy", "confidence": 0.8})
    monkeypatch.setattr(main.gemini_vision, "analyze_image", lambda *a, **kw: {"signal": "Buy", "confidence": 0.8, "entry_type": entry_type})
    monkeypatch.setattr(main.ollama_vision, "analyze_image", lambda *a: {"signal": "Sell", "confidence": 0.8})
    monkeypatch.setattr(main, "save_analysis", lambda path, result: SimpleNamespace(to_dict=lambda: result))
    image = io.BytesIO()
    Image.new("RGB", (2, 2)).save(image, "PNG")
    response = TestClient(main.app).post("/analyze", files={"file": ("test.png", image.getvalue(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["final_signal"] == expected
    if expected == "Buy":
        assert result["entry_price"] == new_ask and result["evaluation_plan"] == "market"
    else:
        assert result["entry_price"] is None and "evaluation_plan" not in result


def test_old_or_conditional_records_are_not_treated_as_executed():
    row = SimpleNamespace(final_signal="Buy", entry_price=1, tp_price=2, sl_price=0.5)
    assert evaluator._plan(row, {"market_context": {"symbol": "EURUSD"}})[0] is None


def test_model_comparison_separate_plans_and_skip_conditions():
    ctx = {"symbol": "EURUSD", "bid": 1.0, "ask": 1.01, "atr_14": 0.1, "digits": 2}
    quote = {"bid": 1.0, "ask": 1.01, "digits": 2}
    gemini = {"signal": "Buy", "entry_type": "market"}
    qwen = {"signal": "Sell"}
    plans = evaluator.create_model_comparison(ctx, quote, gemini, qwen)
    assert plans["gemini"]["entry_price"] == 1.01 and plans["gemini"]["tp_price"] == 1.21
    assert plans["qwen"]["entry_price"] == 1.0 and plans["qwen"]["tp_price"] == 0.8
    assert plans["gemini"]["outcome"] is None and plans["qwen"]["outcome"] is None
    assert evaluator.create_model_comparison(ctx, quote, {"signal": "Buy", "entry_type": "wait_breakout"}, {"error": "down"})["gemini"]["outcome"] == "SKIP"
    assert evaluator.create_model_comparison(ctx, None, gemini, qwen)["qwen"]["outcome"] == "SKIP"
    assert evaluator.create_model_comparison(ctx, {**quote, "ask": 1.2}, gemini, qwen)["gemini"]["outcome"] == "SKIP"


def test_model_results_evaluate_after_finalized_consensus_and_are_idempotent(monkeypatch):
    ctx = {"symbol": "EURUSD", "bid": 1.0, "ask": 1.01, "atr_14": 0.1, "digits": 2}
    plans = evaluator.create_model_comparison(ctx, {"bid": 1.0, "ask": 1.01, "digits": 2},
                                               {"signal": "Buy", "entry_type": "market"}, {"signal": "Sell"})
    row = db.Analysis(timestamp=dt.datetime.utcnow(), final_signal="Hold", outcome="SKIP", outcome_version=3,
                      raw_result=json.dumps({"market_context": ctx, "model_comparison": plans}))
    session = db.SessionLocal()
    session.add(row)
    session.commit()
    row_id = row.id
    session.close()
    monkeypatch.setattr(evaluator, "mt5", SimpleNamespace(initialize=lambda: True, shutdown=lambda: None))
    calls = []
    def fake_path(symbol, direction, tp, sl, timestamp, now, horizon, offset):
        calls.append((direction, horizon, offset))
        return "TP" if direction == "Buy" else "SL"
    monkeypatch.setattr(evaluator, "_evaluate_price_path", fake_path)
    evaluator.evaluate_pending()
    session = db.SessionLocal()
    saved = session.get(db.Analysis, row_id)
    data = json.loads(saved.raw_result)
    assert saved.outcome == "SKIP"
    assert data["model_comparison"]["gemini"]["outcome"] == "TP"
    assert data["model_comparison"]["qwen"]["outcome"] == "SL"
    stats = evaluator.summarize_models([saved])
    assert stats["gemini"]["tp"] == 1 and stats["qwen"]["sl"] == 1
    session.close()
    first_calls = list(calls)
    evaluator.evaluate_pending()
    assert [(d, h, o) for d, h, o in first_calls if o == 0] == [("Buy", 24.0, 0), ("Sell", 24.0, 0)]
    assert calls == first_calls
