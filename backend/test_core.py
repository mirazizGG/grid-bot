"""Focused regression tests. Uses an isolated temporary SQLite database."""
import datetime as dt
import io
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import db
import evaluator
import gemini_vision
import main
import mt5_context
import ollama_vision
from consensus import compute_consensus
from fastapi.testclient import TestClient


def local_client():
    return TestClient(main.app, base_url="http://127.0.0.1:8811", client=("127.0.0.1", 50000),
                      headers={"X-Grid-Bot-Request": "1"})


def test_consensus_thresholds_and_genuine_hold():
    vision = {"signal": "Buy", "confidence": 0.65}
    assert compute_consensus({"signal": "Sell", "confidence": 0.8}, vision)["final_signal"] == "Buy"
    assert compute_consensus({"signal": "Buy", "confidence": 0.54}, vision)["final_signal"] == "Hold"
    assert compute_consensus({"signal": "Hold", "confidence": 0.4}, vision)["final_signal"] == "Buy"
    hold = compute_consensus({"signal": "Hold", "confidence": 0.8},
                             {"signal": "Hold", "confidence": 0.9, "entry_price": 100})
    assert hold["final_confidence"] is None and hold["entry_price"] is None


def test_missing_mt5_forces_hold(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(main.mt5_context, "get_context", lambda *args: None)
    monkeypatch.setattr(main.custom_model, "analyze", lambda ctx: {"signal": "Hold", "confidence": 0.34})
    monkeypatch.setattr(main.gemini_vision, "analyze_image", lambda *a, **kw: {
        "signal": "Buy", "confidence": 0.9, "entry_price": 1, "tp_price": 2, "sl_price": 0.5})
    monkeypatch.setattr(main.ollama_vision, "analyze_image", lambda *a, **kw: {"signal": "Sell", "confidence": 0.8})
    monkeypatch.setattr(main, "save_analysis", lambda path, result: SimpleNamespace(to_dict=lambda: result))
    image = io.BytesIO()
    Image.new("RGB", (2, 2)).save(image, "PNG")
    response = local_client().post("/analyze", files={"file": ("test.png", image.getvalue(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["final_signal"] == "Hold"
    assert result["entry_price"] is None and result["tp_price"] is None
    assert result["vision_ai"]["signal"] == "Buy"
    assert result["qwen_vision"]["status"] == "not_requested"
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


def test_primary_signal_does_not_wait_for_optional_qwen(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "UPLOAD_DIR", str(tmp_path))
    context = {"symbol": "EURUSD", "bid": 1.0, "ask": 1.1, "atr_14": 0.1, "digits": 2}
    monkeypatch.setattr(main.mt5_context, "get_context", lambda *args: context)
    monkeypatch.setattr(main.mt5_context, "get_quote", lambda *args: {"bid": 1.0, "ask": 1.1, "digits": 2})
    monkeypatch.setattr(main.custom_model, "analyze", lambda ctx: {"signal": "Buy", "confidence": 0.8})
    monkeypatch.setattr(main.gemini_vision, "analyze_image", lambda *a, **kw: {"signal": "Buy", "confidence": 0.8, "entry_type": "market"})
    calls = []
    def unavailable(*args):
        calls.append(1)
        raise RuntimeError("Ollama unavailable")
    monkeypatch.setattr(main.ollama_vision, "analyze_image", unavailable)
    monkeypatch.setattr(main, "save_analysis", lambda path, result: SimpleNamespace(to_dict=lambda: result))
    image = io.BytesIO()
    Image.new("RGB", (2, 2)).save(image, "PNG")
    response = local_client().post("/analyze", files={"file": ("test.png", image.getvalue(), "image/png")})
    assert response.status_code == 200
    assert response.json()["final_signal"] == "Buy"
    assert response.json()["qwen_vision"]["status"] == "not_requested"
    assert calls == []


def test_evaluation_rejects_missing_tick_interval(monkeypatch):
    now = dt.datetime.utcnow().replace(second=0, microsecond=0)
    signal = now - dt.timedelta(minutes=4, seconds=30)
    epoch = lambda t: int(t.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    dtype = [("time_msc", "i8"), ("bid", "f8"), ("ask", "f8")]
    ticks = np.array([(epoch(signal) + 1000, 1.0, 1.1),
                      (epoch(signal) + 120000, 3.0, 3.1)], dtype=dtype)
    calls = []
    def copy_ticks(symbol, start, end, flags):
        calls.append((start, end))
        return ticks
    fake = SimpleNamespace(initialize=lambda: True, shutdown=lambda: None, COPY_TICKS_INFO=1,
                           copy_ticks_range=copy_ticks)
    monkeypatch.setattr(evaluator, "mt5", fake)
    session = db.SessionLocal()
    row = db.Analysis(timestamp=signal, final_signal="Buy", entry_price=1.0, tp_price=2.0, sl_price=0.5,
                      raw_result=json.dumps({"market_context": {"symbol": "EURUSD"}, "evaluation_plan": "market"}))
    session.add(row)
    session.commit()
    row_id = row.id
    session.close()
    result = evaluator.evaluate_pending()
    assert calls[0][0] == signal.replace(tzinfo=dt.timezone.utc)
    assert result["by_outcome"]["NO_DATA"] == 1
    session = db.SessionLocal()
    saved = session.get(db.Analysis, row_id)
    assert saved.outcome_version == evaluator.OUTCOME_VERSION
    session.close()


def test_sell_outcome_uses_ask_and_horizon_bounds(monkeypatch):
    start = dt.datetime(2026, 1, 1, 10, 0)
    epoch = int(start.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    dtype = [("time_msc", "i8"), ("bid", "f8"), ("ask", "f8")]
    # Bid touches TP but ask does not; later the ask reaches SL.
    ticks = np.array([(epoch - 1, 1.0, 2.1), (epoch + 1000, 0.9, 1.1),
                      (epoch + 2000, 1.9, 2.1)], dtype=dtype)
    monkeypatch.setattr(evaluator, "mt5", SimpleNamespace(COPY_TICKS_INFO=1, copy_ticks_range=lambda *a: ticks))
    assert evaluator._evaluate_price_path("EURUSD", "Sell", 1.0, 2.0, start,
                                          start + dt.timedelta(seconds=10), 1, 0) == "SL"
    # A tick exactly at the end belongs outside the evaluation window.
    ticks = np.array([(epoch + 1000, 1.1, 1.2), (epoch + 10000, 0.9, 0.95)], dtype=dtype)
    assert evaluator._evaluate_price_path("EURUSD", "Sell", 1.0, 2.0, start,
                                          start + dt.timedelta(seconds=10), 1, 0) == "OPEN"


def test_empty_short_tick_interval_is_unknown(monkeypatch):
    start = dt.datetime(2026, 1, 1, 10, 0)
    monkeypatch.setattr(evaluator, "mt5", SimpleNamespace(COPY_TICKS_INFO=1, copy_ticks_range=lambda *a: []))
    assert evaluator._tick_path_outcome("EURUSD", "Buy", 2.0, 0.5, start, start + dt.timedelta(seconds=10)) == "NO_DATA"


def test_legacy_results_are_excluded_from_tick_statistics():
    session = db.SessionLocal()
    old = db.Analysis(timestamp=dt.datetime.utcnow(), outcome="TP", outcome_kind="signal", outcome_version=3,
                      raw_result=json.dumps({"model_comparison": {"gemini": {"version": 1, "outcome": "TP"}}}))
    session.add(old)
    session.commit()
    stats = evaluator.summarize(session)["signal"]
    assert stats["tp"] == 0 and stats["legacy"] >= 1
    assert evaluator.summarize_models([old])["gemini"]["tp"] == 0
    session.delete(old)
    session.commit()
    session.close()


def test_tick_same_millisecond_conflict_is_conservative(monkeypatch):
    start = dt.datetime(2026, 1, 1, 10, 0)
    epoch = int(start.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    dtype = [("time_msc", "i8"), ("bid", "f8"), ("ask", "f8")]
    ticks = np.array([(epoch + 1000, 2.0, 2.1), (epoch + 1000, 0.4, 0.5)], dtype=dtype)
    monkeypatch.setattr(evaluator, "mt5", SimpleNamespace(COPY_TICKS_INFO=1, copy_ticks_range=lambda *a: ticks))
    assert evaluator._tick_path_outcome("EURUSD", "Buy", 2.0, 0.5, start, start + dt.timedelta(seconds=5)) == "SL"


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
    positions = []
    now = time.time()
    def copy_rates(symbol, tf, position, count):
        positions.append(position)
        seconds = {1: 900, 2: 3600, 3: 14400}[tf]
        current = int(now // seconds) * seconds
        return np.array([(current + (i - 100) * seconds, 100 + i, 102 + i, 99 + i, 101 + i, 1)
                         for i in range(101)], dtype=dtype)
    fake = SimpleNamespace(initialize=lambda: True, shutdown=lambda: None, symbol_select=lambda *a: True,
                           symbol_info_tick=lambda *a: SimpleNamespace(bid=200, ask=201, time=now),
                           symbol_info=lambda *a: SimpleNamespace(digits=2),
                           copy_rates_from_pos=copy_rates)
    monkeypatch.setattr(mt5_context, "mt5", fake)
    monkeypatch.setattr(mt5_context, "TIMEFRAME_MAP", {"M15": 1, "H1": 2, "H4": 3})
    context = mt5_context.get_context("XAUUSD", "M15")
    assert positions == [0, 0, 0]
    assert context["ask"] == 201 and context["digits"] == 2
    assert context["last_close"] == 200  # forming candle close=201 was excluded


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
    monkeypatch.setattr(main.ollama_vision, "analyze_image", lambda *a, **kw: {"signal": "Sell", "confidence": 0.8})
    monkeypatch.setattr(main, "save_analysis", lambda path, result: SimpleNamespace(to_dict=lambda: result))
    image = io.BytesIO()
    Image.new("RGB", (2, 2)).save(image, "PNG")
    response = local_client().post("/analyze", files={"file": ("test.png", image.getvalue(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["final_signal"] == expected
    if expected == "Buy":
        assert result["entry_price"] == new_ask and result["evaluation_plan"] == "market"
        assert result["evaluation_parameters"] == {"horizon_hours": 24.0, "utc_offset": 0}
    else:
        assert result["entry_price"] is None and "evaluation_plan" not in result


def test_old_or_conditional_records_are_not_treated_as_executed():
    row = SimpleNamespace(final_signal="Buy", entry_price=1, tp_price=2, sl_price=0.5)
    assert evaluator._plan(row, {"market_context": {"symbol": "EURUSD"}})[0] is None
    stale = {"market_context": {"symbol": "EURUSD", "timeframe": "M15"},
             "image_age_minutes": 16, "evaluation_plan": "market"}
    assert evaluator._plan(row, stale)[0] is None


@pytest.mark.parametrize("age,expected", [(3600, "Hold"), (0, "Buy"), (None, "Buy")])
def test_screenshot_freshness_controls_signal_and_plans(monkeypatch, tmp_path, age, expected):
    monkeypatch.setattr(main, "UPLOAD_DIR", str(tmp_path))
    ctx = {"symbol": "EURUSD", "timeframe": "M15", "bid": 1.0, "ask": 1.01, "atr_14": 0.1, "digits": 2}
    monkeypatch.setattr(main.mt5_context, "get_context", lambda *a: ctx)
    monkeypatch.setattr(main.mt5_context, "get_quote", lambda *a: {"bid": 1.0, "ask": 1.01, "digits": 2})
    monkeypatch.setattr(main.custom_model, "analyze", lambda *a: {"signal": "Buy", "confidence": 0.8})
    monkeypatch.setattr(main.gemini_vision, "analyze_image", lambda *a, **kw: {"signal": "Buy", "confidence": 0.8, "entry_type": "market"})
    monkeypatch.setattr(main.ollama_vision, "analyze_image", lambda *a, **kw: {"signal": "Sell", "confidence": 0.8})
    monkeypatch.setattr(main, "save_analysis", lambda path, result: SimpleNamespace(to_dict=lambda: result))
    image = io.BytesIO()
    Image.new("RGB", (2, 2)).save(image, "PNG")
    data = {} if age is None else {"file_modified_ms": str((time.time() - age) * 1000)}
    response = local_client().post("/analyze", data=data, files={"file": ("test.png", image.getvalue(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["final_signal"] == expected
    if age == 3600:
        assert all(result[key] is None for key in ("entry_price", "tp_price", "sl_price"))
        assert "evaluation_plan" not in result
        assert all(plan["outcome"] == "SKIP" for plan in result["model_comparison"].values())
        assert result["image_freshness"] == "stale_file_timestamp"
    if age is None:
        assert result["image_freshness"] == "unknown"
        assert any("noma'lum" in warning for warning in result["warnings"])


def test_local_api_rejects_cross_site_and_large_uploads(monkeypatch):
    client = local_client()
    assert client.get("/health").status_code == 200
    assert client.get("/history", headers={"Origin": "https://example.com"}).status_code == 403
    assert client.post("/evaluate", headers={"Origin": "https://example.com"}).status_code == 403
    assert client.get("/health", headers={"Host": "attacker.example"}).status_code == 403
    assert client.post("/evaluate", headers={"X-Grid-Bot-Request": ""}).status_code == 403
    remote = TestClient(main.app, base_url="http://127.0.0.1:8811", client=("192.0.2.1", 50000))
    assert remote.get("/health").status_code == 403
    monkeypatch.setattr(main, "MAX_IMAGE_BYTES", 100)
    image = io.BytesIO()
    Image.new("RGB", (50, 50), "red").save(image, "PNG")
    assert client.post("/analyze", files={"file": ("test.png", image.getvalue(), "image/png")}).status_code == 413


def test_invalid_image_time_is_not_treated_as_fresh():
    for value in (None, float("nan"), float("inf"), (time.time() + 60) * 1000):
        assert main._image_age_minutes(value) is None


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


@pytest.mark.parametrize("horizon,offset", [(1, None), (24, 3), (24, -3), (float("inf"), None)])
def test_request_cannot_change_frozen_evaluation_policy(monkeypatch, horizon, offset):
    monkeypatch.setattr(evaluator, "mt5", SimpleNamespace(initialize=lambda: pytest.fail("must validate before MT5")))
    with pytest.raises(ValueError):
        evaluator.evaluate_pending(horizon, offset)
    assert local_client().post(f"/evaluate?horizon_hours={horizon}&utc_offset={offset or 0}").status_code == 400


def test_old_final_outcomes_stay_legacy_and_pending_records_store_parameters(monkeypatch):
    data = {"market_context": {"symbol": "EURUSD"}, "evaluation_plan": "market"}
    session = db.SessionLocal()
    old = db.Analysis(timestamp=dt.datetime.utcnow() - dt.timedelta(hours=2), final_signal="Buy",
                      entry_price=1, tp_price=2, sl_price=0.5, outcome="EXPIRED", outcome_kind="signal",
                      outcome_version=4, raw_result=json.dumps(data))
    pending = db.Analysis(timestamp=dt.datetime.utcnow() - dt.timedelta(hours=2), final_signal="Buy",
                          entry_price=1, tp_price=2, sl_price=0.5, raw_result=json.dumps(data))
    session.add_all([old, pending])
    session.commit()
    old_id, pending_id = old.id, pending.id
    session.close()
    monkeypatch.setattr(evaluator, "mt5", SimpleNamespace(initialize=lambda: True, shutdown=lambda: None))
    monkeypatch.setattr(evaluator, "_tick_path_outcome", lambda *a: "OPEN")
    assert evaluator.evaluate_pending()["by_outcome"] == {"OPEN": 1}
    session = db.SessionLocal()
    assert session.get(db.Analysis, old_id).outcome_version == 4
    pending = session.get(db.Analysis, pending_id)
    assert pending.outcome_version == 5 and pending.outcome == "OPEN"
    assert json.loads(pending.raw_result)["evaluation_parameters"] == evaluator.evaluation_parameters()
    assert evaluator.summarize(session)["signal"]["legacy"] == 1
    session.close()


def test_nonfinite_gemini_response_never_reaches_history(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(main.mt5_context, "get_context", lambda *a: None)
    monkeypatch.setattr(main.custom_model, "analyze", lambda *a: {"signal": "Hold", "confidence": 0.34})
    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: SimpleNamespace(
        text='{"signal":"Buy","confidence":0.9,"trend":"up","reasoning":"test","entry_price":1e309}')))
    monkeypatch.setattr(gemini_vision, "_client", fake_client)
    image = io.BytesIO()
    Image.new("RGB", (2, 2)).save(image, "PNG")
    client = local_client()
    assert client.post("/analyze", files={"file": ("test.png", image.getvalue(), "image/png")}).status_code == 502
    assert client.get("/history").json() == []
    assert list(tmp_path.iterdir()) == []


def _optional_qwen_record(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "UPLOAD_DIR", str(tmp_path))
    filename = "a" * 32 + ".png"
    Image.new("RGB", (2, 2)).save(tmp_path / filename)
    ctx = {"symbol": "EURUSD", "bid": 1.0, "ask": 1.01, "atr_14": 0.1, "digits": 2}
    monkeypatch.setattr(mt5_context, "get_quote", lambda *a: ctx)
    custom = {"signal": "Buy", "confidence": 0.8}
    vision = {"signal": "Buy", "confidence": 0.8, "entry_type": "market"}
    result = compute_consensus(custom, vision)
    result.update(mt5_context.compute_trade_levels(ctx, "Buy"))
    result.update(market_context=ctx, publication_quote=ctx, evaluation_plan="market",
                  evaluation_parameters=evaluator.evaluation_parameters(), image_freshness="recent_file_timestamp",
                  qwen_vision={"status": "not_requested"},
                  model_comparison=evaluator.create_model_comparison(ctx, ctx, vision, {"status": "not_requested"}))
    return db.save_analysis("/uploads/" + filename, result)


def test_optional_qwen_uses_own_publication_and_preserves_primary(monkeypatch, tmp_path):
    row = _optional_qwen_record(monkeypatch, tmp_path)
    primary = row.to_dict()
    monkeypatch.setattr(ollama_vision, "analyze_image", lambda *a, **kw: {"signal": "Sell", "confidence": 0.7})
    monkeypatch.setattr(mt5_context, "get_quote", lambda *a: {"bid": 1.001, "ask": 1.011, "digits": 3})
    client = local_client()
    response = client.post(f"/history/{row.id}/qwen")
    assert response.status_code == 200
    result = response.json()
    assert result["detail"]["qwen_vision"]["status"] == "completed"
    assert result["final_signal"] == primary["final_signal"] == "Buy"
    assert result["detail"]["model_comparison"]["gemini"] == primary["detail"]["model_comparison"]["gemini"]
    plan = result["detail"]["model_comparison"]["qwen"]
    assert plan["entry_price"] == 1.001
    assert plan["timestamp"] >= result["timestamp"]
    assert plan["horizon_hours"] == 24 and plan["utc_offset"] == 0
    monkeypatch.setattr(evaluator, "mt5", SimpleNamespace(initialize=lambda: True, shutdown=lambda: None))
    starts = []
    def evaluate_path(symbol, direction, tp, sl, timestamp, now, horizon, offset):
        starts.append((direction, timestamp))
        return "OPEN"
    monkeypatch.setattr(evaluator, "_evaluate_price_path", evaluate_path)
    evaluator.evaluate_pending()
    expected_start = dt.datetime.fromisoformat(plan["timestamp"].replace("Z", "+00:00")).replace(tzinfo=None)
    assert ("Sell", expected_start) in starts
    monkeypatch.setattr(ollama_vision, "analyze_image", lambda *a, **kw: pytest.fail("completed analysis must not rerun"))
    saved_plan = client.post(f"/history/{row.id}/qwen").json()["detail"]["model_comparison"]["qwen"]
    assert saved_plan["timestamp"] == plan["timestamp"] and saved_plan["outcome"] == "OPEN"


def test_optional_qwen_failure_can_retry_without_changing_primary(monkeypatch, tmp_path):
    row = _optional_qwen_record(monkeypatch, tmp_path)
    def fail(*a, **kw):
        raise RuntimeError("unavailable")
    monkeypatch.setattr(ollama_vision, "analyze_image", fail)
    client = local_client()
    result = client.post(f"/history/{row.id}/qwen").json()
    assert result["final_signal"] == "Buy"
    assert result["detail"]["qwen_vision"]["status"] == "error"
    assert result["detail"]["model_comparison"]["qwen"]["outcome"] == "SKIP"
    monkeypatch.setattr(ollama_vision, "analyze_image", lambda *a, **kw: {"signal": "Hold", "confidence": 0.5})
    assert client.post(f"/history/{row.id}/qwen").json()["detail"]["qwen_vision"]["status"] == "completed"
    assert client.post(f"/history/{row.id}/qwen", headers={"X-Grid-Bot-Request": ""}).status_code == 403
    assert client.post("/history/999999/qwen").status_code == 404


def test_atomic_evaluation_patch_preserves_concurrent_qwen_result(monkeypatch, tmp_path):
    row = _optional_qwen_record(monkeypatch, tmp_path)
    session = db.SessionLocal()
    stale_snapshot = session.get(db.Analysis, row.id)
    assert json.loads(stale_snapshot.raw_result)["qwen_vision"]["status"] == "not_requested"
    other = db.SessionLocal()
    db.patch_analysis_result(other, row.id, {"$.qwen_vision": {"signal": "Sell", "status": "completed"}})
    other.commit()
    other.close()
    db.patch_analysis_result(session, row.id, {"$.model_comparison.gemini.outcome": "TP"})
    session.commit()
    session.expire_all()
    saved = json.loads(session.get(db.Analysis, row.id).raw_result)
    assert saved["qwen_vision"]["status"] == "completed"
    assert saved["model_comparison"]["gemini"]["outcome"] == "TP"
    session.close()


def test_simultaneous_qwen_requests_do_not_run_duplicate_inference(monkeypatch, tmp_path):
    row = _optional_qwen_record(monkeypatch, tmp_path)
    started, release = threading.Event(), threading.Event()
    calls = []
    def analyze(*a, **kw):
        calls.append(1)
        started.set()
        assert release.wait(5), "test must release optional inference"
        return {"signal": "Sell", "confidence": 0.7}
    monkeypatch.setattr(ollama_vision, "analyze_image", analyze)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: local_client().post(f"/history/{row.id}/qwen"))
        try:
            assert started.wait(5)
            duplicate = local_client().post(f"/history/{row.id}/qwen")
            assert duplicate.status_code == 202
            assert duplicate.json()["detail"]["qwen_vision"]["status"] == "running"
        finally:
            release.set()
        assert future.result(timeout=5).status_code == 200
    assert calls == [1]


@pytest.mark.parametrize("stale,quote", [(True, {"bid": 1.0, "ask": 1.01, "digits": 2}),
                                       (False, None),
                                       (False, {"bid": 1.2, "ask": 1.21, "digits": 2})])
def test_delayed_qwen_is_not_scored_with_stale_inputs(monkeypatch, tmp_path, stale, quote):
    row = _optional_qwen_record(monkeypatch, tmp_path)
    if stale:
        session = db.SessionLocal()
        db.patch_analysis_result(session, row.id, {"$.image_freshness": "stale_file_timestamp"})
        session.commit()
        session.close()
    monkeypatch.setattr(mt5_context, "get_quote", lambda *a: quote)
    monkeypatch.setattr(ollama_vision, "analyze_image", lambda *a, **kw: {"signal": "Sell", "confidence": 0.7})
    response = local_client().post(f"/history/{row.id}/qwen")
    assert response.status_code == 200
    assert response.json()["detail"]["model_comparison"]["qwen"]["outcome"] == "SKIP"
    assert response.json()["final_signal"] == "Buy"


def test_expired_qwen_claim_recovers_and_stale_worker_cannot_overwrite(monkeypatch, tmp_path):
    row = _optional_qwen_record(monkeypatch, tmp_path)
    clock = [100.0]
    monkeypatch.setattr(db.time, "time", lambda: clock[0])
    first, claimed = db.claim_qwen_analysis(row.id, "test", lease_seconds=2)
    assert claimed
    old_request = first["detail"]["qwen_vision"]["request_id"]
    clock[0] = 101
    assert db.claim_qwen_analysis(row.id, "test", lease_seconds=2)[1] is False
    clock[0] = 103
    second, claimed = db.claim_qwen_analysis(row.id, "test", lease_seconds=2)
    assert claimed
    new_request = second["detail"]["qwen_vision"]["request_id"]
    assert new_request != old_request
    session = db.SessionLocal()
    assert not db.patch_analysis_result(session, row.id, {"$.qwen_vision": {"status": "completed", "signal": "Buy"}},
                                        qwen_request_id=old_request)
    assert db.patch_analysis_result(session, row.id, {"$.qwen_vision": {"status": "completed", "signal": "Sell"}},
                                   qwen_request_id=new_request)
    session.commit()
    assert json.loads(session.get(db.Analysis, row.id).raw_result)["qwen_vision"]["signal"] == "Sell"
    session.close()


def test_old_context_cannot_create_qwen_plan_even_if_price_returns(monkeypatch, tmp_path):
    row = _optional_qwen_record(monkeypatch, tmp_path)
    session = db.SessionLocal()
    saved = session.get(db.Analysis, row.id)
    saved.timestamp = dt.datetime.utcnow() - dt.timedelta(days=2)
    session.commit()
    session.close()
    monkeypatch.setattr(ollama_vision, "analyze_image", lambda *a, **kw: {"signal": "Sell", "confidence": 0.7})
    response = local_client().post(f"/history/{row.id}/qwen")
    assert response.status_code == 200
    assert response.json()["detail"]["model_comparison"]["qwen"]["outcome"] == "SKIP"


def test_slow_replay_does_not_hold_sqlite_writer_lock(monkeypatch, tmp_path):
    first = _optional_qwen_record(monkeypatch, tmp_path)
    second = _optional_qwen_record(monkeypatch, tmp_path)
    monkeypatch.setattr(evaluator, "mt5", SimpleNamespace(initialize=lambda: True, shutdown=lambda: None))
    calls = []
    def evaluate_path(*args):
        calls.append(1)
        if len(calls) == 2:
            # A separate connection can still write while earlier outcomes are
            # computed and a later broker replay is in progress.
            other = db.SessionLocal()
            db.patch_analysis_result(other, first.id, {"$.qwen_vision.status": "completed"})
            other.commit()
            other.close()
        return "OPEN"
    monkeypatch.setattr(evaluator, "_evaluate_price_path", evaluate_path)
    evaluator.evaluate_pending()
    assert len(calls) >= 2
    session = db.SessionLocal()
    assert json.loads(session.get(db.Analysis, first.id).raw_result)["qwen_vision"]["status"] == "completed"
    assert session.get(db.Analysis, second.id).outcome == "OPEN"
    session.close()


def test_primary_timestamp_matches_comparison_and_quote_follows_image_save(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "UPLOAD_DIR", str(tmp_path))
    ctx = {"symbol": "EURUSD", "timeframe": "M15", "bid": 1.0, "ask": 1.01, "atr_14": 0.1, "digits": 2}
    monkeypatch.setattr(mt5_context, "get_context", lambda *a: ctx)
    monkeypatch.setattr(main.custom_model, "analyze", lambda *a: {"signal": "Buy", "confidence": 0.8})
    monkeypatch.setattr(gemini_vision, "analyze_image", lambda *a, **kw: {"signal": "Buy", "confidence": 0.8, "entry_type": "market"})
    image = io.BytesIO()
    Image.new("RGB", (2, 2)).save(image, "PNG")
    order = []
    original_save = Image.Image.save
    def save_image(self, *args, **kwargs):
        order.append("save")
        return original_save(self, *args, **kwargs)
    monkeypatch.setattr(Image.Image, "save", save_image)
    monkeypatch.setattr(mt5_context, "get_quote", lambda *a: order.append("quote") or ctx)
    response = local_client().post("/analyze", files={"file": ("test.png", image.getvalue(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert order == ["save", "quote"]
    assert result["timestamp"] == result["detail"]["published_at"] == result["detail"]["model_comparison"]["gemini"]["timestamp"]


def test_database_refuses_nonfinite_values_before_persistence(monkeypatch, tmp_path):
    row = _optional_qwen_record(monkeypatch, tmp_path)
    data = row.to_dict()["detail"]
    data["vision_ai"]["entry_price"] = float("inf")
    with pytest.raises(ValueError):
        db.save_analysis(row.image_path, data)
    session = db.SessionLocal()
    assert session.query(db.Analysis).count() == 1
    session.close()
