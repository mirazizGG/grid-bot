"""Candle freshness checks with deterministic MT5 snapshots, without a broker."""
import datetime as dt
from types import SimpleNamespace

import numpy as np
import pytest

import mt5_context


NOW = dt.datetime(2026, 11, 16, 10, 30, 20, tzinfo=dt.timezone.utc).timestamp()
DTYPE = [("time", "i8"), ("open", "f8"), ("high", "f8"),
         ("low", "f8"), ("close", "f8"), ("tick_volume", "i8")]


def snapshot(timeframe="M15", now=NOW, closed_count=100):
    seconds = mt5_context.TIMEFRAME_SECONDS[timeframe]
    forming_time = int(now // seconds) * seconds
    return np.array([(forming_time - (closed_count - i) * seconds,
                      100 + i, 102 + i, 99 + i, 101 + i, 10)
                     for i in range(closed_count + 1)], dtype=DTYPE)


def install_mt5(monkeypatch, snapshots):
    calls = []
    monkeypatch.setattr(mt5_context.time, "time", lambda: NOW)
    monkeypatch.setattr(mt5_context, "TIMEFRAME_MAP", {"M15": 1, "H1": 2, "H4": 3})

    def copy_rates(symbol, timeframe, position, count):
        calls.append((timeframe, position, count))
        return snapshots[timeframe]

    monkeypatch.setattr(mt5_context, "mt5", SimpleNamespace(
        initialize=lambda: True, shutdown=lambda: None,
        symbol_select=lambda *args: True,
        symbol_info_tick=lambda *args: SimpleNamespace(bid=201, ask=201.1, time=NOW),
        symbol_info=lambda *args: SimpleNamespace(digits=2),
        copy_rates_from_pos=copy_rates,
    ))
    return calls


def test_context_uses_closed_bars_from_current_snapshot(monkeypatch):
    primary = snapshot()
    primary[-1]["close"] = 9999  # must never reach model indicators
    calls = install_mt5(monkeypatch, {1: primary, 2: snapshot("H1"), 3: snapshot("H4")})
    context = mt5_context.get_context("XAUUSD", "M15")
    assert context["last_close"] == primary[-2]["close"]
    assert context["bar_time"] == primary[-2]["time"]
    assert set(context["higher_tf"]) == {"H1", "H4"}
    assert calls == [(1, 0, 101), (2, 0, 101), (3, 0, 101)]


def test_current_quote_does_not_make_stale_primary_history_valid(monkeypatch):
    old = snapshot(now=NOW - 365 * 86400)
    calls = install_mt5(monkeypatch, {1: old})
    assert mt5_context.get_context("XAUUSD", "M15") is None
    assert calls == [(1, 0, 101)]


def test_stale_higher_timeframe_is_omitted(monkeypatch):
    install_mt5(monkeypatch, {1: snapshot(), 2: snapshot("H1", now=NOW - 86400),
                             3: snapshot("H4")})
    context = mt5_context.get_context("XAUUSD", "M15")
    assert set(context["higher_tf"]) == {"H4"}


@pytest.mark.parametrize("corruption", ["future", "duplicate", "reversed", "old_closed"])
def test_invalid_timestamps_are_rejected(corruption):
    rates = snapshot()
    if corruption == "future":
        rates["time"][-1] = int(NOW + 60)
    elif corruption == "duplicate":
        rates["time"][-2] = rates["time"][-1]
    elif corruption == "reversed":
        rates[[5, 6]] = rates[[6, 5]]
    else:
        rates["time"][:-1] -= 365 * 86400
    assert mt5_context._checked_closed_rates(rates, "M15", NOW) is None


def test_weekend_gap_is_allowed_when_current_session_is_fresh():
    monday = dt.datetime(2026, 11, 16, 0, 0, 20, tzinfo=dt.timezone.utc).timestamp()
    rates = snapshot(now=monday)
    friday = dt.datetime(2026, 11, 13, 21, 45, tzinfo=dt.timezone.utc).timestamp()
    rates["time"][:-1] -= int(rates["time"][-2] - friday)
    closed = mt5_context._checked_closed_rates(rates, "M15", monday)
    assert closed is not None and closed[-1]["time"] == friday


def test_short_but_sufficient_history_is_retained():
    assert len(mt5_context._checked_closed_rates(snapshot(closed_count=55), "M15", NOW)) == 55
    assert mt5_context._checked_closed_rates(snapshot(closed_count=54), "M15", NOW) is None
