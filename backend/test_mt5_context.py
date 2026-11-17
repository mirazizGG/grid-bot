"""Candle freshness checks with deterministic MT5 snapshots, without a broker."""
import datetime as dt
from types import SimpleNamespace

import numpy as np
import pytest

import mt5_context
import mt5_clock


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
    monkeypatch.setattr(mt5_context.bridge_client, "enabled", lambda: False)
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


@pytest.fixture
def native_quote(monkeypatch):
    monkeypatch.setattr(mt5_context.bridge_client, "enabled", lambda: False)
    monkeypatch.setattr(mt5_context.time, "time", lambda: NOW)


def test_quote_default_keeps_none_for_initialization_failure(monkeypatch, native_quote):
    calls = []
    monkeypatch.setattr(mt5_context, "mt5", SimpleNamespace(
        initialize=lambda: calls.append("initialize") or False,
        last_error=lambda: calls.append("last_error") or (-6, "Terminal: Authorization failed"),
    ))
    assert mt5_context.get_quote("XAUUSD") is None
    assert calls == ["initialize", "last_error"]


def test_strict_quote_reports_safe_authentication_error_while_locked(monkeypatch, native_quote):
    calls = []

    class Guard:
        active = False

        def __enter__(self):
            self.active = True

        def __exit__(self, *args):
            self.active = False

    guard = Guard()

    def initialize():
        assert guard.active
        calls.append("initialize")
        return False

    def last_error():
        assert guard.active
        calls.append("last_error")
        return -6, "login=123456 password=private-password server=private-server " * 100

    monkeypatch.setattr(mt5_context, "MT5_LOCK", guard)
    monkeypatch.setattr(mt5_context, "mt5", SimpleNamespace(initialize=initialize, last_error=last_error))
    with pytest.raises(mt5_context.MT5ReadError) as failure:
        mt5_context.get_quote("XAUUSD", raise_on_error=True)
    message = str(failure.value)
    assert isinstance(failure.value, ValueError)
    assert "-6" in message and "hisobingizga" in message and "server" in message
    assert all(secret not in message for secret in ("123456", "private-password", "private-server"))
    assert len(message) < 300
    assert calls == ["initialize", "last_error"]
    assert not guard.active


@pytest.mark.parametrize("error", [None, (), (True, "private-secret"),
                                   (10 ** 400, "private-secret"), (-10005, "private-secret")])
def test_other_initialization_errors_have_bounded_static_messages(monkeypatch, native_quote, error):
    monkeypatch.setattr(mt5_context, "mt5", SimpleNamespace(initialize=lambda: False, last_error=lambda: error))
    with pytest.raises(mt5_context.MT5ReadError) as failure:
        mt5_context.get_quote("XAUUSD", raise_on_error=True)
    assert "private-secret" not in str(failure.value)
    assert len(str(failure.value)) < 200
    if error and error[0] == -10005:
        assert "-10005" in str(failure.value)


def test_failed_sdk_error_reader_does_not_expose_its_exception(monkeypatch, native_quote):
    def last_error():
        raise RuntimeError("password=private-password")

    monkeypatch.setattr(mt5_context, "mt5", SimpleNamespace(initialize=lambda: False, last_error=last_error))
    with pytest.raises(mt5_context.MT5ReadError) as failure:
        mt5_context.get_quote("XAUUSD", raise_on_error=True)
    assert "private-password" not in str(failure.value)
    assert "noma'lum" in str(failure.value)


def test_missing_native_module_is_optional_by_default(monkeypatch, native_quote):
    monkeypatch.setattr(mt5_context, "mt5", None)
    assert mt5_context.get_quote("XAUUSD") is None
    with pytest.raises(mt5_context.MT5ReadError, match="MetaTrader5.*Windows"):
        mt5_context.get_quote("XAUUSD", raise_on_error=True)


def test_native_quote_recovers_after_login_without_stale_shared_error(monkeypatch, native_quote):
    initialized = iter([False, True])
    shutdowns = []
    monkeypatch.setattr(mt5_context, "mt5", SimpleNamespace(
        initialize=lambda: next(initialized), last_error=lambda: (-6, "Authorization failed"),
        symbol_select=lambda *args: True,
        symbol_info_tick=lambda *args: SimpleNamespace(bid=100, ask=101, time=NOW),
        symbol_info=lambda *args: SimpleNamespace(digits=2),
        shutdown=lambda: shutdowns.append(True),
    ))
    with pytest.raises(mt5_context.MT5ReadError, match="-6"):
        mt5_context.get_quote("XAUUSD", raise_on_error=True)
    quote = mt5_context.get_quote("XAUUSD", raise_on_error=True)
    assert quote["bid"] == 100 and quote["ask"] == 101
    assert shutdowns == [True]


@pytest.mark.parametrize("strict", [False, True])
def test_stale_native_quote_still_fails_closed_without_connection_error(monkeypatch, native_quote, strict):
    monkeypatch.setattr(mt5_context, "mt5", SimpleNamespace(
        initialize=lambda: True, shutdown=lambda: None, symbol_select=lambda *args: True,
        symbol_info_tick=lambda *args: SimpleNamespace(bid=100, ask=101, time=NOW - 61),
        symbol_info=lambda *args: SimpleNamespace(digits=2),
    ))
    assert mt5_context.get_quote("XAUUSD", raise_on_error=strict) is None


@pytest.mark.parametrize("strict", [False, True])
def test_bridge_error_is_optional_and_retry_recovers(monkeypatch, strict):
    monkeypatch.setattr(mt5_context.bridge_client, "enabled", lambda: True)
    monkeypatch.setattr(mt5_context.time, "time", lambda: NOW)
    calls, recorded = [], []

    def request(operation, args, timeout):
        calls.append((operation, args, timeout))
        if len(calls) == 1:
            raise mt5_context.bridge_client.BridgeError("MT5 demo hisobiga kiring (kod -6)")
        return {"bid": 100, "ask": 101, "digits": 2, "tick_time": NOW}

    monkeypatch.setattr(mt5_context.bridge_client, "request", request)
    monkeypatch.setattr(mt5_context.bridge_client, "record_error", recorded.append)
    if strict:
        with pytest.raises(mt5_context.MT5ReadError, match="demo.*-6"):
            mt5_context.get_quote("XAUUSD", raise_on_error=True)
    else:
        assert mt5_context.get_quote("XAUUSD") is None
    assert len(calls) == 1  # no implicit retry or account change
    assert recorded == ["MT5 demo hisobiga kiring (kod -6)"]
    assert mt5_context.get_quote("XAUUSD", raise_on_error=strict)["ask"] == 101
    assert calls == [("quote", {"symbol": "XAUUSD", "clock_profile": mt5_clock.normalize_profile()}, 3)] * 2


def test_malformed_bridge_quote_has_no_raw_value_in_diagnostic(monkeypatch):
    monkeypatch.setattr(mt5_context.bridge_client, "enabled", lambda: True)
    monkeypatch.setattr(mt5_context.bridge_client, "request", lambda *args, **kwargs: {
        "bid": "private-secret", "ask": 101, "digits": 2, "tick_time": NOW})
    recorded = []
    monkeypatch.setattr(mt5_context.bridge_client, "record_error", recorded.append)
    with pytest.raises(mt5_context.MT5ReadError) as failure:
        mt5_context.get_quote("XAUUSD", raise_on_error=True)
    assert "private-secret" not in str(failure.value)
    assert recorded == [str(failure.value)]


def test_explicit_server_clock_normalizes_context_but_preserves_model_hours(monkeypatch):
    profile = mt5_clock.normalize_profile({"server": "MetaQuotes-Demo", "offset_minutes": 120,
                                           "history_mode": "server"})
    snapshots = {1: snapshot(now=NOW + 7200), 2: snapshot("H1", now=NOW + 7200),
                 3: snapshot("H4", now=NOW + 7200)}
    install_mt5(monkeypatch, snapshots)
    monkeypatch.setattr(mt5_context.mt5, "account_info", lambda: SimpleNamespace(server="MetaQuotes-Demo"), raising=False)
    monkeypatch.setattr(mt5_context.mt5, "symbol_info_tick", lambda *a: SimpleNamespace(
        bid=201, ask=201.1, time=NOW + 7200, time_msc=int((NOW + 7200) * 1000)))
    # An unconfigured future quote remains invalid.
    assert mt5_context.get_quote("XAUUSD", clock_profile=mt5_clock.normalize_profile()) is None
    context = mt5_context.get_context("XAUUSD", "M15", clock_profile=profile)
    assert context["tick_time"] == NOW
    assert context["raw_tick_time"] == NOW + 7200
    assert context["bar_time"] == snapshots[1][-2]["time"] - 7200
    assert context["raw_bar_time"] == snapshots[1][-2]["time"]
    assert context["clock_profile"] == profile
    assert context["feature_time_basis"] == "broker_wall"
    assert set(context["higher_tf"]) == {"H1", "H4"}
    broker_hour = dt.datetime.fromtimestamp(int(snapshots[1][-2]["time"]), dt.timezone.utc).hour
    assert context["hour_sin"] == round(float(np.sin(2 * np.pi * broker_hour / 24)), 4)
    assert context["hour_cos"] == round(float(np.cos(2 * np.pi * broker_hour / 24)), 4)
    # Neither the native arrays nor the model's raw-hour inputs are rewritten.
    assert snapshots[1][-1]["time"] > NOW


@pytest.mark.parametrize("offset", [-180, 0, 120])
@pytest.mark.parametrize("age", [-6, 61])
def test_configured_clock_does_not_accept_genuinely_future_or_stale_quotes(monkeypatch, offset, age):
    monkeypatch.setattr(mt5_context.time, "time", lambda: NOW)
    profile = {"server": "MetaQuotes-Demo", "offset_minutes": offset, "history_mode": "server"}
    tick = SimpleNamespace(bid=100, ask=101, time=NOW + offset * 60 - age)
    assert mt5_context._checked_quote(tick, SimpleNamespace(digits=2), clock_profile=profile) is None


def test_bridge_clock_is_already_utc_and_cannot_be_applied_twice(monkeypatch):
    monkeypatch.setattr(mt5_context.time, "time", lambda: NOW)
    profile = mt5_clock.normalize_profile({"server": "MetaQuotes-Demo", "offset_minutes": 120,
                                           "history_mode": "server"})
    quote = {"bid": 100, "ask": 101, "digits": 2, "tick_time": NOW,
             "raw_tick_time": NOW + 7200, "clock_profile": profile, "timestamp_basis": "utc"}
    assert mt5_context._bridge_quote(quote, profile)["tick_time"] == NOW
    with pytest.raises(mt5_context.bridge_client.BridgeError):
        mt5_context._bridge_quote(quote, mt5_clock.normalize_profile())
    with pytest.raises(mt5_context.bridge_client.BridgeError):
        mt5_context._bridge_quote({**quote, "raw_tick_time": NOW}, profile)


def test_native_clock_rejects_another_broker_server(monkeypatch, native_quote):
    monkeypatch.setattr(mt5_context, "mt5", SimpleNamespace(
        initialize=lambda: True, shutdown=lambda: None,
        account_info=lambda: SimpleNamespace(server="Different-Demo"),
        symbol_select=lambda *a: pytest.fail("wrong server must not supply a quote"),
    ))
    with pytest.raises(mt5_context.MT5ReadError, match="server"):
        mt5_context.get_quote("XAUUSD", raise_on_error=True,
                              clock_profile={"server": "MetaQuotes-Demo", "offset_minutes": 120})
