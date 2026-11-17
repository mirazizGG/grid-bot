"""Explicit clock profiles, native query conversion and credential-free diagnosis."""
import datetime as dt
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mt5_clock as clock

START = dt.datetime(2026, 11, 17, 10, 0, tzinfo=dt.timezone.utc)
START_MS = int(START.timestamp() * 1000)


def profile(mode="server", offset=120):
    return {"version": 1, "server": "Demo-Exact", "offset_minutes": offset, "history_mode": mode}


def native(rows):
    calls = []
    api = SimpleNamespace(account_info=lambda: SimpleNamespace(server="Demo-Exact"),
                          copy_ticks_range=lambda *args: calls.append(args) or rows)
    return api, calls


def test_defaults_are_independent_and_nonzero_partial_history_is_unverified():
    identity = clock.normalize_profile()
    assert identity == {"version": 1, "server": "", "offset_minutes": 0, "history_mode": "utc"}
    identity["offset_minutes"] = 60
    assert clock.normalize_profile()["offset_minutes"] == 0
    assert clock.normalize_profile({"offset_minutes": 120, "server": "Demo-Exact"})["history_mode"] == "unverified"
    assert clock.offset_seconds(profile(offset=-90)) == -5400


@pytest.mark.parametrize("value", [[], "auto", {"extra": 1}, {"version": True}, {"version": 2},
    {"offset_minutes": True}, {"offset_minutes": 120.0}, {"offset_minutes": 841},
    {"offset_minutes": -841}, {"offset_minutes": 120}, {"history_mode": "auto"},
    {"server": " nonexact"}, {"server": "nonexact "}, {"server": "bad\nserver"},
    {"server": "Unicode-ё"}, {"server": "x" * 129}])
def test_invalid_profiles_are_rejected(value):
    with pytest.raises(clock.ClockProfileError):
        clock.normalize_profile(value)


def test_server_binding_reads_only_exact_server_metadata():
    class Account:
        server = "Demo-Exact"

        @property
        def login(self):
            pytest.fail("Account identifiers must not be accessed")

    api = SimpleNamespace(account_info=lambda: Account())
    assert clock.validate_server(api, profile()) == profile()
    with pytest.raises(clock.ClockProfileError, match="server"):
        clock.validate_server(api, {**profile(), "server": "demo-exact"})
    clock.validate_server(SimpleNamespace(), None)  # identity mode never reads account info


def test_rates_are_copied_and_only_timestamps_are_normalized():
    raw = np.array([(int(START.timestamp()) + 7200, 123.5)], dtype=[("time", "i8"), ("close", "f8")])
    normalized = clock.normalize_rates(raw, profile())
    assert normalized["time"][0] == int(START.timestamp())
    assert normalized["close"][0] == raw["close"][0]
    assert raw["time"][0] == int(START.timestamp()) + 7200
    assert not np.shares_memory(raw, normalized)
    assert clock.normalize_rates(None, profile()) is None


@pytest.mark.parametrize("mode,offset,query_seconds,result_seconds", [
    ("utc", 120, 0, 0), ("server", 120, 7200, 7200),
    ("utc_query_server_result", 120, 0, 7200), ("server", -90, -5400, -5400),
    ("utc", 0, 0, 0),
])
def test_history_queries_and_results_use_the_explicit_mode(mode, offset, query_seconds, result_seconds):
    raw = np.array([(START_MS + result_seconds * 1000 + 123, 1.1, 1.2)], dtype=clock.TICK_DTYPE)
    api, calls = native(raw)
    result = clock.copy_ticks_utc(api, "XAUUSD", START, START + dt.timedelta(seconds=5), 7, profile(mode, offset))
    assert calls == [("XAUUSD", START + dt.timedelta(seconds=query_seconds),
                      START + dt.timedelta(seconds=5 + query_seconds), 7)]
    assert result.tolist() == [(START_MS + 123, 1.1, 1.2)]
    assert raw["time_msc"][0] == START_MS + result_seconds * 1000 + 123


def test_exact_subsecond_bounds_filter_native_second_granularity_rows():
    raw = [{"time_msc": START_MS + delta, "bid": 1., "ask": 1.1} for delta in (-1, 500, 501, 1000, 1001)]
    api, _ = native(raw)
    result = clock.copy_ticks_utc(api, "XAUUSD", START + dt.timedelta(microseconds=500123),
                                  START + dt.timedelta(seconds=1), 1, profile("utc", 0))
    assert result["time_msc"].tolist() == [START_MS + 501, START_MS + 1000]


@pytest.mark.parametrize("start,end", [(START, START), (START, START - dt.timedelta(seconds=1)),
    (START, START + dt.timedelta(minutes=5, microseconds=1)),
    (START.replace(tzinfo=None), START + dt.timedelta(seconds=1)),
    (START.astimezone(dt.timezone(dt.timedelta(hours=2))), START + dt.timedelta(seconds=1))])
def test_invalid_history_bounds_never_reach_sdk(start, end):
    api, calls = native([])
    with pytest.raises(clock.ClockProfileError):
        clock.copy_ticks_utc(api, "XAUUSD", start, end, 1, profile())
    assert not calls


def test_unverified_history_refuses_sdk_and_five_minute_limit_is_inclusive():
    api, calls = native([])
    with pytest.raises(clock.ClockProfileError, match="tasdiqlanmagan"):
        clock.copy_ticks_utc(api, "XAUUSD", START, START + dt.timedelta(seconds=1), 1, profile("unverified"))
    assert not calls
    assert len(clock.copy_ticks_utc(api, "XAUUSD", START, START + dt.timedelta(minutes=5), 1, None)) == 0


@pytest.mark.parametrize("changes", [{"bid": float("nan")}, {"ask": float("inf")}, {"bid": 0},
    {"ask": .5}, {"bid": True}, {"bid": "private-secret"}, {"bid": 10 ** 400},
    {"time_msc": True}, {"time_msc": float(START_MS)}, {"time_msc": 0}])
def test_invalid_tick_values_fail_closed_without_echoing_values(changes):
    row = {"time_msc": START_MS + 1, "bid": 1., "ask": 1.1, **changes}
    api, _ = native([row])
    with pytest.raises(clock.ClockProfileError) as failure:
        clock.copy_ticks_utc(api, "XAUUSD", START, START + dt.timedelta(seconds=1), 1, None)
    assert "private-secret" not in str(failure.value)


def test_tick_count_limit_and_server_switch_are_rejected():
    row = {"time_msc": START_MS + 1, "bid": 1., "ask": 1.1}
    api, _ = native([row] * (clock.MAX_TICKS + 1))
    with pytest.raises(clock.ClockProfileError):
        clock.copy_ticks_utc(api, "XAUUSD", START, START + dt.timedelta(seconds=1), 1, None)
    servers = iter(["Demo-Exact", "Other-Server"])
    api.account_info = lambda: SimpleNamespace(server=next(servers))
    api.copy_ticks_range = lambda *args: [row]
    with pytest.raises(clock.ClockProfileError, match="server"):
        clock.copy_ticks_utc(api, "XAUUSD", START, START + dt.timedelta(seconds=1), 1, profile("utc"))


def test_native_proxy_snapshots_profile_and_shuts_down_on_server_mismatch():
    calls = []
    api = SimpleNamespace(initialize=lambda: calls.append("initialize") or True,
        shutdown=lambda: calls.append("shutdown"), COPY_TICKS_INFO=7,
        account_info=lambda: SimpleNamespace(server="Other-Server"))
    original = profile()
    proxy = clock.NativeMT5Proxy(api, original)
    original["server"] = "Other-Server"
    assert proxy.COPY_TICKS_INFO == 7 and proxy.profile["server"] == "Demo-Exact"
    with pytest.raises(clock.ClockProfileError):
        proxy.initialize()
    assert calls == ["initialize", "shutdown"]
    assert clock.NativeMT5Proxy(SimpleNamespace(initialize=lambda: False), None).initialize() is False
    assert clock.NativeMT5Proxy(SimpleNamespace(initialize=lambda: True), None).initialize() is True


def test_native_proxy_initialization_failure_is_cleaned_up_without_raw_sdk_error():
    calls = []
    api = SimpleNamespace(initialize=lambda: False, shutdown=lambda: calls.append("shutdown"))
    assert clock.NativeMT5Proxy(api, None).initialize() is False
    assert calls == ["shutdown"]

    def initialize():
        raise RuntimeError("password=private-secret")

    api.initialize = initialize
    api.last_error = lambda: (-6, "password=private-secret")
    with pytest.raises(clock.ClockProfileError) as failure:
        clock.NativeMT5Proxy(api, None).initialize()
    assert "private-secret" not in str(failure.value)
    assert failure.value.sdk_error_code == -6
    assert calls == ["shutdown", "shutdown"]


class FakeTime:
    def __init__(self):
        self.elapsed = 0.

    def time(self):
        return START.timestamp() + 20 + self.elapsed

    def monotonic(self):
        return self.elapsed

    def sleep(self, duration):
        self.elapsed += duration


class DiagnosticAPI:
    COPY_TICKS_INFO = 1
    COPY_TICKS_ALL = 3
    TIMEFRAME_M1, TIMEFRAME_M15, TIMEFRAME_H1 = 60, 900, 3600

    def __init__(self, now, mode="server", offset=120):
        self.now, self.mode, self.offset = now, mode, offset
        self.captured, self.calls = [], []
        self.server_reads = 0
        self.switch_server = False
        self.frozen_quotes = False

    def initialize(self, path, timeout):
        self.calls.append(("initialize", path, timeout))
        return True

    def shutdown(self):
        self.calls.append(("shutdown",))

    def last_error(self):
        return -6, "login=123 password=never-print"

    def account_info(self):
        self.server_reads += 1
        return SimpleNamespace(server="Other" if self.switch_server and self.server_reads > 1 else "Demo-Exact")

    def symbol_select(self, symbol, selected):
        self.calls.append(("select", symbol, selected))
        return True

    def symbol_info_tick(self, symbol):
        milliseconds = int(round((self.now.time() + self.offset * 60) * 1000))
        if self.frozen_quotes and self.captured:
            milliseconds = self.captured[-1]["time_msc"]
        row = {"time_msc": milliseconds, "bid": 1.1, "ask": 1.2}
        self.captured.append(row)
        return SimpleNamespace(time=milliseconds // 1000, **row)

    def copy_rates_from_pos(self, symbol, timeframe, position, count):
        seconds = int(self.now.time() // timeframe) * timeframe + self.offset * 60
        return np.array([(seconds,)], dtype=[("time", "i8")])

    def copy_ticks_range(self, symbol, start, end, flags):
        self.calls.append(("ticks", start, end, flags))
        rows = []
        for captured in self.captured:
            reference = captured["time_msc"] - self.offset * 60_000
            if self.mode == "ambiguous":
                rows.append(dict(captured))
                continue
            if self.mode == "none":
                continue
            query_clock = reference + (self.offset * 60_000 if self.mode == "server" else 0)
            if start.timestamp() * 1000 <= query_clock <= end.timestamp() * 1000:
                result_clock = reference if self.mode == "utc" else captured["time_msc"]
                rows.append({**captured, "time_msc": result_clock})
        return rows


@pytest.fixture
def fake_time(monkeypatch):
    now = FakeTime()
    monkeypatch.setattr(clock.time, "time", now.time)
    monkeypatch.setattr(clock.time, "monotonic", now.monotonic)
    monkeypatch.setattr(clock.time, "sleep", now.sleep)
    return now


@pytest.mark.parametrize("mode,offset", [("utc", 120), ("server", 120),
    ("utc_query_server_result", 120), ("utc", 0)])
def test_diagnosis_matches_live_fingerprints_and_unique_history_mode(fake_time, mode, offset):
    api = DiagnosticAPI(fake_time, mode, offset)
    result = clock._diagnose(api, "Demo-Exact", offset, Path("terminal64.exe"))
    assert result["verified"] and not result["applied"]
    assert result["profile"] == profile(mode, offset)
    assert result["checks"]["advancing_quotes"] == 3
    assert result["checks"]["history_matches"][mode] >= 2
    assert api.server_reads == 2
    assert api.calls[0][0] == "initialize" and api.calls[-1] == ("shutdown",)
    assert ("select", "XAUUSD", True) in api.calls
    assert all(call[3] == api.COPY_TICKS_ALL for call in api.calls if call[0] == "ticks")
    if offset == 0:
        assert set(result["checks"]["history_matches"]) == {"utc"}


@pytest.mark.parametrize("failure", ["ambiguous", "none", "frozen", "server_changed", "wrong_offset"])
def test_diagnosis_refuses_ambiguous_unmatched_stale_or_changed_server(fake_time, failure):
    api = DiagnosticAPI(fake_time, failure if failure in ("ambiguous", "none") else "server")
    api.frozen_quotes = failure == "frozen"
    api.switch_server = failure == "server_changed"
    with pytest.raises(clock.ClockProfileError):
        clock._diagnose(api, "Demo-Exact", 0 if failure == "wrong_offset" else 120, Path("terminal64.exe"))
    assert api.calls[-1] == ("shutdown",)
    assert fake_time.elapsed < 25


def test_cli_applies_only_verified_profile_and_prints_no_account_credentials(fake_time, monkeypatch, tmp_path, capsys):
    terminal = tmp_path / "terminal64.exe"
    terminal.touch()
    api = DiagnosticAPI(fake_time)
    monkeypatch.setattr(clock.importlib, "import_module", lambda name: api)
    applied = []
    monkeypatch.setattr(clock, "_apply_profile", lambda url, value: applied.append((url, value)))
    arguments = ["--diagnose", "--server", "Demo-Exact", "--offset-minutes", "120",
                 "--terminal-path", str(terminal), "--apply-url", "http://127.0.0.1:8811"]
    assert clock.main(arguments) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verified"] and report["applied"]
    assert applied == [("http://127.0.0.1:8811", profile())]
    api.mode = "ambiguous"
    assert clock.main(arguments) == 1
    report = capsys.readouterr().out
    assert "never-print" not in report and "123" not in report
    assert len(applied) == 1


@pytest.mark.parametrize("failure", ["wrong_offset", "stale_bar", "no_history", "ambiguous_history"])
def test_failed_cli_reports_bounded_numeric_time_evidence_without_sensitive_values(
        fake_time, monkeypatch, tmp_path, capsys, failure):
    terminal = tmp_path / "terminal64.exe"
    terminal.touch()
    mode = "none" if failure == "no_history" else "ambiguous" if failure == "ambiguous_history" else "server"
    api = DiagnosticAPI(fake_time, mode)
    if failure == "stale_bar":
        current_bars = api.copy_rates_from_pos

        def stale_bars(*args):
            rows = current_bars(*args)
            rows["time"] -= 3600
            return rows

        api.copy_rates_from_pos = stale_bars
    monkeypatch.setattr(clock.importlib, "import_module", lambda name: api)
    monkeypatch.setattr(clock, "_apply_profile", lambda *args: pytest.fail("An unverified profile must not be applied"))
    arguments = ["--diagnose", "--server", "Demo-Exact", "--offset-minutes",
                 "0" if failure == "wrong_offset" else "120", "--terminal-path", str(terminal),
                 "--apply-url", "http://127.0.0.1:8811"]
    assert clock.main(arguments) == 1
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["verified"] is False and report["applied"] is False
    assert "never-print" not in output and "password" not in output and "login" not in output
    assert str(terminal) not in output
    checks = report["checks"]
    quote = checks["last_quote"]
    assert isinstance(quote["raw_time"], int) and isinstance(quote["raw_time_msc"], int)
    assert isinstance(quote["os_epoch"], (int, float))
    assert quote["corrected_age"] == pytest.approx(-7200 if failure == "wrong_offset" else 0, abs=.001)
    if failure == "stale_bar":
        bar = checks["bar_observations"]["M1"]
        assert isinstance(bar["raw_time"], int) and bar["corrected_age"] > 60
    elif failure in ("no_history", "ambiguous_history"):
        assert set(checks["bar_observations"]) == {"M1", "M15", "H1"}
        assert set(checks["history_windows"]) == {"utc", "server"}
        assert set(checks["history_matches"]) == {"utc", "server", "utc_query_server_result"}
        if failure == "no_history":
            assert all(window == {"count": 0, "last_raw_ms": None} for window in checks["history_windows"].values())
        else:
            assert all(window["count"] >= 2 and isinstance(window["last_raw_ms"], int)
                       for window in checks["history_windows"].values())
    serialized = json.dumps(checks, allow_nan=False, separators=(",", ":"))
    assert len(serialized) < 1200
    assert '"bid"' not in serialized and '"ask"' not in serialized


@pytest.mark.parametrize("url", ["https://127.0.0.1:8811", "http://localhost:8811", "http://192.0.2.1:8811",
    "http://user:secret@127.0.0.1:8811", "http://127.0.0.1:8811/path", "http://127.0.0.1:8811?x=1",
    "http://127.0.0.1:8811#part", "http://127.0.0.1", "http://127.0.0.1:70000"])
def test_apply_refuses_nonfixed_loopback_urls_without_network(monkeypatch, url):
    monkeypatch.setattr(clock, "build_opener", lambda *args: pytest.fail("No HTTP operation allowed"))
    with pytest.raises(clock.ClockProfileError):
        clock._apply_profile(url, profile())


def test_apply_posts_exact_profile_then_independently_reads_it_back(monkeypatch):
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append((request, timeout))
            return io.BytesIO(json.dumps({"mt5_clock_profile": profile()}).encode())

    def opener(*handlers):
        assert handlers[0].proxies == {}
        assert isinstance(handlers[1], clock._NoRedirect)
        return Opener()

    monkeypatch.setattr(clock, "build_opener", opener)
    clock._apply_profile("http://127.0.0.1:8811", profile())
    assert [request.method for request, _ in calls] == ["POST", "GET"]
    assert all(request.full_url == "http://127.0.0.1:8811/control/settings" for request, _ in calls)
    assert calls[0][0].get_header("X-grid-bot-request") == "1"
    assert json.loads(calls[0][0].data) == {"mt5_clock_profile": profile()}


@pytest.mark.parametrize("result", [{}, {"mt5_clock_profile": None}, {"mt5_clock_profile": {"offset_minutes": 0}}])
def test_apply_rejects_missing_or_mismatched_profile_readback(monkeypatch, result):
    monkeypatch.setattr(clock, "build_opener", lambda *args: SimpleNamespace(
        open=lambda *args, **kwargs: io.BytesIO(json.dumps(result).encode())))
    with pytest.raises(clock.ClockProfileError):
        clock._apply_profile("http://127.0.0.1:8811", profile())
