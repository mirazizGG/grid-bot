"""Research pipeline tests use synthetic data, never a broker or model."""
import numpy as np
import pandas as pd
import pytest
import json

from data_pipeline import multi_symbol_dataset as pipeline
from training import backtest
from training.train_numeric_model import validate_labeled
from training import train_numeric_model
from data_pipeline.relabel_from_raw import relabel_dataset
from data_pipeline import mt5_fetch


def test_shared_cutoffs_purge_lookahead_across_instruments():
    times = pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC")
    rows = []
    for symbol, offset in (("A", 0), ("B", 30)):
        for i, timestamp in enumerate(times + pd.Timedelta(minutes=offset)):
            rows.append({"symbol": symbol, "timeframe": "H1", "window_end_idx": i,
                         "time": timestamp, "label_end_time": timestamp + pd.Timedelta(hours=4),
                         "label": "Hold", "label_version": 2})
    split = pipeline.assign_shared_splits(pd.DataFrame(rows))
    validate_labeled(split)
    for before, after in (("train", "val"), ("val", "test")):
        assert split.loc[split["split"] == before, "label_end_time"].max() <= split.loc[split["split"] == after, "time"].min()
    assert len(split) < len(rows)
    cutoff = split.loc[split["split"] == "test", "time"].min()
    assert (split.loc[split["time"] >= cutoff, "split"] == "test").all()


def test_ambiguous_candle_is_not_a_win(monkeypatch):
    monkeypatch.setattr(pipeline, "WINDOW_SIZE", 5)
    monkeypatch.setattr(pipeline, "WINDOW_STRIDE", 10)
    monkeypatch.setattr(pipeline, "ATR_PERIOD", 2)
    monkeypatch.setattr(pipeline, "LOOKAHEAD_BARS", 2)
    df = pd.DataFrame({"time": pd.date_range("2025-01-01", periods=10, freq="h"),
                       "high": [101.] * 10, "low": [99.] * 10, "close": [100.] * 10})
    df.loc[5, ["high", "low"]] = [110., 90.]
    result = pipeline.label_windows(df, "A", "H1")
    assert result.iloc[0]["label"] == "Hold"
    assert backtest.simulate_outcome(np.array([100., 110.]), np.array([100., 90.]), 0, 100., 2., "Buy", 1.5) == "loss"


def test_expiry_is_included_and_incomplete_not_misreported():
    n = backtest.LOOKAHEAD_BARS + 1
    raw = {"high": np.full(n, 100.4), "low": np.full(n, 99.9), "close": np.full(n, 100.25)}
    prediction = {"symbol": "A", "timeframe": "H1", "idx": 0, "entry": 100.,
                  "atr": 1., "signal": "Buy", "confidence": 0.8}
    summary = backtest.summarize([prediction], {("A", "H1"): raw}, 0.5, 2.)
    assert summary == {"wins": 0, "losses": 0, "expired": 1, "incomplete": 0, "mean_r": 0.25}
    assert backtest.simulate_outcome(raw["high"][:-1], raw["low"][:-1], 0, 100., 1., "Buy", 2.) == "incomplete"


def test_legacy_dataset_rejected():
    with pytest.raises(ValueError, match="Eski dataset"):
        validate_labeled(pd.DataFrame({"split": ["test"]}))


def test_history_fetch_without_native_mt5_fails_clearly(monkeypatch):
    monkeypatch.setattr(mt5_fetch, "mt5", None)
    with pytest.raises(RuntimeError, match="Windows hostida MetaTrader5"):
        mt5_fetch.fetch_ohlc("XAUUSD", "M15", 100)


def test_model_provenance_detects_changed_model_or_data(monkeypatch, tmp_path):
    model = tmp_path / "model.joblib"
    model.write_bytes(b"model-v1")
    meta = tmp_path / "meta.json"
    monkeypatch.setattr(backtest, "dataset_provenance", lambda _: {"raw": "v1"})
    meta.write_text(json.dumps({"model_sha256": backtest.file_sha256(model),
                                "training_split": "train", "dataset": {"raw": "v1"}}))
    backtest.verify_provenance(model, meta, None)
    model.write_bytes(b"model-v2")
    with pytest.raises(ValueError, match="provenance"):
        backtest.verify_provenance(model, meta, None)
    model.write_bytes(b"model-v1")
    monkeypatch.setattr(backtest, "dataset_provenance", lambda _: {"raw": "v2"})
    with pytest.raises(ValueError, match="provenance"):
        backtest.verify_provenance(model, meta, None)


def test_test_split_requires_preselected_parameters(monkeypatch):
    monkeypatch.setattr("sys.argv", ["backtest.py", "--split", "test"])
    with pytest.raises(SystemExit) as exc:
        backtest.main()
    assert exc.value.code == 2


def test_migration_backs_up_previous_artifacts_and_validates(tmp_path):
    raw_dir = tmp_path / "multi"
    raw_dir.mkdir()
    combined_path = tmp_path / "multi_labeled.parquet"
    combined_path.write_bytes(b"old-combined")
    previous_path = raw_dir / "XAUUSD_M15_labeled.parquet"
    previous_path.write_bytes(b"old-symbol")
    for symbol in ("XAUUSD", "EURUSD", "GBPUSD", "USDJPY"):
        for timeframe in ("M15", "H1"):
            periods = 1000
            raw = pd.DataFrame({"time": pd.date_range("2025-01-01", periods=periods,
                                                    freq="15min" if timeframe == "M15" else "h"),
                                "high": np.full(periods, 101.), "low": np.full(periods, 99.),
                                "close": np.full(periods, 100.)})
            raw.to_parquet(raw_dir / f"{symbol}_{timeframe}_raw.parquet", index=False)
    result = relabel_dataset(raw_dir, combined_path, tmp_path / "backup")
    validate_labeled(result)
    assert (tmp_path / "backup" / "multi" / combined_path.name).read_bytes() == b"old-combined"
    assert (tmp_path / "backup" / "multi" / previous_path.name).read_bytes() == b"old-symbol"
    pd.testing.assert_frame_equal(pd.read_parquet(combined_path), result.reset_index(drop=True))


def test_feature_builder_rejects_raw_index_timestamp_mismatch(monkeypatch, tmp_path):
    raw = pd.DataFrame({"time": pd.date_range("2025-01-01", periods=120, freq="h"),
                        "high": np.full(120, 101.), "low": np.full(120, 99.),
                        "close": np.full(120, 100.)})
    raw.to_parquet(tmp_path / "A_H1_raw.parquet")
    rows = [{"symbol": "A", "timeframe": "H1", "window_end_idx": 60 + i,
             "time": raw.time.iloc[60 + i] + pd.Timedelta(hours=1),
             "label_end_time": raw.time.iloc[60 + i] + pd.Timedelta(hours=2),
             "label": "Hold", "label_version": 2, "split": split}
            for i, split in ((0, "train"), (10, "val"), (20, "test"))]
    monkeypatch.setattr(train_numeric_model, "MULTI_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="Raw/label"):
        train_numeric_model.build_dataset(pd.DataFrame(rows))
