"""Model switches use temporary artifacts only; production files stay intact."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

import custom_model
import model_control
import mt5_context


def write_artifact(directory, seed, candidate=False, features=None):
    directory.mkdir(parents=True, exist_ok=True)
    features = features or mt5_context.FEATURE_NAMES
    frame = pd.DataFrame(np.random.default_rng(seed).normal(size=(12, 17)), columns=features)
    estimator = DecisionTreeClassifier(max_depth=2, random_state=seed).fit(frame, ["Buy", "Hold", "Sell"] * 4)
    path = directory / "numeric_model.joblib"
    joblib.dump(estimator, path)
    meta = {"feature_names": mt5_context.FEATURE_NAMES, "classes": list(estimator.classes_),
            "model_sha256": custom_model.file_sha256(path),
            "provenance_status": "verified_candidate" if candidate else "legacy_unverified"}
    if candidate:
        meta.update(training_split="train", validation={"accuracy": .38, "majority_class_baseline_accuracy": .41})
    (directory / "numeric_model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


@pytest.fixture
def isolated_models(monkeypatch, tmp_path):
    active, candidate, state = tmp_path / "backend/model", tmp_path / "training/model_candidate", tmp_path / "backups/model-control"
    old_meta = write_artifact(active, 1)
    new_meta = write_artifact(candidate, 2, candidate=True)
    for name in ("_model", "_meta", "_FEATURE_NAMES", "_snapshot_error"):
        monkeypatch.setattr(custom_model, name, getattr(custom_model, name))
    custom_model.install_snapshot(custom_model.load_model(active))
    for name, value in (("PROJECT_ROOT", tmp_path), ("ACTIVE_DIR", active), ("CANDIDATE_DIR", candidate), ("STATE_DIR", state)):
        monkeypatch.setattr(model_control, name, value)
    monkeypatch.setattr(model_control, "_verify_dataset", lambda meta: None)
    return active, candidate, state, old_meta, new_meta


def assert_active(directory, metadata):
    assert custom_model.file_sha256(directory / "numeric_model.joblib") == metadata["model_sha256"]
    assert json.loads((directory / "numeric_model_meta.json").read_text()) == metadata
    assert custom_model.snapshot()[1] == metadata


def test_registry_is_read_only_and_shows_warnings(isolated_models):
    active, candidate, state, old, new = isolated_models
    report = model_control.registry()
    assert report["active"]["sha256"] == old["model_sha256"]
    assert report["active"]["warning"]
    assert report["candidate"]["available"] and report["candidate"]["warning"]
    assert report["candidate"]["validation_accuracy"] == .38
    assert not report["rollback"]["available"]
    assert not state.exists()
    assert_active(active, old)


def test_activate_then_rollback_preserves_artifact_pairs(isolated_models):
    active, candidate, state, old, new = isolated_models
    report = model_control.activate_candidate()
    assert report["candidate"]["active"]
    assert report["rollback"]["sha256"] == old["model_sha256"]
    assert_active(active, new)
    assert not (state / "transaction.json").exists()
    model_control.rollback()
    assert_active(active, old)
    assert custom_model.file_sha256(candidate / "numeric_model.joblib") == new["model_sha256"]
    model_control.activate_candidate()
    with pytest.raises(ValueError, match="allaqachon"):
        model_control.activate_candidate()


@pytest.mark.parametrize("failure", ["hash", "lineage", "features", "dataset"])
def test_invalid_candidate_never_changes_active(isolated_models, monkeypatch, failure):
    active, candidate, state, old, new = isolated_models
    if failure == "hash":
        with (candidate / "numeric_model.joblib").open("ab") as output:
            output.write(b"changed")
    elif failure == "lineage":
        new["provenance_status"] = "legacy_unverified"
        (candidate / "numeric_model_meta.json").write_text(json.dumps(new))
    elif failure == "features":
        write_artifact(candidate, 3, candidate=True, features=list(reversed(mt5_context.FEATURE_NAMES)))
    else:
        def invalid_dataset(meta):
            raise ValueError("dataset provenance mos emas")
        monkeypatch.setattr(model_control, "_verify_dataset", invalid_dataset)
    assert not model_control.registry()["candidate"]["available"]
    with pytest.raises(ValueError):
        model_control.activate_candidate()
    assert_active(active, old)
    assert not state.exists()


def test_reload_failure_restores_disk_and_runtime(isolated_models, monkeypatch):
    active, candidate, state, old, new = isolated_models
    original = custom_model.install_snapshot
    def fail_new_snapshot(pair):
        if pair[1]["model_sha256"] == new["model_sha256"]:
            raise ValueError("simulated reload failure")
        original(pair)
    monkeypatch.setattr(custom_model, "install_snapshot", fail_new_snapshot)
    with pytest.raises(RuntimeError, match="tiklandi"):
        model_control.activate_candidate()
    assert_active(active, old)
    assert not (state / "rollback.json").exists()
    assert not (state / "transaction.json").exists()


def test_partial_replace_failure_restores_both_files(isolated_models, monkeypatch):
    active, candidate, state, old, new = isolated_models
    original = model_control.os.replace
    failed = False
    def fail_metadata_once(source, target):
        nonlocal failed
        if Path(target).resolve() == (active / "numeric_model_meta.json").resolve() and not failed:
            failed = True
            raise OSError("simulated second replace failure")
        original(source, target)
    monkeypatch.setattr(model_control.os, "replace", fail_metadata_once)
    with pytest.raises(RuntimeError, match="tiklandi"):
        model_control.activate_candidate()
    assert_active(active, old)
    assert not (state / "transaction.json").exists()


def test_failed_restore_blocks_inference_and_keeps_recovery_journal(isolated_models, monkeypatch):
    active, candidate, state, old, new = isolated_models
    original = model_control.os.replace
    def fail_all_metadata_replaces(source, target):
        if Path(target).resolve() == (active / "numeric_model_meta.json").resolve():
            raise OSError("simulated persistent filesystem failure")
        original(source, target)
    monkeypatch.setattr(model_control.os, "replace", fail_all_metadata_replaces)
    with pytest.raises(RuntimeError, match="tiklash tugamadi"):
        model_control.activate_candidate()
    assert (state / "transaction.json").exists()
    with pytest.raises(RuntimeError, match="inference to'xtatildi"):
        custom_model.snapshot()
    assert not model_control.registry()["active"]["available"]


def test_interrupted_process_transaction_recovers_before_loading(isolated_models, monkeypatch):
    active, candidate, state, old, new = isolated_models
    original = model_control.os.replace
    def interrupt_metadata(source, target):
        if Path(target).resolve() == (active / "numeric_model_meta.json").resolve():
            raise KeyboardInterrupt("simulated process interruption")
        original(source, target)
    monkeypatch.setattr(model_control.os, "replace", interrupt_metadata)
    with pytest.raises(KeyboardInterrupt):
        model_control.activate_candidate()
    assert (state / "transaction.json").exists()
    monkeypatch.setattr(model_control.os, "replace", original)
    assert model_control.recover_pending_transaction()
    assert_active(active, old)
    assert not (state / "transaction.json").exists()
    assert not (state / "rollback.json").exists()


def test_fresh_process_recovers_before_importing_model(isolated_models, monkeypatch):
    active, candidate, state, old, new = isolated_models
    original = model_control.os.replace
    def interrupt_metadata(source, target):
        if Path(target).resolve() == (active / "numeric_model_meta.json").resolve():
            raise KeyboardInterrupt("simulated process interruption")
        original(source, target)
    monkeypatch.setattr(model_control.os, "replace", interrupt_metadata)
    with pytest.raises(KeyboardInterrupt):
        model_control.activate_candidate()
    script = """import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import config
config.MODEL_DIR = sys.argv[2]
import model_control
model_control.PROJECT_ROOT = Path(sys.argv[3])
model_control.ACTIVE_DIR = Path(sys.argv[2])
model_control.STATE_DIR = Path(sys.argv[4])
import custom_model
print(custom_model.snapshot()[1]['model_sha256'])
"""
    result = subprocess.run([sys.executable, "-c", script, str(Path(custom_model.__file__).resolve().parent),
                             str(active), str(model_control.PROJECT_ROOT), str(state)],
                            capture_output=True, text=True, timeout=20, check=True)
    assert result.stdout.strip() == old["model_sha256"]
    assert_active(active, old)
    assert not (state / "transaction.json").exists()


def test_candidate_cannot_escape_fixed_workspace(isolated_models, monkeypatch, tmp_path):
    active, candidate, state, old, new = isolated_models
    monkeypatch.setattr(model_control, "CANDIDATE_DIR", tmp_path.parent / "outside")
    with pytest.raises(ValueError, match="loyiha ichida"):
        model_control.activate_candidate()
    assert_active(active, old)


def test_inflight_inference_keeps_its_model_metadata_pair(isolated_models, monkeypatch):
    active, candidate, state, old, new = isolated_models
    model, _ = custom_model.snapshot()
    started, release = threading.Event(), threading.Event()
    original = model.predict_proba
    def wait_during_prediction(features):
        started.set()
        assert release.wait(5)
        return original(features)
    monkeypatch.setattr(model, "predict_proba", wait_during_prediction)
    monkeypatch.setattr(mt5_context, "to_feature_vector", lambda context: [0.] * 17)
    monkeypatch.setattr(mt5_context, "compute_trade_levels", lambda *args: {"entry_price": 1, "tp_price": 2, "sl_price": .5})
    monkeypatch.setattr(custom_model.reasoning, "explain", lambda *args: {"trend": "test", "reasoning": "test"})
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(custom_model.analyze, {"symbol": "TEST"})
        assert started.wait(5)
        model_control.activate_candidate()
        release.set()
        assert pending.result(timeout=5)["model_sha256"] == old["model_sha256"]
    assert custom_model.analyze({"symbol": "TEST"})["model_sha256"] == new["model_sha256"]
