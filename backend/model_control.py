"""Fixed local model registry and recoverable, explicitly requested switches."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import uuid

import pandas as pd

from config import MODEL_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DIR = Path(MODEL_DIR)
CANDIDATE_DIR = PROJECT_ROOT / "training" / "model_candidate"
STATE_DIR = PROJECT_ROOT / "backups" / "model-control"
FILENAMES = ("numeric_model.joblib", "numeric_model_meta.json")


def _checked(path):
    """Reject links/reparse points and any escape from the application tree."""
    path = Path(path).absolute()
    root = Path(PROJECT_ROOT).absolute()
    try:
        canonical_root = root.resolve()
        path.resolve().relative_to(canonical_root)
        # Desktop can be an existing workspace junction to X:\Projects. Accept
        # that configured root alias, but never links inside the artifact tree.
        aliases = (root, Path(MODEL_DIR).absolute().parents[1])
        base = next(alias for alias in aliases if alias.resolve() == canonical_root and path.is_relative_to(alias))
        relative = path.relative_to(base)
    except (ValueError, StopIteration) as exc:
        raise ValueError("Model yo'li loyiha ichida bo'lishi kerak") from exc
    current = base
    for component in relative.parts:
        current = current / component
        if current.exists() or current.is_symlink():
            info = current.lstat()
            if current.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                raise ValueError("Model yo'lida havola yoki reparse point taqiqlangan")
    return path.resolve()


def _json_read(path):
    with _checked(path).open(encoding="utf-8") as source:
        return json.load(source)


def _json_write(path, value):
    path = _checked(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _backup_path(pointer):
    if not isinstance(pointer, dict) or not re.fullmatch(r"[0-9TZ]+-[0-9a-f]{32}", str(pointer.get("backup_id", ""))):
        raise ValueError("Rollback yozuvi yaroqsiz")
    return _checked(STATE_DIR / "versions" / pointer["backup_id"])


def _validate_backup(pointer):
    import custom_model
    directory = _backup_path(pointer)
    for filename, key in zip(FILENAMES, ("model_sha256", "metadata_sha256")):
        if custom_model.file_sha256(_checked(directory / filename)) != pointer.get(key):
            raise ValueError("Rollback fayli SHA256 mos emas")
    return directory


def _replace_pair(directory):
    staged = []
    try:
        for filename in FILENAMES:
            target = _checked(ACTIVE_DIR / filename)
            temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
            with _checked(directory / filename).open("rb") as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
            staged.append((temporary, target))
        for temporary, target in staged:
            os.replace(temporary, target)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def _restore_pointer(pointer):
    target = _checked(STATE_DIR / "rollback.json")
    if pointer is None:
        target.unlink(missing_ok=True)
    else:
        _json_write(target, pointer)


def recover_pending_transaction():
    """Restore the complete previous pair after a crash, before initial load."""
    journal_path = _checked(STATE_DIR / "transaction.json")
    if not journal_path.exists():
        return False
    journal = _json_read(journal_path)
    directory = _validate_backup(journal["restore"])
    _replace_pair(directory)
    _restore_pointer(journal.get("previous_rollback"))
    journal_path.unlink()
    return True


def _verify_candidate():
    import custom_model
    directory = _checked(CANDIDATE_DIR)
    for filename in FILENAMES:
        _checked(directory / filename)
    pair = custom_model.load_model(directory)
    meta = pair[1]
    if meta.get("provenance_status") != "verified_candidate" or meta.get("training_split") != "train":
        raise ValueError("Candidate o'qitish provenance tasdiqlanmagan")
    _verify_dataset(meta)
    return pair


def _verify_dataset(meta):
    # Reuse the research checks and hashes; importing this module never trains.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from training.train_numeric_model import validate_labeled, dataset_provenance
    labeled_path = _checked(PROJECT_ROOT / "data_pipeline" / "output" / "multi_labeled.parquet")
    labeled = pd.read_parquet(labeled_path)
    validate_labeled(labeled)
    for symbol, timeframe in labeled[["symbol", "timeframe"]].drop_duplicates().itertuples(index=False, name=None):
        if symbol not in {"XAUUSD", "EURUSD", "GBPUSD", "USDJPY"} or timeframe not in {"M15", "H1"}:
            raise ValueError("Candidate dataset symbol/timeframe yaroqsiz")
        _checked(labeled_path.parent / "multi" / f"{symbol}_{timeframe}_raw.parquet")
    if meta.get("dataset") != dataset_provenance(labeled):
        raise ValueError("Candidate dataset provenance mos emas")


def _describe(identifier, pair):
    import custom_model
    meta = pair[1]
    validation = meta.get("validation", {})
    return {"id": identifier, "available": True, "sha256": meta["model_sha256"],
            "provenance_status": meta.get("provenance_status", "legacy_unverified"),
            "validation_accuracy": validation.get("accuracy"),
            "majority_baseline": validation.get("majority_class_baseline_accuracy"),
            "warning": custom_model.metadata_warning(meta)}


def _unavailable(identifier, exc):
    warning = str(exc)
    if isinstance(exc, FileNotFoundError):
        warning = ("Oldingi model zaxirasi hali yaratilmagan" if identifier == "rollback"
                   else "Model yoki dataset fayllari topilmadi")
    elif isinstance(exc, OSError):
        warning = "Model fayllarini o'qib bo'lmadi"
    return {"id": identifier, "available": False, "sha256": None, "provenance_status": None,
            "validation_accuracy": None, "majority_baseline": None, "warning": warning}


def registry():
    """Read-only status: no directory creation, backups, training or activation."""
    import custom_model
    with custom_model.MODEL_LOCK:
        try:
            pair = custom_model.snapshot()
            disk_meta = _json_read(ACTIVE_DIR / FILENAMES[1])
            if disk_meta != pair[1] or custom_model.file_sha256(_checked(ACTIVE_DIR / FILENAMES[0])) != pair[1]["model_sha256"]:
                raise ValueError("Faol disk va xotira modeli mos emas")
            active = _describe("active", pair)
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            active = _unavailable("active", exc)
        try:
            candidate = _describe("candidate", _verify_candidate())
        except (OSError, ValueError, TypeError, KeyError) as exc:
            candidate = _unavailable("candidate", exc)
        candidate["active"] = bool(candidate["available"] and candidate["sha256"] == active["sha256"])
        try:
            pointer = _json_read(STATE_DIR / "rollback.json")
            rollback_info = _describe("rollback", custom_model.load_model(_validate_backup(pointer)))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            rollback_info = _unavailable("rollback", exc)
        return {"active": active, "candidate": candidate, "rollback": rollback_info}


def _switch(directory, pair):
    import custom_model
    previous_snapshot = custom_model.snapshot()
    # Ensure the running snapshot and the disk backup represent the same pair.
    current = custom_model.load_model(_checked(ACTIVE_DIR))
    if current[1] != previous_snapshot[1]:
        raise ValueError("Faol disk va xotira modeli mos emas; serverni qayta tekshiring")
    pointer_path = _checked(STATE_DIR / "rollback.json")
    previous_pointer = _json_read(pointer_path) if pointer_path.exists() else None
    backup_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    backup = _backup_path({"backup_id": backup_id})
    backup.mkdir(parents=True, exist_ok=False)
    for filename in FILENAMES:
        with _checked(ACTIVE_DIR / filename).open("rb") as source, (backup / filename).open("xb") as destination:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
    pointer = {"backup_id": backup_id, "model_sha256": current[1]["model_sha256"],
               "metadata_sha256": custom_model.file_sha256(backup / FILENAMES[1])}
    _validate_backup(pointer)
    journal_path = STATE_DIR / "transaction.json"
    _json_write(journal_path, {"restore": pointer, "previous_rollback": previous_pointer})
    try:
        _replace_pair(directory)
        installed = custom_model.load_model(ACTIVE_DIR)
        if installed[1] != pair[1]:
            raise ValueError("Candidate nusxasi tekshiruvdan o'tmadi")
        custom_model.install_snapshot(installed)
        _json_write(pointer_path, pointer)
        _checked(journal_path).unlink()
    except Exception as exc:
        try:
            _replace_pair(backup)
            custom_model.install_snapshot(previous_snapshot)
            _restore_pointer(previous_pointer)
            _checked(journal_path).unlink(missing_ok=True)
        except Exception as restore_exc:
            custom_model.disable_inference("Model transaction tiklanmadi; server tekshirilguncha inference to'xtatildi")
            raise RuntimeError("Modelni tiklash tugamadi; transaction yozuvi saqlandi. Serverni tekshiring.") from restore_exc
        raise RuntimeError("Model almashtirilmadi; oldingi model va metadata tiklandi") from exc


def activate_candidate():
    import custom_model
    with custom_model.MODEL_LOCK:
        if _checked(STATE_DIR / "transaction.json").exists():
            raise ValueError("Tugallanmagan model transaction mavjud")
        pair = _verify_candidate()
        if pair[1]["model_sha256"] == custom_model.snapshot()[1]["model_sha256"]:
            raise ValueError("Candidate allaqachon faol")
        _switch(CANDIDATE_DIR, pair)
        return registry()


def rollback():
    import custom_model
    with custom_model.MODEL_LOCK:
        if _checked(STATE_DIR / "transaction.json").exists():
            raise ValueError("Tugallanmagan model transaction mavjud")
        try:
            pointer = _json_read(STATE_DIR / "rollback.json")
            directory = _validate_backup(pointer)
            pair = custom_model.load_model(directory)
        except FileNotFoundError as exc:
            raise ValueError("Rollback modeli mavjud emas") from exc
        _switch(directory, pair)
        return registry()
