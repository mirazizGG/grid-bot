"""Local administration routes and a bounded, recoverable operation queue."""
import csv
import datetime as dt
import io
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import FileResponse, StreamingResponse

import config
import control_runtime
import db_backup
import evaluator
import gemini_vision
import model_control
import ollama_vision
from db import SessionLocal, Analysis

BACKUP_DIR = Path(__file__).resolve().parents[1] / "backups"
UPLOAD_DIR = config.UPLOAD_DIR


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _public_result(value):
    if isinstance(value, dict):
        return {str(key): _public_result(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public_result(item) for item in value]
    if isinstance(value, str):
        return control_runtime._redact(value)
    return value


class JobQueue:
    def __init__(self, path, workers=2, capacity=16):
        self.path = Path(path)
        self.capacity = capacity
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="grid-control")
        self.jobs = []
        self.restarting = False
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(saved, list):
                self.jobs = saved[-100:]
                for job in self.jobs:
                    if job["status"] in ("queued", "running"):
                        job.update(status="failed", finished_at=_now(), error="Server qayta ishga tushgan; amalni qayta boshlang.")
        except (OSError, ValueError, KeyError, TypeError):
            self.jobs = []

    def _persist(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             suffix=".tmp", delete=False) as stream:
                temporary = stream.name
                json.dump(self.jobs, stream, allow_nan=False)
            os.replace(temporary, self.path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(list(reversed(self.jobs))))

    def submit(self, kind, function):
        with self.lock:
            active = [j for j in self.jobs if j["status"] in ("queued", "running")]
            duplicate = next((j for j in active if j["kind"] == kind), None)
            if duplicate:
                return dict(duplicate)
            if self.restarting:
                raise ValueError("Backend qayta ishga tushmoqda; yangi vazifalar vaqtincha qabul qilinmaydi")
            if kind == "backend:restart" and active:
                raise ValueError("Backendni qayta boshlashdan oldin joriy vazifalar tugashini kuting")
            if len(active) >= self.capacity:
                raise ValueError("Vazifalar navbati to'la; joriy vazifalar tugashini kuting")
            job = {"id": uuid.uuid4().hex, "kind": kind, "status": "queued", "created_at": _now(),
                   "started_at": None, "finished_at": None, "result": None, "error": None}
            previous = self.jobs
            self.jobs = [j for j in self.jobs if j["status"] in ("queued", "running")] + [
                j for j in self.jobs if j["status"] not in ("queued", "running")][-83:]
            self.jobs.append(job)
            try:
                self._persist()
            except OSError:
                self.jobs = previous
                raise
            if kind == "backend:restart":
                self.restarting = True
            queued = dict(job)
            self.pool.submit(self._run, job, function)
            return queued

    def _run(self, job, function):
        with self.lock:
            job.update(status="running", started_at=_now())
            try:
                self._persist()
            except OSError:
                job.update(status="failed", finished_at=_now(), error="Vazifa holatini saqlab bo'lmadi")
                if job["kind"] == "backend:restart":
                    self.restarting = False
                return
        try:
            result = function()
            # Keep credentials out of both the API and the durable job journal.
            result = json.loads(json.dumps(_public_result(result), allow_nan=False, default=str))
            status, error = "succeeded", None
        except Exception as exc:
            result, status = None, "failed"
            error = control_runtime._redact(str(exc))[:1200]
        with self.lock:
            if job["kind"] == "backend:restart" and status == "failed":
                self.restarting = False
            job.update(status=status, result=result, error=error, finished_at=_now())
            try:
                self._persist()
            except OSError:
                job.update(status="failed", error="Vazifa tugadi, lekin natija jurnalga saqlanmadi")


jobs = JobQueue(config.RUNTIME_DIR / "jobs.json")


def service_capabilities():
    capabilities = control_runtime.service_capabilities()
    if any(job["status"] in ("queued", "running") for job in jobs.snapshot()) or jobs.restarting:
        if "backend" in capabilities:
            capabilities["backend"]["restart"] = {"enabled": False, "reason": "Joriy vazifalar tugashini kuting"}
    return capabilities


def storage_snapshot():
    uploads = [p for p in UPLOAD_DIR.glob("*") if p.is_file() and not p.is_symlink() and p.name != ".gitkeep"]
    path = Path(config.DB_PATH)
    return {"db_bytes": path.stat().st_size if path.is_file() else 0,
            "uploads_bytes": sum(p.stat().st_size for p in uploads), "upload_count": len(uploads),
            "backup_count": len(list(BACKUP_DIR.glob("*.zip")))}


def list_backups():
    results = []
    for path in BACKUP_DIR.glob("*.zip"):
        if path.is_file() and not path.is_symlink() and path.resolve().parent == BACKUP_DIR.resolve():
            info = path.stat()
            results.append({"name": path.name, "bytes": info.st_size,
                            "created_at": dt.datetime.fromtimestamp(info.st_mtime, dt.timezone.utc).isoformat(),
                            "download_url": f"/control/backups/{path.name}"})
    return sorted(results, key=lambda item: item["created_at"], reverse=True)[:100]


def create_backup():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    name = "grid-bot-backup-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8] + ".zip"
    destination = BACKUP_DIR / name
    try:
        db_backup.export_backup(str(destination))
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return {"name": name, "bytes": destination.stat().st_size, "download_url": f"/control/backups/{name}"}


def apply_settings(changes):
    with config.SETTINGS_LOCK:
        result = config.save_settings(changes)
        values = config.get_settings()
        gemini_vision.configure(values["gemini_api_key"], values["gemini_model"])
        ollama_vision.configure(values["qwen_model"], values["qwen_timeout"])
        return result


def build_router(service_reader, activity_reader):
    router = APIRouter(prefix="/control", tags=["Control panel"])

    @router.get("/overview")
    def overview():
        services = service_reader()
        services["capabilities"] = service_capabilities()
        return {"generated_at": _now(), "services": services,
                "system": control_runtime.runtime_snapshot(), "settings": config.get_settings(public=True),
                "models": model_control.registry(), "jobs": jobs.snapshot(), "storage": storage_snapshot(),
                "analysis": activity_reader()}

    @router.get("/settings")
    def settings():
        return config.get_settings(public=True)

    @router.post("/settings")
    def update_settings(changes: dict = Body(...)):
        try:
            return apply_settings(changes)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except OSError:
            raise HTTPException(503, "Sozlamalarni diskka saqlab bo'lmadi")

    @router.get("/jobs")
    def list_jobs():
        return {"jobs": jobs.snapshot()}

    def submit(kind, function):
        try:
            return jobs.submit(kind, function)
        except (OSError, ValueError) as exc:
            raise HTTPException(409, control_runtime._redact(str(exc)))

    @router.post("/jobs/{kind}", status_code=202)
    def run_job(kind: str):
        functions = {"evaluate": evaluator.evaluate_pending, "backup": create_backup,
                     "validate_model": model_control.registry}
        if kind not in functions:
            raise HTTPException(400, "Noma'lum vazifa")
        return submit(kind, functions[kind])

    @router.post("/services/{service}/{action}", status_code=202)
    def service_action(service: str, action: str):
        capabilities = service_capabilities()
        permission = capabilities.get(service, {}).get(action)
        if not permission:
            raise HTTPException(400, "Noma'lum xizmat amali")
        if not permission["enabled"]:
            raise HTTPException(409, permission.get("reason") or "Bu amal hozir bajarilmaydi")
        return submit(f"{service}:{action}", lambda: control_runtime.service_action(service, action))

    @router.post("/models/{action}", status_code=202)
    def change_model(action: str):
        functions = {"activate_candidate": model_control.activate_candidate, "rollback": model_control.rollback}
        if action not in functions:
            raise HTTPException(400, "Noma'lum model amali")
        return submit(f"model:{action}", functions[action])

    @router.get("/logs")
    def logs(source: str = "server", lines: int = Query(100, ge=1, le=500)):
        try:
            return control_runtime.read_logs(source, lines)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except (OSError, RuntimeError):
            raise HTTPException(503, "Log faylini o'qib bo'lmadi")

    @router.get("/backups")
    def backups():
        return {"backups": list_backups()}

    @router.get("/backups/{name}")
    def download_backup(name: str):
        path = BACKUP_DIR / name
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.zip", name) or ".." in name or
                path.is_symlink() or path.resolve().parent != BACKUP_DIR.resolve() or not path.is_file()):
            raise HTTPException(404, "Zaxira topilmadi")
        return FileResponse(path, filename=name, media_type="application/zip")

    @router.get("/history/export")
    def export_history():
        def rows():
            output = io.StringIO(newline="")
            writer = csv.writer(output)
            writer.writerow(["id", "timestamp_utc", "symbol", "timeframe", "signal", "entry", "tp", "sl", "outcome"])
            yield "\ufeff" + output.getvalue()
            output.seek(0)
            output.truncate(0)
            session = SessionLocal()
            try:
                for row in session.query(Analysis).order_by(Analysis.id).yield_per(100):
                    ctx = json.loads(row.raw_result).get("market_context") or {}
                    values = [row.id, row.timestamp.isoformat() + "Z", ctx.get("symbol"), ctx.get("timeframe"),
                              row.final_signal, row.entry_price, row.tp_price, row.sl_price, row.outcome]
                    writer.writerow(["'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@")) else value
                                     for value in values])
                    yield output.getvalue()
                    output.seek(0)
                    output.truncate(0)
            finally:
                session.close()
        return StreamingResponse(rows(), media_type="text/csv; charset=utf-8",
                                 headers={"Content-Disposition": 'attachment; filename="grid-bot-history.csv"'})

    return router
