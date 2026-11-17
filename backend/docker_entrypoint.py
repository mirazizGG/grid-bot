"""Seed persistent model storage and supervise only this container's API process."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import config

FILENAMES = ("numeric_model.joblib", "numeric_model_meta.json")


def seed_model(source=Path("/opt/grid-bot-model"), destination=None):
    destination = Path(destination or config.MODEL_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    existing = [(destination / name).is_file() for name in FILENAMES]
    if all(existing):
        return
    if any(existing):
        raise RuntimeError("Model saqlash papkasi to'liq emas; zaxira nusxani tekshiring")
    staged = []
    try:
        for name in FILENAMES:
            temporary = destination / (name + ".seed.tmp")
            staged.append((temporary, destination / name))
            shutil.copyfile(source / name, temporary)
        # Install only after both source files were copied successfully.
        for temporary, target in staged:
            os.replace(temporary, target)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def main():
    seed_model()
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(GRID_BOT_SUPERVISED="1", GRID_BOT_SUPERVISOR_PID=str(os.getpid()))
    stopping, child = False, None

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (config.LOG_DIR / "server_log.txt").open("a", encoding="utf-8", buffering=1) as log:
        while not stopping:
            child = subprocess.Popen(
                [sys.executable, "-c", "from control_runtime import run_supervised_server; run_supervised_server()"],
                cwd=Path(__file__).resolve().parent, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            )
            for line in child.stdout:
                log.write(line)
                print(line, end="", flush=True)
            code = child.wait()
            if stopping or code != 75:
                return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
