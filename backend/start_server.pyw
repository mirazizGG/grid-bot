"""Start Grid-Bot after interactive Windows sign-in without a visible console."""
import os
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "server_log.txt"


def listening(port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main():
    ollama_app = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama app.exe"
    if ollama_app.is_file() and not listening(11434):
        subprocess.Popen([str(ollama_app), "--hide", "--fast-startup"], cwd=str(ollama_app.parent),
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), close_fds=True)
    # If any application owns the port, never start a duplicate or terminate it.
    if listening(8811):
        return
    python = Path(sys.executable).with_name("python.exe")
    with LOG.open("ab") as log:
        subprocess.Popen(
            [str(python), "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8811"],
            cwd=str(HERE), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            close_fds=True,
        )


if __name__ == "__main__":
    main()
