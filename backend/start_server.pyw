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
    # If any application owns the port, never start a duplicate or terminate it.
    if listening(8811):
        return
    python = Path(sys.executable).with_name("python.exe")
    environment = os.environ.copy()
    environment.update(GRID_BOT_SUPERVISED="1", GRID_BOT_SUPERVISOR_PID=str(os.getpid()))
    with LOG.open("ab") as log:
        while True:
            child = subprocess.Popen(
                [str(python), "-c", "from control_runtime import run_supervised_server; run_supervised_server()"],
                cwd=str(HERE), env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), close_fds=True, shell=False,
            )
            if child.wait() != 75 or listening(8811):
                return


if __name__ == "__main__":
    main()
