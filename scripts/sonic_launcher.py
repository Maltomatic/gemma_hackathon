"""Socket launcher for SONIC trajectories.

Listens on a TCP port. A client (any device on the network) sends a trajectory number
as an ASCII line; the launcher runs ``g1_sim.run_sonic`` on the matching clip as a
subprocess, waits for it to finish naturally, then writes a status line back on the
same connection.

    python scripts/sonic_launcher.py                  # 0.0.0.0:5555, MuJoCo viewer
    python scripts/sonic_launcher.py --headless       # no window (CI / remote box)
    python scripts/sonic_launcher.py --port 6000 --hold 1.5

Protocol (newline-terminated ASCII, one request at a time per connection):

    client -> "3\\n"        run trajectory 3 (forward)
    server -> "0\\n"        trajectory finished (sent when the sim process exits)

    client -> "?\\n"        list trajectories
    server -> "0 tap ok\\n1 shake ok\\n ... \\n.\\n"

Status codes written back:
    0  completed
    1  unknown trajectory number, or its CSV is missing
    2  busy, another trajectory is still running
    3  the sim process exited with an error

Trajectory table: 0 tap, 1 shake, 2 slap, 3 forward, 4 left, 5 right
(``motion_demo/<name>.csv``, Unitree LAFAN1 retargeting format).

Quick manual test from another machine:

    printf '0\\n' | nc <laptop-ip> 5555          # prints 0 when the tap finishes
"""
from __future__ import annotations

import argparse
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MOTION_DIR = REPO_ROOT / "motion_demo"
TRAJECTORIES: dict[int, str] = {0: "tap", 1: "shake", 2: "slap", 3: "forward", 4: "left", 5: "right"}

STATUS_DONE, STATUS_UNKNOWN, STATUS_BUSY, STATUS_FAILED = 0, 1, 2, 3


class LauncherState:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        self.current: subprocess.Popen | None = None
        self.current_name: str | None = None

    def motion_path(self, idx: int) -> Path | None:
        name = TRAJECTORIES.get(idx)
        if name is None:
            return None
        p = MOTION_DIR / f"{name}.csv"
        return p if p.exists() else None

    def try_start(self, idx: int) -> tuple[int, subprocess.Popen | None]:
        """Start the sim for trajectory idx. Returns (status, process)."""
        path = self.motion_path(idx)
        if path is None:
            return STATUS_UNKNOWN, None
        with self.lock:
            if self.current is not None and self.current.poll() is None:
                return STATUS_BUSY, None
            cmd = [sys.executable, "-m", "g1_sim.run_sonic", "--motion", str(path),
                   "--hold", str(self.args.hold)]
            if self.args.headless:
                # headless: run exactly hold + clip length (+ settle margin) then exit
                cmd += ["--headless", str(self.args.hold + clip_seconds(path) + 0.5)]
            else:
                cmd += ["--exit-on-end", "--end-hold", str(self.args.end_hold)]
            log(f"start  {idx} ({TRAJECTORIES[idx]}): {' '.join(cmd[2:])}")
            self.current = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
            self.current_name = TRAJECTORIES[idx]
            return STATUS_DONE, self.current

    def listing(self) -> str:
        lines = []
        for idx, name in TRAJECTORIES.items():
            lines.append(f"{idx} {name} {'ok' if self.motion_path(idx) else 'missing'}")
        return "\n".join(lines) + "\n.\n"


def clip_seconds(csv_path: Path, fps: float = 30.0) -> float:
    with open(csv_path, "rb") as f:
        n = sum(1 for _ in f)
    return n / fps


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Handler(socketserver.StreamRequestHandler):
    state: LauncherState  # set on the server class

    def handle(self) -> None:
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        log(f"client {peer} connected")
        try:
            while True:
                raw = self.rfile.readline()
                if not raw:
                    break
                msg = raw.decode("utf-8", "replace").strip()
                if not msg:
                    continue
                if msg in ("?", "list"):
                    self.wfile.write(self.state.listing().encode())
                    continue
                try:
                    idx = int(msg)
                except ValueError:
                    log(f"client {peer}: bad request {msg!r}")
                    self.reply(STATUS_UNKNOWN)
                    continue

                status, proc = self.state.try_start(idx)
                if proc is None:
                    log(f"client {peer}: trajectory {idx} -> status {status}")
                    self.reply(status)
                    continue

                rc = proc.wait()  # blocks this connection until the sim exits on its own
                status = STATUS_DONE if rc == 0 else STATUS_FAILED
                log(f"done   {idx} ({TRAJECTORIES[idx]}) exit code {rc} -> status {status}")
                self.reply(status)
        except (ConnectionResetError, BrokenPipeError):
            log(f"client {peer} dropped")
        finally:
            log(f"client {peer} closed")

    def reply(self, status: int) -> None:
        self.wfile.write(f"{status}\n".encode())
        self.wfile.flush()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--hold", type=float, default=2.0, help="seconds the policy holds frame 0 before playing")
    p.add_argument("--end-hold", type=float, default=0.5, help="seconds to keep the final pose before the sim exits")
    p.add_argument("--headless", action="store_true", help="run the sim without a window")
    args = p.parse_args()

    Handler.state = LauncherState(args)
    with Server((args.host, args.port), Handler) as srv:
        ips = sorted({i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)})
        log(f"SONIC launcher listening on {args.host}:{args.port}  (this machine: {', '.join(ips)})")
        log("trajectories:\n" + Handler.state.listing().rstrip("\n."))
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            log("shutting down")


if __name__ == "__main__":
    main()
