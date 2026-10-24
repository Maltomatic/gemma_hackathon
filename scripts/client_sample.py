"""Generic client for scripts/sonic_launcher.py. Run it anywhere on the network.

Only the Python standard library is needed, so copy this one file to the other device.

    python client_sample.py 192.168.7.23            # interactive: type 0-5, ?, or q
    python client_sample.py 192.168.7.23 3          # run trajectory 3, then stay at the prompt
    python client_sample.py 192.168.7.23 0 1 2      # several in sequence, then stay at the prompt
    python client_sample.py 192.168.7.23 3 --once   # run and exit (for scripts)
    python client_sample.py 192.168.7.23 ?          # list trajectories
    python client_sample.py 192.168.7.23 --port 6000 3

The connection stays open between motions; type q (or Ctrl-C) to leave. With --once the
exit code is the launcher's status: 0 done, 1 unknown trajectory, 2 busy, 3 sim failed,
4 connection problem.
"""
from __future__ import annotations

import argparse
import socket
import sys
import time

TRAJECTORIES = {0: "tap", 1: "shake", 2: "slap", 3: "forward", 4: "left", 5: "right"}
STATUS_TEXT = {0: "done", 1: "unknown trajectory / missing file", 2: "busy", 3: "sim process failed"}
CONNECT_ERROR = 4


class LauncherClient:
    def __init__(self, host: str, port: int, timeout: float | None = None):
        self.host, self.port, self.timeout = host, port, timeout
        self.sock: socket.socket | None = None
        self.buf = b""

    def __enter__(self) -> "LauncherClient":
        self.sock = socket.create_connection((self.host, self.port), timeout=10)
        self.sock.settimeout(self.timeout)  # None = wait as long as the trajectory takes
        return self

    def __exit__(self, *exc) -> None:
        if self.sock:
            self.sock.close()

    def _readline(self) -> str:
        while b"\n" not in self.buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("launcher closed the connection")
            self.buf += chunk
        line, _, self.buf = self.buf.partition(b"\n")
        return line.decode("utf-8", "replace").strip()

    def send(self, text: str) -> None:
        self.sock.sendall((text.strip() + "\n").encode())

    def list_trajectories(self) -> list[str]:
        self.send("?")
        lines = []
        while True:
            ln = self._readline()
            if ln == ".":
                return lines
            lines.append(ln)

    def run(self, idx: int) -> int:
        """Request trajectory idx and block until the launcher reports completion."""
        self.send(str(idx))
        return int(self._readline())


def describe(idx: int) -> str:
    return f"{idx} ({TRAJECTORIES.get(idx, '?')})"


def run_one(client: LauncherClient, idx: int) -> int:
    print(f"-> {describe(idx)} ... ", end="", flush=True)
    t0 = time.perf_counter()
    status = client.run(idx)
    print(f"{status} = {STATUS_TEXT.get(status, 'unknown status')} after {time.perf_counter() - t0:.1f}s")
    return status


def interactive(client: LauncherClient) -> None:
    print("connected. commands: 0-5 run trajectory, ? list, q quit")
    for idx, name in TRAJECTORIES.items():
        print(f"  {idx}  {name}")
    while True:
        try:
            cmd = input("sonic> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not cmd:
            continue
        if cmd in ("q", "quit", "exit"):
            return
        if cmd in ("?", "list"):
            for ln in client.list_trajectories():
                print("  " + ln)
            continue
        try:
            idx = int(cmd)
        except ValueError:
            print("  type a trajectory number, ? or q")
            continue
        run_one(client, idx)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("host", help="IP or hostname of the machine running sonic_launcher.py")
    p.add_argument("commands", nargs="*", help="trajectory numbers (0-5) and/or '?'; none = interactive")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--once", action="store_true",
                   help="exit after the command-line trajectories instead of staying at the prompt")
    args = p.parse_args()

    try:
        with LauncherClient(args.host, args.port) as client:
            last = 0
            for cmd in args.commands:
                if cmd in ("?", "list"):
                    for ln in client.list_trajectories():
                        print(ln)
                    continue
                last = run_one(client, int(cmd))
                if last != 0 and args.once:
                    return last
            if args.once:
                return last
            interactive(client)
            return 0
    except (OSError, ConnectionError) as e:
        print(f"connection problem with {args.host}:{args.port}: {e}", file=sys.stderr)
        return CONNECT_ERROR


if __name__ == "__main__":
    sys.exit(main())
