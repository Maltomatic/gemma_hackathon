"""Drive SONIC on the real G1 through the team's sonic-daemon (robot/daemon.py).

The daemon runs as a systemd service on the robot and listens on ws://<robot>:8765. It
launches the C++ SONIC binary on "start", proxies every other JSON message to it, and
forwards the binary's status broadcasts back. This script speaks that protocol so you
can run it from a laptop without the relay server or the browser GUI.

    python scripts/robot_sonic.py status                       # daemon + sonic state, motion list
    python scripts/robot_sonic.py start                        # launch SONIC, policy takes over (robot moves!)
    python scripts/robot_sonic.py play 01_emerson_move         # play a reference by name, wait until done
    python scripts/robot_sonic.py play 01_emerson_move --no-wait
    python scripts/robot_sonic.py rescan                       # re-read reference/ after adding a motion
    python scripts/robot_sonic.py stop                         # stop policy and kill the binary (robot goes limp!)
    python scripts/robot_sonic.py run 01_emerson_move          # start, play, wait, (optionally --stop-after)

Safety, read before "start":
  * On launch the binary ramps the robot to its default standing pose over 3 s, then
    the policy balances on its own once "start" is forwarded. The robot must be slack on
    the gantry with the native Unitree controller released, exactly as the team does
    with the browser GUI.
  * "stop" sends the binary's safe-shutdown command: damping only, no position hold.
    Have the robot supported before stopping.
  * This script never touches files on the robot.

Requires the `websockets` package (already in this repo's .venv).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

DEFAULT_HOST = "192.168.123.164"
DEFAULT_PORT = 8765


class SonicDaemonClient:
    def __init__(self, host: str, port: int, verbose: bool = True):
        self.uri = f"ws://{host}:{port}"
        self.verbose = verbose
        self.ws = None
        self.daemon_state: dict = {}
        self.status: dict = {}

    async def __aenter__(self):
        import websockets

        self.ws = await websockets.connect(self.uri, open_timeout=10, ping_interval=20, ping_timeout=20)
        return self

    async def __aexit__(self, *exc):
        if self.ws is not None:
            await self.ws.close()

    # ---------------------------------------------------------------- plumbing
    def log(self, msg: str) -> None:
        if self.verbose:
            print(time.strftime("%H:%M:%S"), msg, flush=True)

    async def send(self, payload: dict) -> None:
        await self.ws.send(json.dumps(payload))

    async def recv_one(self, timeout: float) -> dict | None:
        try:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return None
        t = msg.get("type")
        if t == "daemon_status":
            self.daemon_state = msg
        elif t == "status":
            self.status = msg
        return msg

    async def wait_for(self, pred, timeout: float, what: str) -> dict:
        """Pump messages until pred(msg) is true. Raises TimeoutError."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {what} after {timeout:.0f}s")
            msg = await self.recv_one(min(remaining, 5.0))
            if msg is not None and pred(msg):
                return msg

    # ----------------------------------------------------------------- queries
    async def snapshot(self, seconds: float = 2.0) -> None:
        """Collect whatever the daemon/sonic broadcast within a short window."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            await self.recv_one(max(0.1, end - time.monotonic()))

    def motions(self) -> list[str]:
        return list(self.status.get("motions", []))

    def motion_index(self, name: str) -> int:
        names = self.motions()
        if name in names:
            return names.index(name)
        # allow a unique substring / suffix match ("emerson" -> "01_emerson_move")
        hits = [i for i, n in enumerate(names) if name in n]
        if len(hits) == 1:
            return hits[0]
        raise KeyError(f"motion {name!r} not found or ambiguous; have: {names}")

    # ---------------------------------------------------------------- commands
    async def start(self, timeout: float = 300.0) -> None:
        """Launch SONIC via the daemon and wait until the policy is live (status frames arrive)."""
        self.log(f"-> control start  ({self.uri})  ROBOT WILL MOVE TO DEFAULT POSE AND START BALANCING")
        await self.send({"type": "control", "action": "start"})
        await self.wait_for(lambda m: m.get("type") == "status", timeout, "first status from sonic")
        self.log(f"sonic live: planner_enabled={self.status.get('planner_enabled')} "
                 f"motions={len(self.motions())} current={self.status.get('motion_name')!r}")

    async def play(self, name: str, wait: bool = True, timeout: float = 600.0) -> None:
        if not self.motions():
            await self.wait_for(lambda m: m.get("type") == "status", 10.0, "status with motion list")
        idx = self.motion_index(name)
        self.log(f"-> goto_and_play {idx} ({self.motions()[idx]})")
        await self.send({"type": "motion", "action": "goto_and_play", "index": idx})
        if not wait:
            return
        await self.wait_for(lambda m: m.get("type") == "status" and m.get("playing") is True, 15.0, "playback to begin")
        t0 = time.monotonic()
        self.log("playing ...")
        await self.wait_for(lambda m: m.get("type") == "status" and m.get("playing") is False, timeout, "playback to finish")
        self.log(f"done after {time.monotonic() - t0:.1f}s, robot blending back to planner idle")

    async def rescan(self) -> None:
        before = len(self.motions())
        self.log("-> rescan reference/")
        await self.send({"type": "motion", "action": "rescan"})
        await self.snapshot(2.0)
        self.log(f"motions: {before} -> {len(self.motions())}")

    async def stop(self) -> None:
        self.log("-> control stop  (policy stops, motors go to damping; robot must be supported)")
        await self.send({"type": "control", "action": "stop"})
        await self.snapshot(3.0)
        self.log(f"daemon: {self.daemon_state}")

    async def emergency_stop(self) -> None:
        await self.send({"type": "control", "action": "emergency_stop"})


def confirm(prompt: str) -> None:
    ans = input(f"{prompt} Type YES to continue: ").strip()
    if ans != "YES":
        print("aborted")
        sys.exit(1)


async def amain(args) -> int:
    async with SonicDaemonClient(args.host, args.port) as c:
        if args.cmd == "status":
            await c.snapshot(3.0)
            print("daemon :", json.dumps(c.daemon_state) if c.daemon_state else "(no daemon_status received)")
            if c.status:
                s = {k: v for k, v in c.status.items() if k != "motions"}
                print("sonic  :", json.dumps(s))
                print("motions:")
                for i, n in enumerate(c.motions()):
                    print(f"  {i:3d}  {n}")
            else:
                print("sonic  : not running (no status frames)")
            return 0

        if args.cmd == "start":
            if not args.yes:
                confirm("Robot slack on gantry, native controller released, area clear?")
            await c.start()
            return 0

        if args.cmd == "play":
            await c.play(args.motion, wait=not args.no_wait)
            return 0

        if args.cmd == "rescan":
            await c.rescan()
            return 0

        if args.cmd == "stop":
            if not args.yes:
                confirm("Robot supported? Stopping drops to damping.")
            await c.stop()
            return 0

        if args.cmd == "run":
            if not args.yes:
                confirm("Robot slack on gantry, native controller released, area clear?")
            await c.start()
            await asyncio.sleep(args.settle)
            await c.play(args.motion, wait=True)
            if args.stop_after:
                await asyncio.sleep(2.0)
                await c.stop()
            return 0
    return 2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("-y", "--yes", action="store_true", help="skip the safety confirmation prompts")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("start")
    pp = sub.add_parser("play"); pp.add_argument("motion"); pp.add_argument("--no-wait", action="store_true")
    sub.add_parser("rescan")
    sub.add_parser("stop")
    pr = sub.add_parser("run"); pr.add_argument("motion")
    pr.add_argument("--settle", type=float, default=5.0, help="seconds to let the policy balance before playing")
    pr.add_argument("--stop-after", action="store_true")
    args = p.parse_args()
    try:
        return asyncio.run(amain(args))
    except (OSError, TimeoutError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
