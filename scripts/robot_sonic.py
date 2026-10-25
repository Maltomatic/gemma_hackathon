"""Drive SONIC on the real G1 through the team's sonic-daemon (robot/daemon.py).

The daemon runs as a systemd service on the robot and listens on ws://<robot>:8765. It
launches the C++ SONIC binary on "start", proxies every other JSON message to it, and
forwards the binary's status broadcasts back. This script speaks that protocol so you
can run it from a laptop without the relay server or the browser GUI.

    python scripts/robot_sonic.py status                 # daemon + sonic state, motion list
    python scripts/robot_sonic.py run                    # 3 s countdown, start SONIC, then prompt for motions
    python scripts/robot_sonic.py run 01_emerson_move    # same, but play this one first
    python scripts/robot_sonic.py shell                  # prompt against an already-running SONIC
    python scripts/robot_sonic.py play ufb_tap           # one motion, wait until done, exit (SONIC keeps running)
    python scripts/robot_sonic.py rescan                 # re-read reference/ after adding a motion
    python scripts/robot_sonic.py stop                   # 3 s countdown, then stop SONIC (robot goes limp!)
    python scripts/robot_sonic.py estop                  # panic button: emergency_stop then stop, no countdown

At the prompt: a motion name or index plays it and waits for completion, then the robot
returns to planner idle and the prompt comes back. `list`, `rescan`, `status`, `stop`
(stops SONIC), `q` (leave SONIC running and exit).

Safety, read before "run":
  * On launch the binary ramps the robot to its default standing pose over 3 s, then
    the policy balances on its own once "start" is forwarded. The robot must be slack on
    the gantry with the native Unitree controller released, exactly as the team does
    with the browser GUI.
  * SONIC is never stopped automatically. Only `stop`, `estop`, or `stop` at the prompt
    do that, and they drop the motors to damping: have the robot supported.
  * Ctrl-C during the countdown aborts without sending anything.
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
    """Websocket client with a background reader; the latest status is always in `self.status`."""

    def __init__(self, host: str, port: int, verbose: bool = True):
        self.uri = f"ws://{host}:{port}"
        self.verbose = verbose
        self.ws = None
        self.daemon_state: dict = {}
        self.status: dict = {}
        self._changed = asyncio.Event()
        self._reader: asyncio.Task | None = None
        self._closed = False

    async def __aenter__(self):
        import websockets

        self.ws = await websockets.connect(self.uri, open_timeout=10, ping_interval=20, ping_timeout=20)
        self._reader = asyncio.create_task(self._pump())
        return self

    async def __aexit__(self, *exc):
        if self._reader:
            self._reader.cancel()
        if self.ws is not None:
            await self.ws.close()

    # ---------------------------------------------------------------- plumbing
    def log(self, msg: str) -> None:
        if self.verbose:
            print(time.strftime("%H:%M:%S"), msg, flush=True)

    async def _pump(self) -> None:
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                t = msg.get("type")
                if t == "daemon_status":
                    self.daemon_state = msg
                elif t == "status":
                    self.status = msg
                else:
                    continue
                self._changed.set()
        except Exception as e:  # noqa: BLE001 - connection dropped
            self.log(f"connection to daemon closed: {e}")
        finally:
            self._closed = True
            self._changed.set()

    async def send(self, payload: dict) -> None:
        if self._closed:
            raise ConnectionError("daemon connection is closed; re-run the command to reconnect")
        await self.ws.send(json.dumps(payload))

    async def wait_for(self, pred, timeout: float, what: str) -> None:
        """Block until pred(status, daemon_state) is true. Raises TimeoutError / ConnectionError."""
        deadline = time.monotonic() + timeout
        while True:
            if pred(self.status, self.daemon_state):
                return
            if self._closed:
                raise ConnectionError(f"daemon connection closed while waiting for {what}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {what} after {timeout:.0f}s")
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=min(remaining, 5.0))
            except asyncio.TimeoutError:
                pass

    async def settle(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    # ----------------------------------------------------------------- queries
    @property
    def sonic_live(self) -> bool:
        return bool(self.status)

    def motions(self) -> list[str]:
        return list(self.status.get("motions", []))

    def motion_index(self, name: str) -> int:
        names = self.motions()
        if name.isdigit() and int(name) < len(names):
            return int(name)
        if name in names:
            return names.index(name)
        hits = [i for i, n in enumerate(names) if name in n]  # unique substring, e.g. "emerson"
        if len(hits) == 1:
            return hits[0]
        raise KeyError(f"motion {name!r} not found or ambiguous ({len(hits)} matches)")

    def print_status(self) -> None:
        print("daemon :", json.dumps(self.daemon_state) if self.daemon_state else "(no daemon_status yet)")
        if self.status:
            s = {k: v for k, v in self.status.items() if k != "motions"}
            print("sonic  :", json.dumps(s))
        else:
            print("sonic  : not running (no status frames)")

    def print_motions(self) -> None:
        for i, n in enumerate(self.motions()):
            print(f"  {i:3d}  {n}")

    # ---------------------------------------------------------------- commands
    async def start(self, timeout: float = 300.0) -> None:
        """Launch SONIC via the daemon and wait until status frames arrive (policy live)."""
        if self.sonic_live:
            self.log("SONIC already running")
            return
        self.log(f"-> control start  ({self.uri})  ROBOT MOVES TO DEFAULT POSE, THEN BALANCES")
        await self.send({"type": "control", "action": "start"})
        await self.wait_for(lambda s, d: bool(s), timeout, "first status from sonic")
        self.log(f"sonic live: planner_enabled={self.status.get('planner_enabled')} "
                 f"motions={len(self.motions())} current={self.status.get('motion_name')!r}")

    async def play(self, name: str, timeout: float = 600.0) -> None:
        if not self.sonic_live:
            await self.wait_for(lambda s, d: bool(s), 10.0, "status with motion list")
        idx = self.motion_index(name)
        self.log(f"-> goto_and_play {idx} ({self.motions()[idx]})")
        await self.send({"type": "motion", "action": "goto_and_play", "index": idx})
        await self.wait_for(lambda s, d: s.get("playing") is True, 15.0, "playback to begin")
        t0 = time.monotonic()
        self.log("playing ...")
        await self.wait_for(lambda s, d: s.get("playing") is False, timeout, "playback to finish")
        self.log(f"done after {time.monotonic() - t0:.1f}s; robot blending back to planner idle")

    async def rescan(self) -> None:
        before = len(self.motions())
        self.log("-> rescan reference/")
        await self.send({"type": "motion", "action": "rescan"})
        await asyncio.sleep(2.0)
        self.log(f"motions: {before} -> {len(self.motions())}")

    async def stop(self) -> None:
        self.log("-> control stop  (policy stops, motors go to damping; robot must be supported)")
        await self.send({"type": "control", "action": "stop"})
        try:
            await self.wait_for(lambda s, d: d.get("sonic_running") is False, 10.0, "daemon to report sonic stopped")
        except (TimeoutError, ConnectionError) as e:
            self.log(str(e))
        self.log(f"daemon: {self.daemon_state}")

    async def emergency_stop(self) -> None:
        await self.send({"type": "control", "action": "emergency_stop"})


async def countdown(seconds: float, what: str) -> None:
    """Print a countdown before a robot-moving action. Ctrl-C aborts."""
    try:
        for s in range(int(seconds), 0, -1):
            print(f"{what} in {s}... (Ctrl-C to abort)", flush=True)
            await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        print("aborted")
        sys.exit(1)


async def prompt_loop(c: SonicDaemonClient) -> None:
    """Idle at a prompt; each motion plays to completion and the robot returns to planner idle."""
    print("\nSONIC running, robot in planner idle. Type a motion name/index, or: list, rescan, status, stop, q")
    c.print_motions()
    loop = asyncio.get_event_loop()
    while True:
        try:
            cmd = (await loop.run_in_executor(None, input, "sonic> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nleaving SONIC running")
            return
        if not cmd:
            continue
        if cmd in ("q", "quit", "exit"):
            print("leaving SONIC running")
            return
        try:
            if cmd in ("list", "?"):
                c.print_motions()
            elif cmd == "rescan":
                await c.rescan()
            elif cmd == "status":
                c.print_status()
            elif cmd == "stop":
                await c.stop()
                return
            elif cmd == "estop":
                await c.emergency_stop()
                await c.stop()
                return
            else:
                await c.play(cmd)
        except KeyError as e:
            print(f"  {e}")
        except (TimeoutError, ConnectionError) as e:
            print(f"  error: {e}")
            if c._closed:
                return


async def amain(args) -> int:
    async with SonicDaemonClient(args.host, args.port) as c:
        # give the daemon (and sonic, if running) a moment to broadcast
        try:
            await c.wait_for(lambda s, d: bool(d) or bool(s), 3.0, "daemon status")
            if c.daemon_state.get("sonic_running") and not c.sonic_live:
                await c.wait_for(lambda s, d: bool(s), 5.0, "first status frame from running sonic")
        except TimeoutError:
            pass

        if args.cmd == "status":
            await asyncio.sleep(1.0)
            c.print_status()
            if c.sonic_live:
                print("motions:")
                c.print_motions()
            return 0

        if args.cmd == "run":
            if not c.sonic_live:
                if not args.yes:
                    await countdown(args.countdown, "STARTING SONIC (robot will move)")
                await c.start()
                await c.settle(args.settle)
            if args.motion:
                await c.play(args.motion)
            await prompt_loop(c)
            return 0

        if args.cmd == "shell":
            if not c.sonic_live:
                print("SONIC is not running; use `run` to start it")
                return 1
            await prompt_loop(c)
            return 0

        if args.cmd == "play":
            await c.play(args.motion)
            return 0

        if args.cmd == "rescan":
            await c.rescan()
            return 0

        if args.cmd == "stop":
            if not args.yes:
                await countdown(args.countdown, "STOPPING (robot drops to damping)")
            await c.stop()
            return 0

        if args.cmd == "estop":
            c.log("-> emergency_stop, then control stop")
            await c.emergency_stop()
            await c.stop()
            return 0
    return 2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("-y", "--yes", action="store_true", help="skip the countdown before start/stop")
    p.add_argument("--countdown", type=float, default=1.0, help="seconds of countdown before start/stop (default 1)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    pr = sub.add_parser("run", help="start SONIC if needed, optionally play one motion, then prompt for more")
    pr.add_argument("motion", nargs="?", default=None)
    pr.add_argument("--settle", type=float, default=5.0, help="seconds to let the policy balance after start")
    sub.add_parser("shell", help="prompt for motions against an already-running SONIC")
    pp = sub.add_parser("play", help="play one motion, wait for completion, exit (SONIC keeps running)")
    pp.add_argument("motion")
    sub.add_parser("rescan")
    sub.add_parser("stop")
    sub.add_parser("estop", help="emergency_stop + stop, no countdown")
    args = p.parse_args()
    try:
        return asyncio.run(amain(args))
    except (OSError, TimeoutError, ConnectionError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 4
    except KeyboardInterrupt:
        print("\ninterrupted; SONIC left as is")
        return 130


if __name__ == "__main__":
    sys.exit(main())
