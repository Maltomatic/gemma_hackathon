"""TCP launcher for the real G1: same wire protocol as scripts/sonic_launcher.py, but the
motion runs on the robot through the team's sonic-daemon instead of in the MuJoCo sim.

    client -> "3\\n"   play trajectory 3 (ufb_forward) on the robot
    server -> "0\\n"   sent when the motion has finished and the robot is back in planner idle

    client -> "?\\n"   list trajectories (one "idx name ok|missing" line each, then ".")

Status codes (superset of the sim launcher's):
    0 done   1 unknown trajectory / not on robot   2 busy   3 playback failed or timed out
    5 SONIC not running (start it with robot_sonic.py run, or launch this with --start)

Trajectory table (same numbers as the sim): 0 tap, 1 shake, 2 slap, 3 forward, 4 left, 5 right
-> reference/ufb_<name> on the robot. A non-numeric request is treated as a motion name
(e.g. "01_emerson_move"), matched exactly or by unique substring.

Run it on the laptop (default) or copy it next to robot_sonic.py onto the robot and point
--daemon-host at 127.0.0.1. SONIC must already be running (robot_sonic.py run), or pass
--start to launch it here after a 3 s countdown. This launcher never stops SONIC: after each
motion the robot returns to planner idle and waits for the next request.

    python scripts/robot_launcher.py                       # TCP :5555 -> daemon ws://192.168.123.164:8765
    python scripts/robot_launcher.py --start               # also start SONIC first (robot moves!)
    python scripts/client_sample.py <this-machine-ip> 0    # from any device, unchanged client
"""
from __future__ import annotations

import argparse
import asyncio
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_sonic import SonicDaemonClient, countdown  # noqa: E402

TRAJECTORIES: dict[int, str] = {0: "ufb_tap", 1: "ufb_shake", 2: "ufb_slap", 3: "ufb_forward", 4: "ufb_left", 5: "ufb_right"}
STATUS_DONE, STATUS_UNKNOWN, STATUS_BUSY, STATUS_FAILED, STATUS_NO_SONIC = 0, 1, 2, 3, 5


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Bridge:
    def __init__(self, daemon_host: str, daemon_port: int, play_timeout: float):
        self.daemon_host, self.daemon_port = daemon_host, daemon_port
        self.play_timeout = play_timeout
        self.client: SonicDaemonClient | None = None
        self.busy = asyncio.Lock()
        self.current: str | None = None

    async def connect(self) -> SonicDaemonClient:
        """(Re)connect to the daemon if needed."""
        if self.client is None or self.client._closed:
            if self.client is not None:
                await self.client.__aexit__(None, None, None)
            self.client = await SonicDaemonClient(self.daemon_host, self.daemon_port, verbose=False).__aenter__()
            try:
                await self.client.wait_for(lambda s, d: bool(d) or bool(s), 3.0, "daemon")
                if self.client.daemon_state.get("sonic_running") and not self.client.sonic_live:
                    await self.client.wait_for(lambda s, d: bool(s), 5.0, "status")
            except TimeoutError:
                pass
            log(f"daemon ws://{self.daemon_host}:{self.daemon_port}: {self.client.daemon_state or 'no daemon_status'}, "
                f"sonic {'live' if self.client.sonic_live else 'not running'}")
        return self.client

    def resolve(self, req: str) -> str | None:
        if req.isdigit():
            return TRAJECTORIES.get(int(req))
        return req  # motion name

    async def run(self, req: str) -> int:
        name = self.resolve(req)
        if name is None:
            return STATUS_UNKNOWN
        if self.busy.locked():
            return STATUS_BUSY
        async with self.busy:
            try:
                c = await self.connect()
            except (OSError, ConnectionError) as e:
                log(f"cannot reach daemon: {e}")
                return STATUS_FAILED
            if not c.sonic_live:
                return STATUS_NO_SONIC
            try:
                idx = c.motion_index(name)
            except KeyError:
                return STATUS_UNKNOWN
            self.current = c.motions()[idx]
            log(f"start  {req} -> {self.current}")
            t0 = time.monotonic()
            try:
                await c.play(self.current, timeout=self.play_timeout, assume_done_on_timeout=True)
            except (TimeoutError, ConnectionError, KeyError) as e:
                log(f"failed {self.current}: {e}")
                return STATUS_FAILED
            except asyncio.CancelledError:
                log(f"wait for {self.current} cancelled")
                raise
            finally:
                self.current = None
            log(f"done   {req} ({time.monotonic() - t0:.1f}s), robot back in planner idle")
            return STATUS_DONE

    async def listing(self) -> str:
        try:
            c = await self.connect()
            have = set(c.motions()) if c.sonic_live else None
        except (OSError, ConnectionError):
            have = None
        if have is None:  # no motion list without a running SONIC
            lines = [f"{i} {n} unknown(SONIC not running)" for i, n in TRAJECTORIES.items()]
        else:
            lines = [f"{i} {n} {'ok' if n in have else 'missing'}" for i, n in TRAJECTORIES.items()]
        return "\n".join(lines) + "\n.\n"


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, bridge: Bridge) -> None:
    peer = writer.get_extra_info("peername")
    log(f"client {peer[0]}:{peer[1]} connected")
    pending: bytes | None = None             # a line that arrived while a motion was running
    read_task: asyncio.Task | None = None    # one outstanding readline() at a time, reused across turns
    try:
        while True:
            if pending is not None:
                raw, pending = pending, None
            else:
                if read_task is None:
                    read_task = asyncio.create_task(reader.readline())
                raw = await read_task
                read_task = None
            if not raw:
                break
            msg = raw.decode("utf-8", "replace").strip()
            if not msg:
                continue
            if msg in ("?", "list"):
                writer.write((await bridge.listing()).encode())
                await writer.drain()
                continue

            # Run the motion while also watching the socket: if the client goes away
            # (Ctrl-C) we cancel the wait so the busy lock is released immediately.
            run_task = asyncio.create_task(bridge.run(msg))
            if read_task is None:
                read_task = asyncio.create_task(reader.readline())
            done, _ = await asyncio.wait({run_task, read_task}, return_when=asyncio.FIRST_COMPLETED)
            if run_task not in done:
                nxt = read_task.result()
                read_task = None
                if not nxt:  # EOF: client disconnected mid-motion
                    run_task.cancel()
                    log(f"client {peer[0]} left during {msg!r}; wait cancelled (motion itself continues on the robot)")
                    break
                pending = nxt  # client sent another request early; serve it after this one
                await run_task
            elif read_task.done():
                pending = read_task.result() or None
                read_task = None
            # else: read_task still pending; it becomes the next request's read
            status = run_task.result()
            if status != STATUS_DONE:
                log(f"client {peer[0]}: {msg!r} -> status {status}")
            writer.write(f"{status}\n".encode())
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
        pass
    finally:
        if read_task is not None and not read_task.done():
            read_task.cancel()
        writer.close()
        log(f"client {peer[0]}:{peer[1]} closed")


async def amain(args) -> None:
    bridge = Bridge(args.daemon_host, args.daemon_port, args.play_timeout)
    c = await bridge.connect()
    if args.start and not c.sonic_live:
        if not args.yes:
            await countdown(args.countdown, "STARTING SONIC on the robot (robot will move)")
        c.verbose = True
        await c.start()
        await asyncio.sleep(args.settle)
        c.verbose = False
    if not c.sonic_live:
        log("SONIC is not running; requests will get status 5 until it is (robot_sonic.py run, or --start)")

    server = await asyncio.start_server(lambda r, w: handle(r, w, bridge), args.host, args.port)
    ips = sorted({i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)})
    log(f"robot launcher listening on {args.host}:{args.port}  (this machine: {', '.join(ips)})")
    log("trajectories:\n" + (await bridge.listing()).rstrip("\n."))
    async with server:
        await server.serve_forever()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--daemon-host", default="192.168.123.164", help="robot running sonic-daemon (127.0.0.1 if this runs on the robot)")
    p.add_argument("--daemon-port", type=int, default=8765)
    p.add_argument("--start", action="store_true", help="start SONIC via the daemon on launch if it is not running")
    p.add_argument("--settle", type=float, default=5.0, help="seconds to let the policy balance after --start")
    p.add_argument("--countdown", type=float, default=1.0, help="seconds of countdown before --start (default 1)")
    p.add_argument("-y", "--yes", action="store_true", help="skip the countdown for --start")
    p.add_argument("--play-timeout", "--max-motion-seconds", dest="play_timeout", type=float, default=8.0,
                   help="if no completion is reported within this many seconds, reply 0 anyway "
                        "(all demo clips are under 5.2 s plus a 1 s blend-in; default 8)")
    args = p.parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        log("shutting down (SONIC left as is)")


if __name__ == "__main__":
    main()
