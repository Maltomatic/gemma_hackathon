"""Run the G1 under SONIC in MuJoCo, tracking a reference motion clip.

    python -m g1_sim.run_sonic                       # viewer, sample walking clip
    python -m g1_sim.run_sonic --headless 12         # no window, print tracking stats
    python -m g1_sim.run_sonic --motion path/to.pkl --clip name --loop

Timeline: the robot settles in SONIC's default stance, the policy holds the clip's first
frame for ``--hold`` seconds (so it walks into the pose), then playback starts.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .env import G1Sim, REPO_ROOT
from .sonic_controller import SonicController, sonic_reset
from .sonic_motion import load_reference_motions

DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "sonic" / "sonic_v1_1"
DEFAULT_MOTION = REPO_ROOT / "models" / "sonic" / "sample_data" / "robot_filtered" / "210531" / "walk_forward_amateur_001__A001.pkl"


def make(args) -> tuple[G1Sim, SonicController]:
    clips = load_reference_motions(args.motion)
    name = args.clip or next(iter(clips))
    if name not in clips:
        raise SystemExit(f"clip {name!r} not in {list(clips)}")
    motion = clips[name]
    sim = G1Sim(mode="torque", gains=SonicController.pd_gains(not args.no_ankle_boost), control_hz=50.0)
    ctrl = SonicController(args.model_dir, motion=motion, ankle_gain_boost=not args.no_ankle_boost, loop=args.loop)
    print(f"clip {name}: {motion.num_frames} frames @ {motion.fps:.0f} Hz ({motion.duration:.1f}s)")
    return sim, ctrl


def run_headless(sim: G1Sim, ctrl: SonicController, seconds: float, hold: float) -> None:
    obs = sonic_reset(sim, ctrl)
    n_hold = int(hold / sim.control_dt)
    n_total = int(seconds / sim.control_dt)
    t0 = time.perf_counter()
    infer = []
    min_h, max_tilt, errs = 1e9, 0.0, []
    for i in range(n_total):
        if i == n_hold:
            ctrl.play(True)
            print(f"t={obs.time:.2f}s  play")
        t = time.perf_counter()
        obs = sim.step(ctrl)
        infer.append(time.perf_counter() - t)
        h = obs.base_pos[2]
        tilt = float(np.degrees(np.arccos(np.clip(-obs.proj_gravity[2], -1, 1))))
        min_h, max_tilt = min(min_h, h), max(max_tilt, tilt)
        errs.append(ctrl.tracking_error(obs).mean())
        if i % 50 == 0:
            print(f"t={obs.time:5.2f}s frame={ctrl.frame:4d} pelvis z={h:.3f} tilt={tilt:5.1f}deg "
                  f"track err={errs[-1]:.3f} rad pos=({obs.base_pos[0]:+.2f},{obs.base_pos[1]:+.2f})")
    wall = time.perf_counter() - t0
    print(f"\nsimulated {seconds:.1f}s in {wall:.1f}s ({seconds / wall:.1f}x realtime), "
          f"policy tick {1e3 * np.mean(infer):.1f} ms avg")
    print(f"min pelvis height {min_h:.3f} m, max tilt {max_tilt:.1f} deg, "
          f"mean tracking err {np.mean(errs[n_hold:]):.3f} rad, final pos {np.round(obs.base_pos, 2)}")
    fell = min_h < 0.4 or max_tilt > 60
    print("RESULT:", "FELL" if fell else "STAYED UP")


def run_viewer(sim: G1Sim, ctrl: SonicController, hold: float, max_seconds: float | None) -> None:
    import mujoco.viewer

    obs = sonic_reset(sim, ctrl)
    print("viewer: policy holds first frame, playback starts after hold. Close window to exit.")
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 3.0, -15, 140
        t_launch = time.perf_counter()
        ticks = 0
        while viewer.is_running():
            t_start = time.perf_counter()
            if not ctrl.playing and obs.time >= hold:
                ctrl.play(True)
            target = ctrl(sim.observe())
            for _ in range(sim.decimation):
                sim.substep(target)
            obs = sim.observe()
            sim.last_target_q = target
            viewer.cam.lookat[:] = obs.base_pos
            viewer.sync()
            ticks += 1
            if max_seconds is not None and t_start - t_launch > max_seconds:
                break
            sleep = sim.control_dt - (time.perf_counter() - t_start)
            if sleep > 0:
                time.sleep(sleep)
    wall = time.perf_counter() - t_launch
    print(f"viewer closed: sim {sim.data.time:.1f}s in {wall:.1f}s wall ({sim.data.time / wall:.2f}x), "
          f"pelvis z={obs.base_pos[2]:.3f}, frame {ctrl.frame}/{ctrl.motion.num_frames}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--motion", type=Path, default=DEFAULT_MOTION, help="robot_filtered pickle")
    p.add_argument("--clip", default=None, help="clip name inside the pickle (default: first)")
    p.add_argument("--loop", action="store_true", help="loop the clip instead of holding its last frame")
    p.add_argument("--hold", type=float, default=2.0, help="seconds to hold the first frame before playing")
    p.add_argument("--no-ankle-boost", action="store_true", help="disable the v1.1 ankle-pitch 1.5x gain tuning")
    p.add_argument("--headless", type=float, metavar="SECONDS", default=None)
    p.add_argument("--auto-close", type=float, metavar="SECONDS", default=None)
    args = p.parse_args()

    sim, ctrl = make(args)
    if args.headless is not None:
        run_headless(sim, ctrl, args.headless, args.hold)
    else:
        run_viewer(sim, ctrl, args.hold, args.auto_close)


if __name__ == "__main__":
    main()
