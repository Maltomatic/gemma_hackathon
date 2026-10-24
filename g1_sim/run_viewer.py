"""Launch the interactive MuJoCo viewer with the G1 holding its stand pose.

    python -m g1_sim.run_viewer                        # MJCF position actuators, stands
    python -m g1_sim.run_viewer --mode torque          # explicit PD at RL gains (SONIC path)
    python -m g1_sim.run_viewer --mode torque --gain-scale 4
    python -m g1_sim.run_viewer --headless 5           # no window, run 5 s and print stats

The controller is a HoldPoseController; swap it for a SONIC wrapper later.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .controllers import DEFAULT_RL_GAINS, HoldPoseController, PDGains
from .env import G1Sim


def run_headless(sim: G1Sim, controller, seconds: float) -> None:
    n = int(seconds / sim.control_dt)
    t0 = time.perf_counter()
    obs = sim.reset()
    for _ in range(n):
        obs = sim.step(controller)
    wall = time.perf_counter() - t0
    err = np.abs(obs.q - controller.target_q)
    print(f"simulated {seconds:.1f}s in {wall:.2f}s wall ({seconds / wall:.0f}x realtime)")
    print(f"pelvis height {obs.base_pos[2]:.3f} m   proj_gravity {np.round(obs.proj_gravity, 3)}")
    print(f"joint tracking error: max {err.max():.4f} rad, mean {err.mean():.4f} rad")
    print(f"imu gyro {np.round(obs.imu_gyro, 4)}  accel {np.round(obs.imu_accel, 3)}")


def run_viewer(sim: G1Sim, controller, max_seconds: float | None = None) -> None:
    import mujoco.viewer

    obs = sim.reset()
    target_q = controller(obs)
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 140
        viewer.cam.lookat[:] = (0.0, 0.0, 0.75)
        ticks = 0
        t_launch = time.perf_counter()
        # One viewer frame per control tick (50 Hz): syncing the window every
        # 2 ms physics substep costs more than the physics and drops below realtime.
        while viewer.is_running():
            t_start = time.perf_counter()
            target_q = controller(sim.observe())
            for _ in range(sim.decimation):
                sim.substep(target_q)
            ticks += 1
            viewer.sync()
            if max_seconds is not None and t_start - t_launch > max_seconds:
                break
            sleep = sim.control_dt - (time.perf_counter() - t_start)
            if sleep > 0:
                time.sleep(sleep)
    wall = time.perf_counter() - t_launch
    print(f"viewer closed after {ticks} ticks: sim time {sim.data.time:.2f}s in {wall:.2f}s wall "
          f"({sim.data.time / wall:.2f}x realtime), pelvis height {sim.data.xpos[sim.pelvis_id][2]:.3f} m")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["torque", "position"], default="position",
                   help="position: MJCF position actuators (kp=500, stands on its own). "
                        "torque: explicit PD at RL gains, the path SONIC will use; with the "
                        "hold-pose controller the ankles sag and it tips over in ~3 s "
                        "because joint PD alone does not balance (see --gain-scale)")
    p.add_argument("--gain-scale", type=float, default=1.0,
                   help="torque mode only: multiply kp by this (kd by sqrt). 4 holds the stand pose")
    p.add_argument("--hz", type=float, default=50.0, help="policy/controller rate")
    p.add_argument("--headless", type=float, metavar="SECONDS", default=None,
                   help="run without a window for this many sim seconds, then print stats")
    p.add_argument("--auto-close", type=float, metavar="SECONDS", default=None,
                   help="close the viewer window after this many wall seconds (for smoke tests)")
    args = p.parse_args()

    gains = PDGains(kp=DEFAULT_RL_GAINS.kp * args.gain_scale,
                    kd=DEFAULT_RL_GAINS.kd * np.sqrt(args.gain_scale))
    sim = G1Sim(mode=args.mode, control_hz=args.hz, gains=gains)
    controller = HoldPoseController(sim.stand_q)
    print(f"G1Sim: mode={sim.mode} sim_dt={sim.sim_dt*1e3:.0f}ms control_dt={sim.control_dt*1e3:.0f}ms "
          f"(decimation {sim.decimation}), nq={sim.model.nq} nu={sim.model.nu}")
    if args.headless is not None:
        run_headless(sim, controller, args.headless)
    else:
        run_viewer(sim, controller, max_seconds=args.auto_close)


if __name__ == "__main__":
    main()
