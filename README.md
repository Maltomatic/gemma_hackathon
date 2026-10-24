# Unitree G1 in MuJoCo

MuJoCo simulation of the Unitree G1 (29-DoF, no hands) with a controller hook
ready for SONIC or any other whole-body policy.

## Layout

```
.venv/                              Python 3.13 venv (mujoco 3.17, numpy, glfw)
third_party/mujoco_menagerie/       sparse clone, only unitree_g1/ checked out (BSD-3)
g1_sim/
  env.py          G1Sim: model/data owner, 50 Hz control tick over 2 ms physics,
                  observation builder, position- or torque-mode actuation
  controllers.py  Controller protocol, HoldPoseController, PD gains, torque limits,
                  ACTUATED_JOINTS (the canonical 29-joint order)
  run_viewer.py   CLI: interactive viewer or headless benchmark
```

## Run

```bash
.venv\Scripts\python -m g1_sim.run_viewer
```

Opens the MuJoCo viewer with the G1 standing in its `stand` keyframe.
Useful flags:

| flag | what it does |
|---|---|
| `--headless 10` | no window; simulate 10 s and print realtime factor, pelvis height, tracking error |
| `--mode torque` | explicit PD loop in Python at RL-style gains, the path a sim-to-real policy uses |
| `--gain-scale 4` | torque mode only; scales kp (kd by sqrt). At 1x the hold-pose demo tips over, at 4x it stands |
| `--hz 50` | controller rate (default 50 Hz = decimation 10 over the 2 ms physics step) |
| `--auto-close 5` | close the window after 5 s wall time (smoke tests) |

Headless numbers on this laptop (CPU only): roughly 40x realtime in torque mode,
60x in position mode. The viewer runs at realtime.

## Control modes

The Menagerie MJCF ships `<position kp=500>` actuators. `G1Sim` supports two ways
to drive them from a vector of 29 target joint angles:

* **position** (viewer default): `ctrl = target_q`, MuJoCo's own stiff PD. Stands
  on its own. Good for sanity checks and visualising poses.
* **torque** (`G1Sim` default): at load time the actuators are rewritten into
  pure torque motors (`gain=1, bias=0`, ctrlrange = joint torque limit) and
  `tau = kp*(q_des - q) - kd*dq` is computed in Python every physics substep.
  This is what a policy trained with motor-level PD expects. With the plain
  hold-pose controller and RL-level gains the ankles sag ~0.28 rad, the center
  of mass drifts forward and the robot falls in ~3 s. That is expected: joint
  PD does not balance, the policy does.

## Wiring in SONIC

Everything SONIC needs is already exposed; the sim does not change.

1. **Controller.** Implement the `Controller` protocol from `g1_sim.controllers`:
   `__call__(obs: G1Obs) -> np.ndarray[29]` of target joint positions in
   `ACTUATED_JOINTS` order (legs L/R, waist, left arm, right arm, which matches
   the Unitree SDK order). Build the policy input from `G1Obs`: `q`, `dq`,
   `base_ang_vel`, `proj_gravity`, `last_target_q`, plus whatever command /
   motion-reference SONIC takes. `imu_gyro` / `imu_accel` are the noisy sensor
   versions if you want hardware-like inputs. `base_pos` / `base_lin_vel` are
   sim-only ground truth.
2. **Gains.** SONIC ships per-joint kp/kd in its deployment config. Load them
   with `PDGains.from_joint_dicts(kp, kd)` and pass `G1Sim(mode="torque",
   gains=...)`. The `DEFAULT_RL_GAINS` in `controllers.py` are a placeholder in
   the unitree_rl_gym range (hip 100, knee 150, ankle 40, waist 150, arm 40,
   wrist 20), not SONIC's.
3. **Rate.** `G1Sim(control_hz=...)` to match SONIC's policy frequency; the PD
   still runs at 500 Hz underneath.
4. **Loop.** `obs = sim.reset(); while ...: obs = sim.step(policy)`, or use
   `run_viewer.run_viewer(sim, policy)` to watch it.

Also note `g1_mjx.xml` / `scene_mjx.xml` in the Menagerie folder: a lighter
collision model with lower gains, used by MuJoCo Playground and verified on
hardware. Worth switching to if SONIC's reference setup used it.

## Rebuild from scratch

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie sparse-checkout set unitree_g1
```
