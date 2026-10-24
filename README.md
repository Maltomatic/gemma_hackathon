# Unitree G1 in MuJoCo, driven by NVIDIA SONIC

MuJoCo simulation of the Unitree G1 (29-DoF, no hands) with a controller hook, and a
pure-Python port of NVIDIA's [GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)
whole-body controller running on CPU through onnxruntime. No Linux, CUDA, or TensorRT needed.

## Setup

```bash
git clone https://github.com/Maltomatic/gemma_hackathon.git
cd gemma_hackathon
uv venv .venv --python 3.13
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie sparse-checkout set unitree_g1
.venv\Scripts\python scripts/download_sonic.py     # ~200 MB from huggingface.co/nvidia/GEAR-SONIC
```

On macOS / Linux replace `.venv\Scripts\python` with `.venv/bin/python`. Python 3.13 and
`uv` are what this was built with; anything 3.10+ with `pip` should work too.

## Run

```bash
.venv\Scripts\python -m g1_sim.run_sonic
```

Opens the MuJoCo viewer. The G1 settles into SONIC's default stance, the policy holds the
first frame of the sample walking clip for 2 s (so it walks into the pose), then playback
starts and the robot tracks the clip. Close the window to exit.

| flag | what it does |
|---|---|
| `--headless 12` | no window; run 12 sim seconds and print height / tilt / tracking error |
| `--motion file.pkl --clip name` | another `robot_filtered` clip from the GEAR-SONIC dataset |
| `--motion file.csv` | a Unitree LAFAN1 retargeting CSV (36 cols: root xyz, quat xyzw, 29 joints, 30 fps) |
| `--loop` | loop the clip instead of holding its last frame |
| `--hold 2` | seconds the policy holds frame 0 before playing |
| `--no-ankle-boost` | drop the v1.1 recommended 1.5x ankle-pitch gain tuning |

## Remote trigger: `scripts/sonic_launcher.py`

A TCP launcher so another app on another device can fire trajectories:

```bash
.venv\Scripts\python scripts/sonic_launcher.py              # listens on 0.0.0.0:5555
.venv\Scripts\python scripts/sonic_launcher.py --headless   # no window
```

Protocol, newline-terminated ASCII on one connection (reuse it for the next request):

| client sends | server replies | meaning |
|---|---|---|
| `0` .. `5` | `0` when done | run that trajectory once in a fresh sim process, reply when it exits |
| `?` | one `idx name ok|missing` line each, then `.` | list trajectories |

Trajectories: 0 tap, 1 shake, 2 slap, 3 forward, 4 left, 5 right (`motion_demo/<name>.csv`).
Non-zero replies: `1` unknown number or missing CSV, `2` busy (one at a time), `3` sim process failed.
Each run takes hold (2 s default, `--hold`) + clip length + 0.5 s end hold; the viewer window
opens and closes per request. Allow the port through Windows Firewall for other devices.

Smoke test from another machine: `printf '0\n' | nc <laptop-ip> 5555` prints `0` when the tap ends.

The plain sim without SONIC is still there for debugging the env itself:

```bash
.venv\Scripts\python -m g1_sim.run_viewer --headless 10     # hold-pose controller
```

## Results on this laptop (CPU only, no GPU)

`python -m g1_sim.run_sonic --headless 44 --hold 2` on the sample clip
`walk_forward_amateur_001__A001` (stand 8 s, walk ~8 m out and back, stand):

| metric | value |
|---|---|
| outcome | stayed up for the whole 42 s clip |
| distance walked | ~6.8 m out, back to ~0.5 m from start |
| min pelvis height / max tilt | 0.74 m / 9.6 deg |
| mean joint tracking error (playing) | 0.076 rad (0.03 standing, ~0.07 walking) |
| policy tick (encoder + decoder, onnxruntime CPU) | 6 to 7 ms |
| headless speed | 2.7x realtime |
| viewer | realtime (1.0x) |

Other clips, same command with `--motion`:

| clip | source | length | outcome | mean track err | max tilt |
|---|---|---|---|---|---|
| `motion_demo/tap.csv` | LAFAN CSV (in repo) | 5 s, also looped 20 s | stayed up, no drift | 0.07 rad | 11 deg |
| `take_on_me_iris.csv` | LAFAN CSV (dance) | 89 s | stayed up | 0.14 rad | 19 deg |

Gotchas seen so far: the default-stance settle takes ~0.5 s and the policy needs ~2 s of
holding frame 0 to walk into the clip's start pose before playback (`--hold`). When the clip
ends and the last frame is held, the tracking error rises to ~0.2 rad as the policy relaxes
toward its own stance; pass `--loop` to keep moving instead.

## Layout

```
g1_sim/
  env.py               G1Sim: owns model/data, 50 Hz control tick over 2 ms physics,
                       G1Obs (joints, pelvis quat / ang vel, projected gravity, IMU)
  controllers.py       Controller protocol, HoldPoseController, PDGains, torque limits,
                       ACTUATED_JOINTS (hardware / Unitree SDK order)
  sonic_params.py      Everything transcribed from the SONIC deploy stack: Isaac Lab <-> hardware
                       joint maps, armature-derived kp/kd, action scales, default angles,
                       encoder/decoder tensor layouts
  sonic_motion.py      Load GEAR-SONIC robot_filtered pickles or LAFAN1 CSVs, resample
                       30 -> 50 Hz, Isaac Lab joint order, forward-difference velocities
  sonic_controller.py  SonicController: 10-frame proprio history, g1-mode encoder input,
                       ONNX encoder -> 64-d token -> ONNX decoder -> 29-d action -> PD target
  quat.py              wxyz quaternion helpers (heading, 6-D rotation, slerp)
  run_sonic.py         SONIC demo CLI;  run_viewer.py  env-only CLI
scripts/download_sonic.py   parallel range-request downloader for the HF checkpoint
scripts/sonic_launcher.py   TCP launcher: trajectory number in, 0 out when the run finishes
motion_demo/*.csv           six LAFAN-format demo clips (videos next to them are gitignored)
models/sonic/               (gitignored) sonic_v1_1/*.onnx, configs, sample_data/
third_party/mujoco_menagerie/unitree_g1/   (gitignored) Menagerie G1 MJCF, BSD-3
```

## How the SONIC port works

SONIC's released checkpoint is two ONNX graphs plus an observation config. The official
runtime is C++ with TensorRT talking to a MuJoCo sim over Unitree SDK2 DDS, Linux and NVIDIA
GPU only. This repo re-implements the observation pipeline from that C++ code in
`sonic_controller.py` and runs the same ONNX files with onnxruntime on CPU.

Per 50 Hz tick:

1. **Proprio history.** The current pelvis angular velocity (body frame), joint positions
   minus default angles, joint velocities, previous raw action, and gravity direction in the
   pelvis frame are pushed into 10-frame buffers (oldest first), all in Isaac Lab joint order.
2. **Encoder.** The next 10 reference frames spaced 100 ms apart are packed as joint
   positions, joint velocities, and the reference pelvis orientation expressed relative to
   the robot's *current heading* (yaw only) as a 6-D rotation. The clip is yaw-aligned to
   the robot's heading when control starts. Teleop / SMPL slots stay zero (g1 mode). The
   encoder returns a 64-d motion token.
3. **Decoder.** token + histories (994 floats) -> 29 actions.
4. **Action.** `q_target = default_angles + action * action_scale` in hardware order;
   `G1Sim` applies `tau = kp (q_target - q) - kd dq` at 500 Hz with SONIC's motor-armature
   gains (hips/knees ~99 Nm/rad, ankles ~28, arms ~14) and the v1.1 ankle-pitch 1.5x tuning.

Isaac Lab orders joints breadth-first (left_hip_pitch, right_hip_pitch, waist_yaw, ...),
the hardware orders them by limb. `sonic_params.MUJOCO_TO_ISAACLAB` / `ISAACLAB_TO_MUJOCO`
carry the remap and are asserted against joint names at import.

## Swapping the controller

Anything that satisfies `g1_sim.controllers.Controller` (`__call__(obs) -> 29 target joint
angles in hardware order`) plugs into `G1Sim.step`. `HoldPoseController` is the trivial
example, `SonicController` the real one. For a different policy, pass its PD gains via
`G1Sim(gains=PDGains(...))`.

## Licenses

Code here: your choice. Menagerie G1 model: BSD-3 (Google DeepMind). SONIC weights:
NVIDIA Open Model License (see `models/sonic/LICENSE` after download). SONIC source
that this port is derived from: Apache 2.0.
