# Running SONIC on the real G1 (g1-7644)

Survey of the robot as found on 2026-10-24, and the procedure to launch SONIC and play a
recorded motion. Nothing here was executed on the robot; `scripts/robot_sonic.py status`
was the only command run against it (a read-only websocket query).

## What is on the robot

| item | state |
|---|---|
| hardware | Jetson Orin, JetPack 6 (L4T R36.4.7), Ubuntu 22.04, 8 cores, 15 GB |
| network | `enP8p1s0` 192.168.123.164/24 (Unitree bus), Tailscale 100.68.183.20, Wi-Fi 192.168.8.213 |
| repo | `~/ufb/GR00T-WholeBodyControl-ufb` at commit `06adea0 G1 config` (team fork, see `UFB_CHANGES.md`) |
| binary | `gear_sonic_deploy/target/release/g1_deploy_onnx_ref` built 2026-07-16; `just` 1.43 present; TensorRT at `~/TensorRT` |
| policy | `policy/release/` = **default SONIC release** (decoder 40.9 MB, 10-frame history, g1/teleop/smpl encoder modes). TensorRT engines already cached next to the ONNX files, so startup does not re-convert |
| planner | `planner/target_vel/V2/planner_sonic.onnx` + cached `.trt` |
| references | `gear_sonic_deploy/reference/` has 28 motion folders incl. `01_emerson_move` (239 frames @ 50 Hz = 4.8 s, converted from a 144-frame 30 fps LAFAN CSV) and `99_idle` |
| daemon | `sonic-daemon.service` (systemd, user unitree, `uv run daemon.py` in `gear_sonic_deploy/robot/`) **running**, listening on ws://0.0.0.0:8765. SONIC itself is **not** running |
| other | UFB `robot-client` docker container running as root with `policy-runner damp-g1 --backend damp`; its `ROBOT_UNITREE_LOW_CMD_WRITER` defaults to `disabled` (shadow mode) because there is no `.env`, so it should not be commanding motors |

Working-tree oddities worth knowing (not fixed, per "do not overwrite"):

- `gear_sonic_deploy/reference/convert_motions.py` is **deleted** in the working tree. The same
  converter (with `--convert_lafan_csv`) lives at `~/ufb/sonic_motions/convert_motions.py`, and
  `git show HEAD:gear_sonic_deploy/reference/convert_motions.py` still has it.
- `robot/daemon.py` has a one-line uncommitted change (tomllib import fallback). The daemon runs
  under Python 3.14 in `robot/.venv`, so that fallback is harmless.
- `01_emerson_move` and seven other reference folders are untracked (added 2026-10-09).

## How launching works (from `robot/daemon.py` + `daemon_config.toml`)

```
you  --ws:8765-->  daemon.py  --spawn-->  just run g1_deploy_onnx_ref real policy/release/model_decoder.onnx reference/
                                              --obs-config policy/release/observation_config.yaml
                                              --encoder-file policy/release/model_encoder.onnx
                                              --planner-file planner/target_vel/V2/planner_sonic.onnx
                                              --input-type websocket --output-type all --zmq-host localhost --ws-port 8766
                   daemon.py  <--ws:8766-->  sonic binary (proxied both ways)
```

1. Client sends `{"type":"control","action":"start"}`.
2. Daemon spawns the command above (cwd `gear_sonic_deploy/`), waits for the binary's websocket on
   8766, then waits for the binary to print `Init Done` (timeout 300 s).
   **The binary's INIT state ramps the robot to the default standing pose over 3 s as soon as it
   starts.** This is the first physical movement.
3. Daemon forwards `start` to the binary: INIT → CONTROL. The policy balances, the planner is
   auto-enabled, `play` is set, the robot stands on its own (planner IDLE).
4. Binary broadcasts `{"type":"status", ..., "motion_index", "motion_name", "playing", "motions":[names]}`
   at ~5 Hz. `motions` is the list of reference folders found recursively under `reference/`
   (relative path as name, e.g. `01_emerson_move`, `ted/02_bruce_lee_punch`).
5. `{"type":"motion","action":"goto_and_play","index":i}` plays reference `i` once with the team's
   1 s blend-in; on completion it blends back to planner idle and `playing` goes false.
6. `{"type":"control","action":"stop"}`: daemon forwards stop (binary writes its damping-only
   safe-shutdown command), waits 1.5 s, kills the process.

Other accepted messages: `motion/play`, `motion/restart`, `motion/goto`, `motion/rescan`
(re-reads `reference/` without restarting), `control/emergency_stop`, `control/reinitialize`,
`locomotion/...`, `heading/...`.

## Pre-start checklist (physical)

Same as the team's browser-GUI flow and the upstream teleop docs:

1. Robot hanging **slack on the gantry**, feet just touching or slightly off the floor.
2. Native Unitree controller released (the robot in the damping / debug state you normally use
   before the GUI Start). The binary owns `rt/lowcmd` from the moment it starts.
3. Confirm the UFB container is not writing low-level commands:
   `sudo docker exec $(sudo docker ps -q) env | grep ROBOT_UNITREE_LOW_CMD_WRITER` should print `disabled`.
4. `scripts/robot_sonic.py status` from the laptop shows `daemon_state: idle` and no sonic status
   (SONIC not already running).
5. Someone at the e-stop. `stop` drops the robot to damping, so be ready to take its weight.

## Launch and play `01_emerson_move`

From this repo on the laptop (robot reachable on 192.168.123.164):

```bash
.venv\Scripts\python scripts/robot_sonic.py status
.venv\Scripts\python scripts/robot_sonic.py start            # asks for YES; robot ramps to default pose, policy starts
.venv\Scripts\python scripts/robot_sonic.py play 01_emerson_move   # waits until playing goes false (~5 s + blends)
.venv\Scripts\python scripts/robot_sonic.py stop             # asks for YES; robot goes to damping
```

or in one go: `scripts/robot_sonic.py run 01_emerson_move --settle 5`.

Logs while it runs: `ssh unitree@192.168.123.164 journalctl -u sonic-daemon -f` (the binary's
stdout is forwarded there with a `[sonic]` prefix). Per-run CSV logs land in
`gear_sonic_deploy/logs/<date>/<time>/`.

## Getting a recorded motion onto the robot (new files only)

The binary consumes a folder of `joint_pos.csv`, `joint_vel.csv`, `body_pos.csv`,
`body_quat.csv`, `body_lin_vel.csv`, `body_ang_vel.csv`, `metadata.txt` (+ optional `info.txt`),
50 Hz, joints in Isaac Lab order. That is exactly what the `.zip` files in `motion_demo/` hold
(`tap.zip` = one such folder), and what `convert_motions.py --convert_lafan_csv` produces.

**All six demo motions in one go** (the zips all carry the same stale inner folder name, so
`unzip -j` flattens each into its own `ufb_<name>` folder; `-n` never overwrites):

```bash
scp motion_demo/*.zip unitree@192.168.123.164:/tmp/
ssh unitree@192.168.123.164 'cd ~/ufb/GR00T-WholeBodyControl-ufb/gear_sonic_deploy/reference && for m in tap shake slap forward left right; do mkdir -p ufb_$m && unzip -n -j /tmp/$m.zip -d ufb_$m >/dev/null && echo "ufb_$m: $(wc -l < ufb_$m/joint_pos.csv) rows"; done'
```

Motion names on the robot are then `ufb_tap`, `ufb_shake`, `ufb_slap`, `ufb_forward`, `ufb_left`, `ufb_right`.

**From a single `.zip` (already in deploy format):**

```bash
# laptop -> robot, into a NEW folder; pick a name that does not exist yet under reference/
scp motion_demo/tap.zip unitree@192.168.123.164:/tmp/
ssh unitree@192.168.123.164 "cd ~/ufb/GR00T-WholeBodyControl-ufb/gear_sonic_deploy/reference && mkdir ufb_tap && unzip -n /tmp/tap.zip -d ufb_tap"
.venv\Scripts\python scripts/robot_sonic.py rescan        # if SONIC is already running; otherwise it is picked up at start
.venv\Scripts\python scripts/robot_sonic.py play ufb_tap
```

`unzip -n` never overwrites. The zip's inner folder becomes the motion name
(`ufb_tap/tap_something_in_front_...`); substring matching in `robot_sonic.py play` handles that.

**From a LAFAN `.csv` (36 columns, 30 fps):** convert on the robot with the team's converter,
writing to a new folder:

```bash
scp motion_demo/shake.csv unitree@192.168.123.164:/tmp/
ssh unitree@192.168.123.164 "cd ~/ufb/GR00T-WholeBodyControl-ufb/gear_sonic_deploy/reference && python3 ~/ufb/sonic_motions/convert_motions.py --convert_lafan_csv /tmp/shake.csv ufb_shake/"
```

The robot's system `python3` (3.10) has numpy 1.21 and scipy 1.8, which the converter needs.
Then `rescan` and `play ufb_shake` as above. Alternatively convert on the laptop with the
converter from the repo (it is pure numpy) and scp the resulting folder.

## Killing it when a motion goes wrong

In order of preference. Keep the robot on the gantry for all of these.

1. `scripts/robot_sonic.py estop` from the laptop: sends `emergency_stop` then `stop`; the binary
   writes its damping-only command and the daemon kills it. No confirmation prompt.
2. If the websocket path is dead, over SSH: `pkill -INT -f g1_deploy_onnx_ref` (clean exit path),
   then `pkill -9 -f g1_deploy_onnx_ref` if it is still there after a couple of seconds.
3. `sudo systemctl restart sonic-daemon`: systemd kills the whole cgroup (daemon and binary) and
   brings the daemon back idle. Use this if the daemon itself is wedged.
4. Physical: the robot's e-stop / remote damping shortcut and the gantry. Once the binary is gone
   nothing is publishing `rt/lowcmd`; do not count on the motors holding position.

## Sim parity note

The laptop sim (`g1_sim/run_sonic.py`) runs **SONIC v1.1**; the robot runs the **default
release**. Both are 29-DoF 50 Hz with the same decoder layout, but they are different weights
and the default release has no heading-normalised orientation term. If you want the sim to
predict the robot exactly, download the default variant (`model_encoder.onnx`,
`model_decoder.onnx`, `observation_config.yaml` at the top level of `nvidia/GEAR-SONIC`) and
point `SonicController` at it; the encoder layout differs slightly (`motion_anchor_orientation`
instead of the `_heading` variants, plus `motion_root_z_position` terms), so
`sonic_params.py` offsets would need a second table.
