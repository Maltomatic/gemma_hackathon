# Architecture

Three pieces talk over the local network: the phone app, a SONIC launcher, and either the MuJoCo sim or the real G1. The phone does all the AI work. The launchers only play motions.

## 1. The phone app (`android/`)

Kotlin, minSdk 26, built against Android SDK 35. Four files do the work:

| file | job |
|---|---|
| `GemmaJudge.kt` | Loads the `.litertlm` model with LiteRT-LM and sends one frame plus a text turn per call. Holds the two prompts. |
| `Harness.kt` | The loop around Gemma: frame history, JSON log, voting, state machine, rule check, strikes, voice lines. |
| `RobotLink.kt` | TCP client for the launcher protocol. Keeps one connection open and retries up to 3 times on slow Wi-Fi. |
| `MainActivity.kt` | Camera (CameraX), on-screen status and log, settings, test buttons, `adb` switches. |

### Model and runtime

- **Model:** `gemma-4-E2B-it.litertlm` from `litert-community/gemma-4-E2B-it-litert-lm` on Hugging Face (2.59 GB).
- **Runtime:** `com.google.ai.edge.litertlm:litertlm-android:0.18.1`. The text model and the vision encoder both run on the GPU (`Backend.GPU()`), through LiteRT's OpenCL path. The phone's log shows all 2,712 nodes of the decode graph handed to the `LITERT_CL` delegate.
- **Settings that matter:**
  - Speculative decoding is off, because its extra `verify` graph doesn't fit in the phone's memory.
  - The image budget is 280 tokens.
  - Replies are capped at 48 tokens.
  - The context is capped at 2,048 tokens.
  - Each frame gets a fresh conversation, so no history builds up inside the model; the harness passes recent history as text instead.
- **Frames:** taken from CameraX and turned upright based on how the phone is physically held. They're cropped to what the screen shows, scaled to at most 768 px on the long side, and JPEG-encoded.

### The two prompts

Both prompts are about one **target**: "the person wearing a bright pink wig" by default (setting `target`, adb switch `-e target "..."`). Everyone else in the frame is ignored, which matters on a stage with people in the background. If the target isn't visible, Gemma reports `"person": "none"`, and a "none" never counts as a slouch vote.

**Watch prompt.** Gemma rates slouching 0 to 10 using the hip–shoulder–ear line, with anchors at 0, 5, and 10, and two hard rules:
- If the head is at or below shoulder height, or the torso leans forward more than about 30°, the score is at least 7.
- Only a clearly straight back scores 0 to 2.

Gemma also reports where the person is in the frame (left, center, right, or none) and how far away they are (far, near, or reach). The reply is one line of JSON.

**Approach prompt.** Gemma picks the next motion from `rotate_left`, `rotate_right`, `move_forward`, and `arrived`, following written rules: turn toward the person's side of the frame, step forward when they're centered but not within reach, and stop when they're centered and within reach.

Each user turn also carries the last 5 entries of the harness's JSON log. That log is the robot's memory of what it saw and did.

### States

```
LOADING ──► WATCHING ──(2 slouch ratings ≥ 5 in a row)──► APPROACHING ──► PUNISHING ──► COOLDOWN
                ▲                                                                           │
                └──────────────────────────── Reset ────────────────────────────────────────┘
```

- **Strikes** count confirmed slouches since the app started. Strike 1 is `tap` ("Sit up straight."), strike 2 is `shake` ("I said, sit up straight."), and strike 3 and later is `slap` ("You need to stop slouching."). **Forgive** in Settings clears them.
- **Fixed mode** (`fixedSteps > 0`): after a slouch is confirmed, the robot plays `move_forward` exactly N times with no Gemma calls, then punishes. This is the predictable option for the stage.
- **Gemma mode** (`fixedSteps = 0`): Gemma picks each step from a new frame. It gives up after 10 steps.

### Rule check (Gemma mode)

Gemma suggests a step; this code makes the final decision. It errs toward stopping, because walking into someone is the failure that matters. Rules are checked top to bottom:

| situation | what the robot does |
|---|---|
| Gemma says `arrived`, or rates the distance as "reach" | stop (arrived) |
| it has already taken `maxForward` forward steps (default 3) | stop (arrived) |
| no person in view, after walking toward one who wasn't "far" | stop: they're too close for the head camera to see |
| no person in view otherwise | turn toward the side where they were last seen |
| Gemma says `rotate_left` / `rotate_right` | allowed |
| person centered | step forward |
| person off-center | turn toward them |

Every decision is also written to logcat under the tag `SlouchPatrol` (`adb logcat -s SlouchPatrol`), so a run can be reviewed afterwards.

An earlier version overrode Gemma's "arrived" whenever it rated the person only "near". On the real robot, that sent it stepping forward while it was already at the person, because up close the head camera sees only part of them. The rules above fix that.

## 2. The launcher protocol

Both launchers speak the same newline-terminated ASCII over TCP, on one reused connection:

| phone sends | launcher replies |
|---|---|
| `0` tap, `1` shake, `2` slap, `3` forward, `4` left, `5` right | `0` when the motion has finished |
| `?` | one `idx name ok|missing` line each, then `.` |

Error replies: `1` unknown motion, `2` busy, `3` playback failed or timed out, `5` SONIC not running (real robot only).

The app maps its motion names onto these numbers: `tap`→0, `shake`→1, `slap`→2, `move_forward`→3, `rotate_left`→4, `rotate_right`→5.

## 3. Motion execution

- **Sim** (`scripts/sonic_launcher.py`, port 5555): each request runs one fresh MuJoCo process. It holds for 2 s, plays the clip from `motion_demo/<name>.csv` through the Python SONIC port, holds for another 0.5 s, then replies. See [sonic_sim.md](sonic_sim.md).
- **Real robot** (`scripts/robot_launcher.py`, port 5556 when run on the robot): it plays `reference/ufb_<name>` through the robot's `sonic-daemon`, waits until the robot is back in planner idle, then replies. SONIC keeps running between motions. See [robot_sonic.md](robot_sonic.md).

## Measured timings (Samsung Galaxy S26 FE, Exynos 2500, 8 GB RAM)

| step | time |
|---|---|
| model load (E2B, GPU, cached) | under 10 s |
| one Gemma call (one image) | 3.5 to 5.5 s |
| slouch confirmed (2 ratings in a row) | about 10 to 15 s |
| one walking step in Gemma mode (Gemma + sim clip + 0.7 s settle) | about 17 s |
| app memory with E2B on GPU | about 2.2 GB |
