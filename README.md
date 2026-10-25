# Slouch Patrol: Gemma on a phone drives a Unitree G1

A Unitree G1 humanoid watches you work. When you slouch, it walks over, taps you, and says "Sit up straight." Keep slouching and the tap turns into a shake, then a slap.

The robot's brain is a Samsung phone mounted on its head. Gemma 4 E2B runs on the phone's GPU through LiteRT-LM, looks at the camera feed, decides when someone is slouching, and chooses which pre-recorded motion the robot plays next. NVIDIA's SONIC whole-body controller turns those motions into joint commands, either in a MuJoCo simulation or on the real robot. Nothing goes to the cloud.

Built at the Google Gemma Hardware Hackathon, AGI House, 2026-10-24.

## How it works

```
Samsung phone (on the robot's head)
  camera ──► frame every ~5 s ──► Gemma 4 E2B (LiteRT-LM, phone GPU)
                                    │  {"slouch": 0-10, "person": left|center|right|none, "distance": far|near|reach}
                                    ▼
              harness: JSON log, 2-in-a-row trigger, state machine, rule check, strike count
                                    │  "3\n" = move_forward, "0\n" = tap, ...
                                    ▼
            TCP to a SONIC launcher (same protocol for both)
              ├─ scripts/sonic_launcher.py  → MuJoCo sim on a laptop      (port 5555)
              └─ scripts/robot_launcher.py  → real G1 through its daemon  (port 5556)
                                    │  "0\n" when the motion is done
                                    ▼
              phone speaks the line with Android's offline text-to-speech
```

The phone runs a small loop around Gemma (the "harness"):

1. **Watch.** Gemma rates each frame for slouching from 0 to 10. Two ratings of 5 or more in a row confirm a slouch.
2. **Walk.** In fixed mode, the robot steps forward a set number of times. In Gemma mode, Gemma looks again before every step and picks `rotate_left`, `rotate_right`, `move_forward`, or `arrived`. A rule check overrides unsafe choices.
3. **Punish.** First strike is a tap, second a shake, third and later a slap, each with a spoken line.
4. **Cooldown.** Wait for a teammate to put the robot back on its mark and tap Reset.

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## What's in this repo

| path | what it is |
|---|---|
| `android/` | The phone app (Kotlin): camera, Gemma through LiteRT-LM, the harness, the robot link. See [docs/ANDROID_APP.md](docs/ANDROID_APP.md) |
| `g1_sim/` | MuJoCo G1 simulation with a pure-Python port of SONIC (onnxruntime on CPU). See [docs/sonic_sim.md](docs/sonic_sim.md) |
| `scripts/sonic_launcher.py` | TCP launcher that plays a motion in the sim and replies when it ends |
| `scripts/robot_launcher.py`, `scripts/robot_sonic.py` | The same protocol bridged to the real G1. See [docs/robot_sonic.md](docs/robot_sonic.md) |
| `scripts/client_sample.py` | Standard-library test client for either launcher |
| `motion_demo/*.csv` | The six demo motions: tap, shake, slap, forward, left, right |
| `docs/` | Architecture, app guide, demo runbook, presentation notes |

## Quick start

1. Start a launcher. For the sim, on the laptop: `.venv\Scripts\python scripts/sonic_launcher.py`. It prints the laptop's addresses. For the real robot, see [docs/robot_sonic.md](docs/robot_sonic.md).
2. Build and install the app: `cd android && ./gradlew assembleDebug && adb install -r app/build/outputs/apk/debug/app-debug.apk`.
3. Put the model on the phone: `adb push gemma-4-E2B-it.litertlm /data/local/tmp/llm/`.
4. Point the app at the launcher: `adb shell am start -n com.slouchpatrol/.MainActivity -e robot <launcher-ip>:5555`.
5. Point the phone at someone working on a laptop, side-on. Slouch.

Step-by-step setup and stage checklist: [docs/DEMO_RUNBOOK.md](docs/DEMO_RUNBOOK.md). Pitch material and measured numbers: [docs/PRESENTATION_NOTES.md](docs/PRESENTATION_NOTES.md).

## Licenses

App and harness code: the team's choice. Gemma 4: Gemma Terms of Use. LiteRT-LM: Apache 2.0. Menagerie G1 model: BSD-3 (Google DeepMind). SONIC weights: NVIDIA Open Model License. SONIC source that the sim port is derived from: Apache 2.0.
