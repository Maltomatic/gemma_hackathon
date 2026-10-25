# Presentation notes

Material for the slides and the stage talk. Every number here was measured on the day; don't round them up.

## One line

A phone on a humanoid's head runs Gemma fully offline. It watches you work, and when you slouch it decides to walk over and correct you, escalating from a tap to a shake to a slap.

## Structure (about 3 minutes)

1. **Cold open, the demo (60 s).** Someone types at a laptop and slouches. The robot walks over and taps them, and the phone says "Sit up straight." Let the laugh land.
2. **What just happened (45 s).** The phone on the robot's head is the whole brain. Gemma 4 E2B runs on the phone's GPU through LiteRT-LM and looks at a camera frame every few seconds. It decides two things: is this person slouching, and what should the robot do next. It picks from a menu of motions (step forward, turn left, turn right, tap, shake, slap). NVIDIA's SONIC controller turns each motion into joint commands. No cloud, no laptop doing inference.
3. **The harness (45 s).** Gemma doesn't drive the robot directly. A small loop on the phone:
   - keeps a JSON log, which is the robot's memory;
   - waits for two slouch ratings in a row before acting;
   - runs a rule check that overrides unsafe choices, so the robot never steps into someone already within reach;
   - counts strikes to escalate.

   Gemma proposes, the harness decides.
4. **Why it matters (30 s).** Swap "slouching" for "fell on the floor" and this is elder care: a robot that notices, walks over, and checks on someone, privately, with no camera feed leaving the house. Swap the motion menu for terrain moves and the same planner picks "step up" or "side-step". The comedy is the demo; the pattern is an on-device planner over a library of safe, pre-validated motions.

## Tracks we fit

| prize | why we fit | caveat |
|---|---|---|
| **Physical AI, IoT & Robotics (main track)** | A real humanoid, steered by an on-device Gemma, doing a closed loop of see, decide, act | Be clear about what ran on the real G1 and what ran in the MuJoCo sim |
| **Best Use of LiteRT** | Gemma runs through LiteRT-LM on the phone GPU, with the vision encoder on the GPU too. We tuned LiteRT to fit an 8 GB phone (see below) | — |
| Best in Show (Android & Pixel) | It is an Android app end to end | We used a Samsung, not a Pixel, so it's a weaker fit |
| Qualcomm, Pixel/TPU | Not eligible | The phone is an Exynos 2500, not Snapdragon or Tensor |

## The engineering story (judges like real constraints)

- **E4B didn't fit.** Our first choice was Gemma 4 E4B. On the Galaxy S26 FE (8 GB RAM, about 4 GB free), Android's memory killer stopped the app every time it loaded E4B on the GPU.
  - It still happened with the camera off and with speculative decoding disabled.
  - E4B's GPU-specific build loads, but it has no vision encoder.
  - E4B on the CPU worked but took **37 s per frame**.
  - With the text model on the CPU, the vision encoder on the GPU, and fewer image tokens, it took **14 s**.
- **E2B on the GPU: 3.5–5.5 s per frame, about 2.2 GB of memory.** That's what we ship. Speculative decoding is off (its extra graph costs memory we don't have), and the image budget is 280 tokens.
- **Seeing slouching was the hard part.**
  - The first version said "not slouching" for an obvious hunch. The frame was upside down because the phone was mounted flipped, so we now rotate based on how the phone is physically held.
  - The camera's full frame made the person tiny, so we crop to what the screen shows.
  - A plain yes/no question pulled the small model toward "no", so we ask for a 0–10 score with concrete anchors: a head at or below shoulder height scores at least 7.
  - After that, a hard hunch scored 7–10 and decent posture scored 3–6.
- **Safety by design.** Motions are pre-recorded clips that SONIC has to keep balanced. Gemma can only choose from the menu, never send joint angles. A rule check refuses "arrived" unless the person is centered and within reach. Fixed mode removes Gemma from the walk entirely for the stage, and a teammate holds the e-stop.
- **Flaky venue Wi-Fi.** Pings to the launcher spiked to 900 ms, so the robot link reuses one connection and retries up to 3 times.

## Measured numbers

| what | value |
|---|---|
| Phone | Samsung Galaxy S26 FE (SM-S741U), Exynos 2500, 8 GB RAM |
| Model | Gemma 4 E2B, `gemma-4-E2B-it.litertlm`, 2.59 GB |
| Runtime | LiteRT-LM 0.18.1, GPU backend for text and vision |
| Time per Gemma call | 3.5–5.5 s (E2B GPU) vs 14 s (E4B CPU + GPU vision) vs 37 s (E4B CPU) |
| App memory | about 2.2 GB (E2B) |
| Slouch to robot moving | about 10–15 s |
| Slouch ratings seen | hard hunch 7–10; upright-ish 3–6 |
| Cloud calls | zero |

## Questions to prepare for

- **Is it really all on-device?** Yes. Turn on airplane mode with Wi-Fi on for the local link to the robot; the status line shows it. Gemma runs on the phone; the launcher only plays motion clips.
- **Why not have Gemma output joint angles?** A 2B model can't balance a 35 kg humanoid. SONIC does balance; Gemma does judgment and sequencing. That split is the point.
- **Why a phone and not the robot's Orin?** The challenge is Gemma on a device you'd actually carry, and a phone is a camera, a speaker, and an AI accelerator in one. The Orin stays busy running SONIC.
- **How accurate is the slouch call?** It's a 2B vision model, so not perfect. The two-in-a-row rule and the 0–10 threshold make it usable for a demo. Next, we'd add a small pose model on LiteRT that measures the ear–shoulder–hip angle and hands Gemma the number.
- **What would you build next?** Fall detection for elder care, with the same loop and a different trigger. Also multi-terrain navigation, where Gemma chooses between step-up, side-step and turn moves from the camera view.

## Team

(Fill in names and roles.)
