# Demo runbook

## The bit

1. The **target wears a bright pink wig** and sits side-on to the robot, typing on a laptop, about 1 to 2 robot steps away. Gemma ignores everyone else, so people in the background are fine.
2. They sit up straight at first. The phone screen shows ratings around 0 to 4.
3. They slouch hard. After two ratings of 5 or more in a row (about 10 to 15 s), the robot walks over and **taps** them. The phone says "Sit up straight."
4. The hit is fixed at launch: `-e hit tap`, `-e hit shake` or `-e hit slap`. To show all three, relaunch with a different `hit` between runs, or use `-e hit escalate`.

## Before you go on

- [ ] **Launcher running:** the sim is `scripts/sonic_launcher.py` on port 5555; the real robot is `robot_launcher.py` on port 5556 (see [robot_sonic.md](robot_sonic.md)).
- [ ] **Phone sees the launcher:** in the app, tap Settings › **Ping**. It must list `0 tap ok … 5 right ok`.
- [ ] **Motions work:** in Settings, tap **↑ Fwd** once, then **Tap**. Both should play.
- [ ] **Use fixed mode on stage.** Gemma mode works, but its distance judgment up close is the weakest part. If you use Gemma mode, keep `-e maxfwd` at or below the number of steps to the person.
- [ ] **Fixed steps match the distance:** set `-e steps N`, where N is the floor distance divided by one forward step.
- [ ] **Voice works:** Settings › **Voice** should speak. Make sure the offline English voice is downloaded.
- [ ] **Strikes are cleared:** Settings › **Forgive**, then **Reset**.
- [ ] **Screen stays on:** set the screen timeout to the maximum, and keep the phone plugged in.
- [ ] **Airplane mode with Wi-Fi on:** this shows the judges nothing goes to the cloud. The status line shows "Airplane mode ON".
- [ ] **Target line is right:** the status line says "Target: person wearing a bright pink wig". Test that someone *without* the wig slouching does nothing.
- [ ] **Frame is upright:** the thumbnail in the top right must show the person upright and side-on.
- [ ] **Someone on the e-stop:** a teammate holds the G1's remote, or is ready to run `robot_sonic.py estop`, for every run.
- [ ] **Backup video:** record one clean run before judging.

## Network

The phone and the launcher must be on the same network. On 2026-10-24:
- the phone was at 192.168.2.74;
- Justin's laptop (sim launcher) was at 192.168.7.23;
- the robot's Wi-Fi was at 192.168.3.18.

All three are on the LatentSpace Wi-Fi (192.168.0.0/21).

The venue Wi-Fi lagged at times, with pings of 160–900 ms. The app retries, but if it gets bad, turn on the Samsung's hotspot and connect the launcher machine to it. Then set the new address with `-e robot <ip>:<port>`.

## If something breaks on stage

| problem | quick fix |
|---|---|
| Robot doesn't move | Settings › Ping. If it fails, it's the network: switch to the hotspot. |
| Never fires | Lower the threshold (`-e threshold 4`), or have the person slouch harder and side-on. |
| Fires on good posture | Raise the threshold (`-e threshold 6`). |
| Stuck in COOLDOWN | Expected after each hit. Put the robot back on its mark and tap Reset. |
| App closed | Relaunch it. The model reloads in under 10 s. |
| Anything worse | Play the backup video. |
