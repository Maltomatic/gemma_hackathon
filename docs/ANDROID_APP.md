# The phone app

## Build and install

You need JDK 17 and the Android SDK (platform 35, build-tools 35). On a Mac with Homebrew:

```bash
brew install openjdk@17
export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
cd android
echo "sdk.dir=$HOME/Library/Android/sdk" > local.properties
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

On the phone, enable USB debugging first:
1. Go to Settings › About phone › Software information and tap Build number 7 times.
2. Go to Settings › Developer options and turn on USB debugging.
3. Plug the phone into the laptop and tap Allow when it asks.

## Put the model on the phone

Use `gemma-4-E2B-it.litertlm` (2.59 GB, SHA-256 `181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c`). Push it to the one folder the app can read:

```bash
adb shell mkdir -p /data/local/tmp/llm
adb push gemma-4-E2B-it.litertlm /data/local/tmp/llm/
adb shell chmod 644 /data/local/tmp/llm/gemma-4-E2B-it.litertlm
```

The app loads the largest `.litertlm` in `/data/local/tmp/llm/`, skipping the `-gpu` and `-web` builds, which have no vision encoder. Keep only one model in that folder.

Don't push the model into `/sdcard/Android/data/com.slouchpatrol/files/`. Files that `adb` puts there are owned by the shell user, and the app can't open them ("Permission denied").

## Switches you can set from the laptop

Each switch is saved on the phone, so you only set it once:

```bash
A="adb shell am start -n com.slouchpatrol/.MainActivity"
$A -e robot 192.168.7.23:5555      # launcher address (sim on Justin's laptop)
$A -e robot 192.168.3.18:5556      # launcher address (real robot, LatentSpace Wi-Fi)
$A -e steps 2                      # fixed mode: 2 forward steps, then the hit
$A -e steps 0                      # Gemma mode: Gemma plans every step
$A -e threshold 5                  # slouch score (0-10) that counts as slouching
$A -e votes 2/2                    # fire when 2 of the last 2 ratings are slouching
$A -e backend GPU                  # GPU | CPU | MIXED (text on CPU, vision on GPU)
```

If the app is already open, force-stop it first so the new values load:

```bash
adb shell am force-stop com.slouchpatrol
```

## The screen

- **Top left:**
  - the state, the walk mode, the strike count, the model, the backend, and the last call's time in ms;
  - whether airplane mode is on;
  - Gemma's raw JSON reply;
  - a status line.
- **Top right:** the exact frame Gemma just saw. If it's sideways or upside down, Gemma can't judge posture.
- **Bottom:** the harness log (ratings, steps, events, robot replies).
- **Buttons:**
  - Reset: back to watching.
  - Pause.
  - Settings: the launcher address, images per call, votes, voice delay, rotation and fixed steps, plus test buttons for all six motions, Voice, Ping (lists the launcher's motions) and Forgive (clears strikes).

## Troubleshooting

| symptom | cause and fix |
|---|---|
| "Model load failed … Permission denied" | Model is in the app's `Android/data` folder. Move it to `/data/local/tmp/llm/` (see above). |
| App closes while "Loading…" | Out of memory. Gemma 4 E4B doesn't fit on an 8 GB phone on the GPU. Use E2B. Run `adb shell am kill-all` to free memory before launching. |
| "tf_lite_vision_encoder not found" | You loaded a `-gpu` build. It has no vision. Use the plain `.litertlm`. |
| Always "not slouching" | Check the thumbnail. The camera should see the person side-on, at chest height, and upright. |
| `error … No route to host` / `SocketTimeout` | The phone can't reach the launcher. Check both are on the same Wi-Fi and the launcher is running. From the laptop: `adb shell 'printf "?\n" \| nc <ip> <port>'` should list six motions. |
| Robot replies `5` | Real robot only: SONIC isn't running. See [robot_sonic.md](robot_sonic.md). |
