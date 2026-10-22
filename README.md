<p align="center"><img src="docs/build/logo.png" width="84" alt="Symbolic Synapse"></p>

# DogSense HUD: real-time canine behaviour telemetry for Rokid glasses

**A Symbolic Synapse prototype.** It reads a dog's posture and vocalisations through the Rokid RV101 glasses, then projects what it measured onto the monochrome-green HUD: arousal and valence gauges, the behavioural state and the cues behind it. It reports what it saw and heard, and it never claims to know what the dog is thinking.

| | |
|---|---|
| **90 s overview video** | [media/DogSense_HUD_90s.mp4](media/DogSense_HUD_90s.mp4) |
| **Field test through the glasses** (HUD with original sound) | [media/DogSense_HUD_test.mp4](media/DogSense_HUD_test.mp4) |
| **Technical documentation** (22 pages) | [docs/DogSense_HUD_Technical_Documentation.pdf](docs/DogSense_HUD_Technical_Documentation.pdf) |

**Highlights**
- **Pose:** YOLO26n-pose fine-tuned on Dog-Pose (24 keypoints, pose mAP50 0.936), plus a second detection stage for dogs facing you or behind gate bars. On the gate clip, tracking rose from 35 % to 79 % of frames.
- **Ego-motion compensation:** every skeleton is measured in the dog's own body frame, so the wearer's head can move.
- **Bioacoustics:** pitch, harmonicity and inter-bark interval, plus an **AST sound-source gate** that ignores the wearer's own voice, shouts and whistles.
- **Neuro-symbolic fusion:** ordered, explainable rules (a play bow overrides a growl; stiff + forward + low growl gives a warning). Every state traces back to its cues.
- **Tested on real footage** from the RV101 (three clips). 10 field issues were found and fixed, and 9 unit tests pass.

> **Status:** research prototype. The behaviour and vocal heads are pre-trained on synthetic data, and real-world accuracy has not yet been measured against labelled clips. It is not a substitute for a qualified trainer or behaviourist.

```
 Rokid Glasses (Android app)            Local GPU server (this PC, RTX 3090)
 ┌──────────────────────────┐  Wi-Fi   ┌────────────────────────────────────────────────┐
 │ CameraX 720p -> JPEG 15fps│ ───────▶ │ YOLO26n-pose (24 dog keypoints)                 │
 │ AudioRecord 16k PCM 100ms │  WS bin  │  └▶ body-frame normalisation (ego-motion comp.) │
 │ Gravity sensor -> roll    │          │  └▶ kinematics: wag Hz/asymmetry, bow, CoM,     │
 │ SpeechGate (reverse VAD)  │          │      ears, flicks, stiffness  + 1D-CNN (30 fr)  │
 │                           │          │ Audio: HPF -> events -> AST voice gate -> F0/HNR│
 │ HudView (Canvas, green)   │ ◀─────── │ Morphology calibration (breed baselines)        │
 │  reticle, gauges, caption │  JSON    │ Neuro-symbolic fusion rules -> state/alert      │
 └──────────────────────────┘          │ Optional async VLM (Ollama) -> trigger tag      │
                                        └────────────────────────────────────────────────┘
```

## Layout

| Path | What |
|---|---|
| `dogsense/keypoints.py` | Dog-Pose 24-keypoint schema, body-frame and gravity-frame normalisation |
| `dogsense/kinematics.py` | Sliding-window kinematic features (tail wag frequency/amplitude/L-R bias, play bow, weight shift, ear carriage, 200 ms ear flicks, stiffness) |
| `dogsense/acoustics.py` | Wind high-pass, streaming vocal-event detection, F0 (autocorrelation), HNR, inter-bark interval, rule-based call type |
| `dogsense/models.py` | `TemporalPoseNet` (dilated 1D-CNN) and `VocalCNN` (log-mel) |
| `dogsense/morphology.py` | Per-dog baseline calibration (prick/semi/drop ears, full/docked/curled tail) |
| `dogsense/fusion.py` | Ordered symbolic rules plus neural priors. Outputs state, alert, arousal, valence and the cues that fired |
| `dogsense/vlm.py` | Async scene-trigger tagging through a local Ollama VLM (optional) |
| `dogsense/pipeline.py` | Per-connection session: tracking, calibration, smoothing, telemetry JSON |
| `server/app.py` | FastAPI WebSocket server (`/ws`, `/health`) |
| `training/` | `train_pose.py`, `train_temporal.py`, `train_audio.py`, `synth.py` |
| `tools/simulate.py` | Desktop HUD simulator for a video file or webcam (+ WAV), no glasses needed |
| `tools/ws_client_test.py` | Glasses emulator that uses the real wire protocol (`--rotate 90` tests sideways frames) |
| `tools/export_session.py` | Turns a recorded live session into a HUD video with sound |
| `server/recorder.py` | Per-session recording of frames, audio and telemetry |
| `start_server.ps1` | Live start: firewall check, LAN URL, recording on |
| `android/` | Kotlin glasses app (CameraX + AudioRecord + OkHttp + Canvas HUD) |
| `models/` | Trained weights: `dog_pose.pt` / `.onnx`, `temporal_pose.pt`, `vocal_cnn.pt` |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests
python -m server.app --host 0.0.0.0 --port 8765            # add --vlm qwen2.5vl:7b if Ollama is running
python tools/simulate.py --video my_dog.mp4 --wav my_dog.wav --skeleton
```

Glasses app (set your PC's LAN IP):

```bash
cd android && ./gradlew assembleDebug -PserverUrl=ws://192.168.1.50:8765/ws
adb install app/build/outputs/apk/debug/app-debug.apk
```

Behind a TLS-inspecting proxy or antivirus, Gradle needs `GRADLE_OPTS=-Djavax.net.ssl.trustStoreType=Windows-ROOT`.
Press the temple touchpad (DPAD_CENTER) to toggle the debug skeleton.

## Live use with the Rokid RV101

1. **Firewall, once.** Run this in an admin PowerShell:
   `netsh advfirewall firewall add rule name="DogSense" dir=in action=allow protocol=TCP localport=8765`
2. **Start the server.** Run `.\start_server.ps1`. It records every session to `recordings\` (add `-NoRecord` to turn that off) and prints the exact `-PserverUrl=ws://<PC-IP>:8765/ws` to use. Use the line marked *use this one*; the other addresses belong to virtual adapters.
3. **Build and install the app:**
   `cd android; .\gradlew assembleDebug -PserverUrl=ws://<PC-IP>:8765/ws; adb install -r app\build\outputs\apk\debug\app-debug.apk`
4. **Calibrate the HUD, once per wearer.** The camera sees a wider area than the display, so the box needs mapping onto what you actually see.
   - Long-press the temple touchpad to enter calibration. The HUD shows the camera-frame outline, a crosshair at the camera centre and the live dog box.
   - Look at the dog. Swipe (LEFT/RIGHT/UP/DOWN) to move the brackets onto the dog, tap to switch from *offset* to *scale*, and long-press to save. BACK cancels.
   - You can also set values over adb:
     `adb shell am start -n ai.dogsense.hud/.MainActivity --ef sx 2.0 --ef sy 2.0 --ef ox 0 --ef oy -0.05`
   - Check how the RV101 touchpad gestures map to key events (`adb shell getevent -l`) and adjust `MainActivity.onKeyDown` if they differ.
5. **Review a walk.** Run `python tools/export_session.py recordings\<session>` to get `session_hud.mp4` with sound.

The status line shows the **true round trip** measured on the glasses (`ms rtt`). Frames are sent sensor-native with their rotation (`0x03`), and the server turns them upright.

## Wire protocol (little-endian)

* `0x03 | f64 unix_ts | f32 head_roll_rad | u16 rotation_deg | JPEG`: video frame the server rotates clockwise to upright (sent by the glasses app).
* `0x01 | f64 unix_ts | f32 head_roll_rad | JPEG`: video frame that is already upright (browser page, older clients). The server answers each frame with telemetry. While the GPU is busy, the glasses send nothing new and the server drops stale frames.
* `0x02 | f32 speech_prob | PCM16 mono 16 kHz`: audio chunk. Chunks where `speech_prob > 0.6` are discarded as the wearer's voice.
* Reply is JSON: `state, detail, alert, trigger, cues[], arousal 0..1, valence -1..1, confidence, box[4], kpts[24][3], morph, kin{}, audio{}, pose_probs{}, latency_ms`.

## Training (GPU)

| Model | Command | Data |
|---|---|---|
| Pose | `python training/train_pose.py --epochs 60` (`--resume` to continue) | Ultralytics Dog-Pose, 6,773 train / 1,703 val images, auto-downloaded. Uses an explicit `flip_idx` so left/right flips stay correct |
| Temporal | `python training/train_temporal.py --synthetic 30000` | Procedural trajectories (see below) |
| ... fine-tune | `python training/train_temporal.py --videos data/clips --init models/temporal_pose.pt` | `data/clips/<relaxed\|play_bow\|alert\|defensive_stiff\|anxious\|excited>/*.mp4` |
| Vocal | `python training/train_audio.py --synthetic 12000` | Synthetic calls with wind, hiss and speech-like background |
| ... fine-tune | `python training/train_audio.py --data data/audio --init models/vocal_cnn.pt` | `data/audio/<none\|growl\|bark_alert\|bark_play\|whine\|howl>/*.wav` (DMAD, CREMD, your own recordings) |

### Read this before trusting the temporal and vocal heads

No public dataset pairs behaviour labels with Dog-Pose keypoint sequences. DMAD and CREMD need manual download and licence acceptance. So both heads ship pre-trained on synthetic data only. The synthetic sequences pass through the same featuriser as live data, with random viewpoints, head motion, scale, occlusion and breed morphology. The high synthetic validation accuracy only shows that the networks learn the designed signatures. It does not measure real-world accuracy. Fine-tune both heads on real labelled clips before relying on them.

The decision itself comes from the rule base in `fusion.py`, which weights the networks at only about 25% (pose) and 15% (audio). The rules use measured kinematics and acoustics directly, so the system behaves sensibly before fine-tuning.

## Design notes

* **Ego-motion:** every skeleton is re-expressed relative to the spine (tail_start→withers), in units of spine length. Head roll from the gravity sensor is removed first. A gravity-aligned copy is kept so spine pitch (play bow) stays measurable. When the spine is hidden (head-on view), the frame falls back to the skeleton centroid and its extent.
* **Tail asymmetry** is reported only when it can be judged, in rear or front views. The sign is mirrored for front views so that "right" always means the dog's right. Side views show no lateral sweep.
* **Breed morphology:** the first ~3 s of neutral frames set per-dog baselines. Ear and bow rules score deviation from those baselines, not absolute values. The baselines reset when the dog is lost for about 1 s.
* **Wind and voice:** a 150 Hz high-pass filter and a noise floor taken from the last 2 s handle wind. Every vocal event then passes a **sound-source gate**: an Audio Spectrogram Transformer (AudioSet) classifies a 1.5 s window around it and rejects speech, shouting and whistling, so the wearer telling the dogs off is not read as barking. The gate adds about 0.75 s of delay to sound readings. The on-device SpeechGate and a long-speech rule are kept as extra filters.
* **HUD:** black is transparent on the waveguide. Urgency is shown with geometry, not colour: thicker blinking brackets and a boxed ALERT caption.
* **Rokid specifics:** the app uses standard Android camera, mic and sensor APIs, which work on Android-based Rokid glasses. If your model streams through the phone-side Rokid SDK instead, feed its frames and PCM into `DogSenseClient.sendFrame/sendAudio`; nothing else changes.


## Licence

[GNU AGPL-3.0](LICENSE), compatible with Ultralytics YOLO (AGPL-3.0). The Dog-Pose dataset and the AST model (`MIT/ast-finetuned-audioset-10-10-0.4593`) are used under their own licences. Field-test footage © Symbolic Synapse.
