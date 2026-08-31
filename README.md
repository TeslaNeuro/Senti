# What Am I Looking At?

A local-first macOS visual assistant that uses your MacBook camera to understand what you are looking at.

**Phase 13 (current):** polish — faster startup, clearer status, busy-state UX, and shortcuts.

## Architecture (planned)

| Layer | Technology |
| --- | --- |
| UI | PySide6 (native macOS look, AVFoundation via Qt Multimedia) |
| Fast loop (perception) | Ultralytics YOLO26 on Apple Silicon (`device=mps`) |
| Deep loop (understanding) | Local VLM (Ollama / configurable backend) |
| OCR | EasyOCR (local, optional) |
| Speech | macOS `NSSpeechSynthesizer` via Qt TextToSpeech |
| Voice input | faster-whisper (local, push-to-talk) |

Two processing loops keep the app responsive:

- **Fast loop:** camera → YOLO26 → tracking → scene-change detection (15–30 FPS target)
- **Slow loop:** frame selection → OCR → local VLM (only on meaningful scene changes or user questions)

## Requirements

- macOS on Apple Silicon (developed for M2 Pro)
- Python 3.11+
- MacBook built-in camera
- Camera permission

## Setup

```bash
cd /path/to/Senti
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional — defaults work for Phase 1
```

## Run (Phase 1)

```bash
chmod +x scripts/run.sh
./scripts/run.sh
```

Or directly:

```bash
source .venv/bin/activate
python -m app
```

The app configures Qt plugin paths automatically via `app/qt_bootstrap.py`.

If the window opens but the camera is black or shows an error, **another app may be using the camera** (FaceTime, Zoom, Teams, Chrome, etc.). Quit those apps and relaunch.

On first launch, macOS will ask for camera permission. Grant access to **Terminal** (or your IDE) if running from the command line.

## What you should see

- Window titled **What Am I Looking At?**
- Live camera preview with **YOLO26 bounding boxes**, labels, confidence, and **track IDs** (`#1`, `#2`, …)
- Each object keeps the same ID while it stays in view
- Distinct box colors per track ID
- State transitions: `WATCHING` → `SCENE_CHANGED` → `WAITING_FOR_STABILITY` → `READY`
- Scene change uses visual diff + object new/lost + spatial movement (not YOLO alone)
- When scene reaches `READY`, the **best frame** is picked from a rolling buffer (sharpness, stability, object size/position, confidence, lighting)
- The app **automatically** sends that frame to the local VLM when the scene stabilizes
- **Ask** follow-up questions in the text box (e.g. "What's that connector?", "What objects do you see?")
- Use the **Focus** dropdown to analyze a specific tracked object (`#1 cup`, `#2 phone`, …)
- Object-focused questions auto-crop the most likely target before sending to the VLM
- **OCR** reads visible text when enabled — ask `read this` or let it run automatically on scene `READY`
- Detected text is passed to the VLM as extra context
- **TTS** speaks assistant responses aloud when enabled (auto or via **Speak** / `say that`)
- **Voice input** — click **Mic**, ask your question aloud, click **Stop** to send it
- Simple questions may be answered instantly from scene memory without a new VLM call
- Click **Analyze** anytime for a manual re-check (bypasses cooldown)
- Assistant response appears in the main panel; VLM status shows in the status bar
- Status bar: Camera ●, YOLO26 ● (device), VLM (inactive)
- Camera FPS, YOLO FPS, and inference latency (ms)
- State: `WATCHING` when the camera is running

On first run, Ultralytics downloads `yolo26n.pt` automatically (~6 MB).

### VLM setup (Phase 6)

Install and run [Ollama](https://ollama.com), then pull a vision model:

```bash
ollama pull gemma4
# or: ollama pull llava
```

Copy config and adjust if needed:

```bash
cp .env.example .env
```

```env
VLM_MODEL=gemma4
VLM_BASE_URL=http://localhost:11434
```

Then launch the app. Analysis runs automatically when the scene reaches `READY`, or click **Analyze** for a manual check.

### OCR setup (Phase 10)

Enable in `.env`:

```env
OCR_ENABLED=true
OCR_RUNTIME=easyocr
OCR_LANGUAGES=en
```

Install dependencies (first run downloads EasyOCR models):

```bash
pip install easyocr
```

Then ask **read this** in the chat box, or let OCR run automatically when the scene stabilizes.

### TTS setup (Phase 11)

Enable in `.env`:

```env
TTS_ENABLED=true
TTS_AUTO_SPEAK=true
```

Uses the built-in macOS speech engine (no extra install). Click **Speak** to replay the current response, or ask **say that** / **read aloud**.

Qt only exposes a small set of voices. If your voice (e.g. **Tessa**) is not in that list, leave `TTS_RUNTIME=auto` (default) and the app will fall back to the macOS `say` command automatically.

### Voice input setup (Phase 12)

Enable in `.env`:

```env
VOICE_ENABLED=true
VOICE_RUNTIME=whisper
VOICE_MODEL=base
```

Install dependencies (first run downloads the Whisper model):

```bash
pip install sounddevice faster-whisper
```

Click **Mic** to start recording, speak your question, then click **Stop**. The transcript is sent to the same Ask pipeline as typed questions.

## Polish (Phase 13)

- **Clear** resets scene memory and the response panel
- **Esc** stops an in-progress voice recording
- Status indicators show full details on hover
- Controls disable while analyzing, transcribing, or running OCR
- Whisper model loads on first voice use (faster app startup)
- Duplicate `.env` keys are logged at startup (last value wins)

## Troubleshooting startup messages

| Message | Meaning | Action |
| --- | --- | --- |
| `Voice 'Tessa' is not exposed by Qt; using macOS say backend` | Normal — Qt only lists a few voices | No action needed with `TTS_RUNTIME=auto` |
| `Using CPU` from EasyOCR | EasyOCR has no Apple Silicon GPU path | Expected on Mac; OCR still works |
| `AVFFrameReceiver` / `AVFAudioReceiver` duplicate class | OpenCV and PyAV both ship FFmpeg libs | Usually harmless; appears when voice/OCR load |
| `torch.quantize_per_tensor` deprecation | PyTorch warning from faster-whisper | Hidden unless `DEBUG_MODE=true` |
| `NSCameraUseContinuityCameraDeviceType` | Continuity Camera hint | Added to `resources/Info.plist` for `run.sh` |

First OCR or voice use may take longer while models load in the background.

## Configuration

See `.env.example`. Key settings:

| Variable | Default | Description |
| --- | --- | --- |
| `CAMERA_DEVICE` | `0` | Camera index |
| `CAMERA_WIDTH` | `1280` | Preferred capture width |
| `CAMERA_HEIGHT` | `720` | Preferred capture height |
| `TARGET_FPS` | `30` | Preferred frame rate |
| `FRAME_BUFFER_SIZE` | `8` | Max in-memory frames (oldest dropped) |
| `YOLO_MODEL` | `yolo26n.pt` | Ultralytics YOLO26 weights |
| `YOLO_CONFIDENCE` | `0.5` | Detection confidence threshold |
| `YOLO_IMAGE_SIZE` | `640` | YOLO inference size |
| `YOLO_DEVICE` | `auto` | `auto`, `mps`, or `cpu` |
| `TRACKING_ENABLED` | `true` | Enable ByteTrack/BoT-SORT tracking |
| `TRACKER_TYPE` | `bytetrack.yaml` | `bytetrack.yaml` or `botsort.yaml` |
| `SCENE_CHANGE_THRESHOLD` | `0.15` | Sensitivity for scene-change detection |
| `STABILITY_FRAMES` | `10` | Stable frames required before `READY` |
| `SELECTION_BUFFER_SIZE` | `12` | Rolling buffer size for best-frame selection |
| `VLM_MODEL` | `gemma4` | Ollama vision model name |
| `VLM_RUNTIME` | `ollama` | Local VLM backend |
| `VLM_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `VLM_COOLDOWN` | `5.0` | Minimum seconds between automatic analyses |
| `VLM_AUTO_ANALYZE` | `true` | Auto-run VLM when scene reaches `READY` |
| `CONVERSATION_MAX_TURNS` | `8` | Recent Q&A turns kept for follow-ups |
| `CONTEXT_STALE_FRAMES` | `45` | Re-analyze with a fresh frame after this many frames |
| `OBJECT_CROP_ENABLED` | `true` | Crop detected objects before focused VLM analysis |
| `OBJECT_CROP_PADDING` | `0.15` | Padding around object crops (ratio of bbox size) |
| `OCR_ENABLED` | `false` | Enable local text recognition |
| `OCR_RUNTIME` | `easyocr` | OCR backend |
| `OCR_LANGUAGES` | `en` | Comma-separated EasyOCR languages |
| `OCR_MIN_CONFIDENCE` | `0.4` | Minimum OCR confidence to keep a line |
| `OCR_AUTO_ON_READY` | `true` | Run OCR when scene reaches `READY` |
| `TTS_ENABLED` | `false` | Speak assistant responses aloud |
| `TTS_RUNTIME` | `auto` | `auto`, `qt`, or `say` (macOS `say` supports all system voices) |
| `TTS_AUTO_SPEAK` | `true` | Auto-speak new VLM/OCR answers |
| `TTS_RATE` | `0.0` | Speech rate from `-1.0` (slow) to `1.0` (fast) |
| `TTS_VOLUME` | `1.0` | Speech volume `0.0`–`1.0` |
| `TTS_VOICE` | *(empty)* | Optional macOS voice name substring |
| `TTS_INTERRUPT` | `true` | Stop current speech when new text arrives |
| `VOICE_ENABLED` | `false` | Enable push-to-talk microphone questions |
| `VOICE_RUNTIME` | `whisper` | Local speech-to-text backend |
| `VOICE_MODEL` | `base` | Whisper model (`tiny`, `base`, `small`, …) |
| `VOICE_LANGUAGE` | `en` | Transcription language (empty = auto-detect) |
| `VOICE_MAX_SECONDS` | `8.0` | Auto-stop recording after this many seconds |
| `VOICE_MIN_SECONDS` | `0.6` | Ignore recordings shorter than this |

## Test

```bash
pytest tests/ -q
```

## Privacy

- No cloud uploads
- No recording or disk storage of camera frames
- Frames remain in a bounded in-memory buffer only

## Roadmap

1. ✅ Phase 1 — Camera preview + FPS
2. ✅ Phase 2 — YOLO26 detection + bounding boxes
3. ✅ Phase 3 — Object tracking
4. ✅ Phase 4 — Scene change detection
5. ✅ Phase 5 — Best-frame selection
6. ✅ Phase 6 — Local VLM (manual Analyze)
7. ✅ Phase 7 — Automatic VLM scheduling
8. ✅ Phase 8 — Conversational scene context
9. ✅ Phase 9 — Object-focused crops
10. ✅ Phase 10 — Local OCR
11. ✅ Phase 11 — macOS TTS
12. ✅ Phase 12 — Voice input
13. ✅ Phase 13 — Polish

## YOLO26 on Apple Silicon

Ultralytics YOLO26 supports inference via PyTorch MPS:

```python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
results = model.predict(source=frame, device="mps")
```

An optional MLX-native path (`yolo-mlx`) can be evaluated later for higher throughput. Phase 2 will use official Ultralytics YOLO26 as required.
