# ⚙️ Configuration

Copyright (c) 2026 Arshia Keshvari. Licensed under the MIT License.

Senti reads a `.env` file in the project root. If `.env` is missing, it falls back to `.env.example`. Duplicate keys are allowed; **the last value wins**, and a warning is logged at startup.

```bash
cp .env.example .env
```

Booleans accept `1`, `true`, `yes`, `on` (any case). Invalid numbers or enums fail fast in `AppConfig.validate()` so the window never starts with a silently broken setup.

## 📷 Camera

| Variable | Default | Description |
| --- | --- | --- |
| `CAMERA_DEVICE` | `0` | Index in Qt’s camera list. `0` is usually the built-in FaceTime camera. |
| `CAMERA_WIDTH` | `1280` | Preferred capture width in pixels. |
| `CAMERA_HEIGHT` | `720` | Preferred capture height in pixels. |
| `TARGET_FPS` | `30` | Preferred frame rate. Actual FPS depends on lighting and camera mode. |
| `FRAME_BUFFER_SIZE` | `8` | In-memory live frames. Oldest is dropped when full. |
| `SELECTION_BUFFER_SIZE` | `12` | Rolling buffer used to pick the best analysis frame. |

Raise resolution only if you have the GPU headroom. Detection still resizes to `YOLO_IMAGE_SIZE`.

## 🎯 Detection and tracking

| Variable | Default | Description |
| --- | --- | --- |
| `YOLO_MODEL` | `yolo26n.pt` | Filename of Ultralytics weights. Always stored and loaded from `models/` (for example `models/yolo26n.pt`). `yolo26s.pt` is slower and more accurate. An existing absolute path is used as-is. For MLX, Senti converts this file to `models/<name>.npz` on first launch. |
| `YOLO_CONFIDENCE` | `0.5` | Minimum detection confidence in `(0, 1]`. |
| `YOLO_IMAGE_SIZE` | `640` | Inference size passed to Ultralytics / yolo-mlx. |
| `YOLO_RUNTIME` | `auto` | `auto`, `ultralytics`, or `mlx`. `mlx` uses [yolo-mlx](https://github.com/thewebAI/yolo-mlx) (Metal). `ultralytics` uses PyTorch. `auto` follows `YOLO_DEVICE` (`mlx` → MLX, otherwise Ultralytics). |
| `YOLO_DEVICE` | `auto` | `auto`, `mps`, `mlx`, `cpu`, `cuda`, or `0`. `auto` picks PyTorch `mps` on Apple Silicon. Set `mlx` (or `YOLO_RUNTIME=mlx`) for yolo-mlx. |
| `TRACKING_ENABLED` | `true` | Keep stable IDs across frames. On the MLX path this uses official `model.track(..., persist=True)`. |
| `TRACKER_TYPE` | `bytetrack.yaml` | `bytetrack.yaml` or `botsort.yaml` (same YAML names as [yolo-mlx tracking](https://github.com/thewebAI/yolo-mlx/blob/main/GUIDE_TRACKING.md)). |

First run downloads the named `.pt` file into `models/`. Weights that Ultralytics previously dropped in the project root are moved there automatically. The MLX backend converts `.pt` → `.npz` once and reuses it.

To use **[yolo-mlx](https://github.com/thewebAI/yolo-mlx)** instead of PyTorch MPS, install the official extras (`[tracking]` for `model.track()`, `[convert]` for `.pt` → `.npz`):

```bash
pip install "yolo-mlx[tracking,convert]"
```

```env
YOLO_RUNTIME=mlx
# or: YOLO_DEVICE=mlx
```

Senti calls the same conversion API as the official CLI, including `--verify`:

```bash
yolo-mlx converters convert models/yolo26n.pt -o models/yolo26n.npz --verify
```

You can run that yourself if you prefer. Live frames are passed as OpenCV BGR numpy arrays, which is what yolo-mlx’s predictor expects. `save` and `show` stay `False` so nothing is written under `results/`. yolo-mlx is AGPL-3.0; it stays an optional extra so Senti’s MIT default path does not pull it in.

To stay on **PyTorch MPS**:

```env
YOLO_RUNTIME=ultralytics
YOLO_DEVICE=mps
```

## 🧭 Scene change and VLM scheduling

| Variable | Default | Description |
| --- | --- | --- |
| `SCENE_CHANGE_THRESHOLD` | `0.15` | Combined visual + object + spatial score in `(0, 1]`. Lower is more sensitive. |
| `STABILITY_FRAMES` | `10` | Consecutive stable frames required before `READY`. |
| `VLM_COOLDOWN` | `5.0` | Minimum seconds between automatic analyses. |
| `VLM_AUTO_ANALYZE` | `true` | Send the best frame to the VLM when the scene reaches `READY`. |
| `CONVERSATION_MAX_TURNS` | `8` | Recent Q&A turns kept for follow-ups. |
| `CONTEXT_STALE_FRAMES` | `45` | After this many new camera frames, follow-ups use a fresh capture. |
| `OBJECT_CROP_ENABLED` | `true` | Crop the focused object before a targeted VLM call. |
| `OBJECT_CROP_PADDING` | `0.15` | Padding around the box as a fraction of bbox size, in `[0, 1]`. |

## 🧠 Vision-language model

| Variable | Default | Description |
| --- | --- | --- |
| `VLM_MODEL` | `gemma4` | Ollama model name (must exist locally). |
| `VLM_RUNTIME` | `ollama` | Only `ollama` is supported. |
| `VLM_BASE_URL` | `http://localhost:11434` | Ollama HTTP API. |

Install Ollama, then:

```bash
ollama pull gemma4
# or: ollama pull llava
```

The app checks that the requested tag exists (including `:latest` aliases). If Ollama is down, the VLM status stays inactive and **Analyze** reports the error instead of hanging the UI.

## 🔤 OCR

| Variable | Default | Description |
| --- | --- | --- |
| `OCR_ENABLED` | `false` | Start the OCR worker. |
| `OCR_RUNTIME` | `easyocr` | Only `easyocr` is supported. |
| `OCR_LANGUAGES` | `en` | Comma-separated EasyOCR language codes. |
| `OCR_MIN_CONFIDENCE` | `0.4` | Drop lines below this score, in `[0, 1]`. |
| `OCR_AUTO_ON_READY` | `true` | Run OCR when the scene becomes `READY`. |

EasyOCR downloads models on first use and runs on CPU on Mac (no MPS path). That is expected.

## 🔊 Text-to-speech

| Variable | Default | Description |
| --- | --- | --- |
| `TTS_ENABLED` | `false` | Create the speech controller. |
| `TTS_RUNTIME` | `auto` | `auto`, `qt`, or `say`. |
| `TTS_AUTO_SPEAK` | `true` | Speak new VLM/OCR answers without clicking **Speak**. |
| `TTS_RATE` | `0.0` | Rate from `-1.0` (slow) to `1.0` (fast). |
| `TTS_VOLUME` | `1.0` | Volume in `[0.0, 1.0]`. |
| `TTS_VOICE` | *(empty)* | Optional macOS voice name substring (e.g. `Tessa`). |
| `TTS_INTERRUPT` | `true` | Stop current speech when new text arrives. |

`auto` tries Qt first and falls back to `/usr/bin/say` when the named voice is not in Qt’s list.

## 🎙️ Voice input

| Variable | Default | Description |
| --- | --- | --- |
| `VOICE_ENABLED` | `false` | Start the voice worker and show **Mic**. |
| `VOICE_RUNTIME` | `whisper` | Only `whisper` (faster-whisper) is supported. |
| `VOICE_MODEL` | `base` | `tiny`, `base`, `small`, … Larger is slower and more accurate. |
| `VOICE_LANGUAGE` | `en` | Transcription language. Empty string lets Whisper auto-detect. |
| `VOICE_MAX_SECONDS` | `8.0` | Auto-stop recording after this duration. |
| `VOICE_MIN_SECONDS` | `0.6` | Ignore recordings shorter than this. |
| `VOICE_SAMPLE_RATE` | `16000` | Capture sample rate in Hz. |

The Whisper weights load on first **Mic** click, not at app start.

## 🖥️ App

| Variable | Default | Description |
| --- | --- | --- |
| `DEBUG_MODE` | `false` | `DEBUG` logging and unfiltered third-party warnings. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |

## 💡 Suggested setups

**Lean (detection only)** — leave OCR, TTS, and voice off. You still get boxes, IDs, and optional VLM if Ollama is running.

**Desk assistant** — `OCR_ENABLED=true`, `TTS_ENABLED=true`, `TTS_AUTO_SPEAK=false`, `YOLO_MODEL=yolo26s.pt`.

**Hands-free** — desk assistant plus `VOICE_ENABLED=true` and `TTS_AUTO_SPEAK=true`. Grant microphone permission.

## 🔗 See also

- [Usage](usage.md) — what the controls do
- [Troubleshooting](troubleshooting.md) — common log lines
- [`.env.example`](../.env.example) — checked-in template
