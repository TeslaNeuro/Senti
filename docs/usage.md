# 🖱️ Usage

Copyright (c) 2026 Arshia Keshvari. Licensed under the MIT License.

This guide covers the desktop window after `./scripts/run.sh` (or `python -m app`) succeeds.

## 🚀 First launch

1. Grant **camera** access when macOS prompts. If you started Senti from a terminal, the prompt is for Terminal (or iTerm, Cursor, VS Code, …), not a bundled `.app`.
2. Wait for the YOLO status pill to turn green. It should read `mps` (default PyTorch) or `mlx` if you set `YOLO_RUNTIME=mlx`. The first run downloads Ultralytics weights into `models/` if they are missing; the MLX path also converts that `.pt` to `.npz` once.
3. Point the camera at a desk, a label, or a room. Boxes and track IDs should appear within a second or two.
4. When the scene state reaches `READY`, the assistant describes what it sees (if Ollama is running and `VLM_AUTO_ANALYZE=true`).

If the preview is black, see [Troubleshooting](troubleshooting.md).

## 🪟 Window layout

- **Preview** — live camera with colored boxes. Each track keeps the same ID (`#1`, `#2`, …) while it stays in view. Color is derived from the track ID.
- **Response panel** — latest VLM or OCR answer, plus routed replies from scene memory.
- **Ask** — text field for questions and commands.
- **Focus** — dropdown of current tracks for object-specific questions.
- **Analyze** — force a VLM pass on the current best frame.
- **Clear** — wipe scene memory and the response panel.
- **Speak** — replay the last answer (when TTS is enabled).
- **Mic / Stop** — push-to-talk (when voice is enabled).
- **Status bar** — Camera, YOLO26 (device), VLM, FPS, latency, scene state. Hover a pill for the full string.

Controls disable while analyzing, transcribing, or running OCR so you cannot stack conflicting jobs.

## 📡 Scene states

Watch the status line:

| State | What you should do |
| --- | --- |
| `WATCHING` | Hold still or keep looking; nothing important changed |
| `SCENE_CHANGED` | You moved the camera or objects entered / left |
| `WAITING_FOR_STABILITY` | Pause motion so a sharp frame can be chosen |
| `READY` | Automatic description may run; good time to Ask |

Rapid motion keeps the app in change / stability states and delays automatic VLM calls on purpose.

## 💬 Asking questions

Type in **Ask** and press Return. Examples:

| You type | Typical route |
| --- | --- |
| `what objects do you see?` | List from current detections (no new VLM call) |
| `what am I looking at?` | Cached or fresh VLM description |
| `what's that connector?` | VLM, often with an object crop if Focus is set |
| `read this` | OCR on the current frame |
| `say that` | Speak the last answer |

If context is stale (the camera has moved a lot since the last analysis), Senti captures a fresh frame instead of reusing the old one.

## 🎯 Focusing on an object

1. Wait until boxes have stable IDs.
2. Open **Focus** and pick e.g. `#2 phone`.
3. Ask a question, or click **Analyze**.

With `OBJECT_CROP_ENABLED=true` (default), the VLM receives a padded crop around that box rather than the whole scene.

## 🔤 OCR

Enable OCR in `.env` (`OCR_ENABLED=true`). Then:

- Ask `read this` / `what does this say?`
- Or let it run automatically when the scene hits `READY` (`OCR_AUTO_ON_READY=true`)

Detected lines are shown and also passed to the VLM as extra context. The first EasyOCR run downloads language models and is slower.

## 🔊 Text-to-speech

Enable with `TTS_ENABLED=true`. New answers can speak automatically (`TTS_AUTO_SPEAK`) or only when you click **Speak**.

Qt lists a small set of voices. If `TTS_VOICE=Tessa` (or another system voice) is missing from Qt, leave `TTS_RUNTIME=auto` so Senti uses macOS `say`.

## 🎙️ Voice input

Enable with `VOICE_ENABLED=true`, then grant **microphone** access.

1. Click **Mic**.
2. Speak a question.
3. Click **Stop**, or wait until `VOICE_MAX_SECONDS`.
4. Press **Esc** to cancel without sending.

Very short taps below `VOICE_MIN_SECONDS` are ignored. The Whisper model loads the first time you use the mic.

## ⌨️ Keyboard

| Key | Action |
| --- | --- |
| Return in Ask | Submit the question |
| Esc | Stop an in-progress voice recording |

## 💡 Everyday tips

- Prefer a well-lit, mostly still scene for OCR and VLM quality.
- Use `yolo26s.pt` in `.env` if `yolo26n.pt` misses small objects (slower, more accurate).
- Click **Clear** after you physically change what is in front of the camera and the description feels stuck.
- Keep Ollama running locally; the VLM pill stays inactive if the daemon or model is missing.
