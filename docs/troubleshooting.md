# 🩺 Troubleshooting

Copyright (c) 2026 Arshia Keshvari. Licensed under the MIT License.

## 📷 Camera

**Black preview or a camera error.** Another process owns the camera (FaceTime, Zoom, Teams, Chrome, Photo Booth). Quit those apps and relaunch Senti.

**Permission dialog never appears / access denied.** Enable the host app under **System Settings → Privacy & Security → Camera**. If you run `python -m app` from a terminal, the host is Terminal, iTerm, Cursor, or VS Code — not “Senti”.

**Continuity Camera / iPhone.** `resources/Info.plist` sets `NSCameraUseContinuityCameraDeviceType`. If the wrong camera is selected, change `CAMERA_DEVICE` in `.env`.

**`NSCameraUseContinuityCameraDeviceType` in logs.** Informational. `scripts/run.sh` and `app/qt_bootstrap.py` point Qt at `resources/Info.plist`.

## 🧩 Qt / startup

**`Could not find the Qt platform plugin "cocoa"`.** Use `./scripts/run.sh`, or `python -m app` after installing requirements in the project venv. `app/qt_bootstrap.py` sets `QT_QPA_PLATFORM_PLUGIN_PATH` before Qt is imported. Do not import PySide6 in a random script without that bootstrap.

**`zsh: permission denied: ./scripts/run.sh`.** The script is not marked executable. `chmod +x scripts/run.sh`, or run `bash scripts/run.sh`. Do not use `sudo`.

**Issue with PySide6.** `pip install --force-reinstall PySide6`

**Wrong Python.** The venv must be created with Python 3.11+ on Apple Silicon. Rosetta x86_64 Python will be slow or fail on MPS.

## 🎯 YOLO

**First launch is slow.** Ultralytics downloads `yolo26n.pt` (or whatever `YOLO_MODEL` names) into `models/`. Weights that used to appear in the project root are moved into `models/` on launch.

**Low FPS.** Try `yolo26n.pt`, `YOLO_IMAGE_SIZE=640`, and confirm the YOLO status pill shows `mps` (PyTorch) or `mlx` (yolo-mlx). `cpu` on an M-series Mac means MPS was unavailable (wrong torch build, or `YOLO_DEVICE=cpu`).

**Use yolo-mlx (Metal) instead of PyTorch MPS.** Follow [thewebAI/yolo-mlx](https://github.com/thewebAI/yolo-mlx): `pip install "yolo-mlx[tracking,convert]"`, then set `YOLO_RUNTIME=mlx` (or `YOLO_DEVICE=mlx`). First launch converts `models/*.pt` to `models/*.npz` the same way as `yolo-mlx converters convert … --verify`. To go back to PyTorch: `YOLO_RUNTIME=ultralytics` and `YOLO_DEVICE=mps`.

**`yolo-mlx is not installed`.** The config asked for MLX but the extra is missing. Install `yolo-mlx[tracking,convert]`, or switch `YOLO_RUNTIME=ultralytics`.

**MLX tracking has no IDs / `[tracking] extras are required`.** Official `model.track()` needs OpenCV, lap, and scipy. Reinstall with `pip install "yolo-mlx[tracking,convert]"` or set `TRACKING_ENABLED=false`.

**Missed objects.** Raise model size (`yolo26s.pt`) or lower `YOLO_CONFIDENCE` slightly. Tracking needs detections on consecutive frames to keep an ID.

## 🦙 VLM / Ollama

**VLM stays inactive.** Start Ollama (`ollama serve` if it is not a login item), then `ollama pull gemma4` (or your `VLM_MODEL`). Check `VLM_BASE_URL`.

**Model not found.** The tag in `.env` must match `ollama list`. `gemma4` and `gemma4:latest` are treated as the same family.

**Answers feel stale.** Click **Analyze**, or **Clear** after you change the physical scene. Increase sensitivity (`SCENE_CHANGE_THRESHOLD`) or lower `CONTEXT_STALE_FRAMES` if follow-ups reuse old frames too long.

**“VLM is already running.”** Wait for the current analysis. Manual **Analyze** cannot overlap an in-flight request.

## 🔤 OCR

**`Using CPU` from EasyOCR.** Expected on Mac. EasyOCR has no Apple Silicon GPU path.

**First `read this` takes a long time.** Language models download and load once.

**Garbage text.** Improve lighting, fill more of the frame with the label, and tune `OCR_MIN_CONFIDENCE`.

## 🔊 TTS

**`Voice 'Tessa' is not exposed by Qt; using macOS say backend`.** Normal with `TTS_RUNTIME=auto`. Qt only lists a few voices; `say` can use the rest.

**No sound.** Confirm `TTS_ENABLED=true` and system output volume. Try `TTS_RUNTIME=say` to isolate Qt.

## 🎙️ Voice input

**No Mic button.** `VOICE_ENABLED` must be `true`.

**Microphone denied.** **System Settings → Privacy & Security → Microphone** for the same host app as camera.

**`torch.quantize_per_tensor` deprecation.** Comes from faster-whisper / PyTorch. Hidden unless `DEBUG_MODE=true`.

**`AVFFrameReceiver` / `AVFAudioReceiver` duplicate class.** OpenCV and PyAV both ship FFmpeg symbols. Usually harmless when OCR or voice loads.

**Whisper is slow on first click.** The model loads lazily. `tiny` or `base` is appropriate for short commands; `small` is heavier.

## 🧾 Environment

**Duplicate key warning at startup.** The same variable appears twice in `.env`. Last assignment wins. Remove the extra line.

**Config `ValueError` on launch.** A value is out of range (see [Configuration](configuration.md)). Fix `.env` rather than catching the error.

## 🧪 Tests

```bash
source .venv/bin/activate
pytest tests/ -q
```

Tests do not open the camera. They will not catch a live AVFoundation permission issue.

## 🆘 Still stuck

Run with more logs:

```env
DEBUG_MODE=true
LOG_LEVEL=DEBUG
```

Then open an issue at [github.com/TeslaNeuro/Senti/issues](https://github.com/TeslaNeuro/Senti/issues) with the log excerpt (no camera frames).
