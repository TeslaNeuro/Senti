<p align="center">
  <img src="docs/assets/senti-logo.png" width="168" alt="Senti iris logo">
</p>

<h1 align="center">Senti</h1>

<p align="center">
  <b>What Am I Looking At?</b><br>
  A local-first visual assistant for macOS — it sees the scene, names the objects, and answers your questions. On your Mac. Nothing uploaded.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="MIT License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="https://www.apple.com/mac/"><img src="https://img.shields.io/badge/macOS-Apple%20Silicon-000000?style=for-the-badge&logo=apple&logoColor=white" alt="macOS Apple Silicon"></a>
  <a href="#privacy"><img src="https://img.shields.io/badge/Privacy-Local--first-34c759?style=for-the-badge&logo=letsencrypt&logoColor=white" alt="Local-first privacy"></a>
  <a href="https://docs.pytest.org/"><img src="https://img.shields.io/badge/pytest-unit%20tests-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="pytest"></a>
</p>

<p align="center">
  <a href="#quick-start"><strong>Quick start</strong></a> ·
  <a href="#how-it-works"><strong>How it works</strong></a> ·
  <a href="#stack"><strong>Stack</strong></a> ·
  <a href="docs/README.md"><strong>Docs</strong></a> ·
  <a href="#license"><strong>License</strong></a>
</p>

<p align="center">
  <img src="docs/assets/senti-banner.jpg" alt="Senti — local-first visual assistant on macOS" width="100%">
</p>

## ✨ Why Senti

Most visual assistants ship camera frames to a cloud API. **Senti does not.**

It is built for Apple Silicon so the fast path — object detection and tracking — stays at interactive rates, while the slow path — a local vision-language model via [Ollama](https://ollama.com) — only runs when the scene actually changes or you ask a question.

| 🎯 Live boxes & track IDs | 💬 Follow-up questions | 🔍 Object focus |
| :---: | :---: | :---: |
| YOLO26 labels, confidence, stable `#1` `#2` IDs | *“What’s that connector?”* *“What objects do you see?”* | Crop `#2 phone` before the VLM sees it |
| 📝 On-device OCR | 🔊 Spoken answers | 🎙️ Push-to-talk |
| Ask `read this` or auto-read on `READY` | macOS TTS (Qt or `say`) | Local Whisper, **Esc** to cancel |

## 🚀 Features

| | Capability | Detail |
| :---: | --- | --- |
| 🎥 | **Live perception** | YOLO26 on Apple Silicon — PyTorch `mps` or yolo-mlx Metal — plus ByteTrack / BoT-SORT IDs |
| 🧭 | **Scene intelligence** | Visual + object + spatial change detection, stability gating, best-frame selection |
| 🧠 | **Local understanding** | Ollama VLM, auto-analysis when the scene settles, conversational memory |
| ✂️ | **Object focus** | Padded crop of the most likely target before a focused question |
| 🔤 | **On-device OCR** | EasyOCR on demand (`read this`) or automatically when the scene is ready |
| 🗣️ | **Speech I/O** | macOS TTS and local Whisper push-to-talk |
| 🔐 | **Privacy by design** | No cloud uploads, no disk recordings, bounded in-memory frame buffer |

<a id="how-it-works"></a>

## 🧠 How it works

Two loops keep the UI live. The camera never waits on the language model.

<p align="center">
  <img src="docs/assets/senti-pipeline.jpg" alt="Camera → detect → understand → speak" width="100%">
</p>

```mermaid
flowchart LR
  Cam["📷 Camera"] --> Fast
  subgraph Fast["⚡ Fast loop — 15–30 FPS"]
    YOLO["🎯 YOLO26 + tracking"]
    Scene["🧭 Scene change + stability"]
    YOLO --> Scene
  end
  Fast --> Slow
  subgraph Slow["🌙 Slow loop — on change or question"]
    Frame["🖼️ Best-frame selection"]
    OCR["🔤 Optional OCR"]
    VLM["🧠 Local VLM"]
    Frame --> OCR --> VLM
  end
  Slow --> UI["🖥️ Desktop UI + TTS"]
  Mic["🎙️ Push-to-talk"] --> UI
  UI --> Ask["💬 Ask / Focus / Analyze"]
  Ask --> Slow
```

1. **⚡ Fast loop** — frames go to YOLO26, then tracking and scene-change detection. Target: 15–30 FPS.
2. **🌙 Slow loop** — when the scene reaches `READY`, Senti picks the sharpest, most stable frame from a rolling buffer and sends it to the local VLM (and OCR, if enabled). Follow-ups reuse scene memory when they can.

State machine: `WATCHING` → `SCENE_CHANGED` → `WAITING_FOR_STABILITY` → `READY`

Full package map and thread model: **[Architecture](docs/architecture.md)**.

<a id="stack"></a>

## 🧩 Stack

Every runtime dependency, with a badge that opens its **official site**. Click through — these are the projects Senti stands on.

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://www.apple.com/macos/"><img src="https://img.shields.io/badge/macOS-Apple%20Silicon-000000?style=for-the-badge&logo=apple&logoColor=white" alt="macOS"></a>
  <a href="https://doc.qt.io/qtforpython-6/"><img src="https://img.shields.io/badge/PySide6-Qt%20for%20Python-41CD52?style=for-the-badge&logo=qt&logoColor=white" alt="PySide6"></a>
  <a href="https://www.qt.io/"><img src="https://img.shields.io/badge/Qt-Multimedia%20%2B%20TTS-41CD52?style=for-the-badge&logo=qt&logoColor=white" alt="Qt"></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-2.x-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="https://opencv.org/"><img src="https://img.shields.io/badge/OpenCV-Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"></a>
  <a href="https://www.ultralytics.com/"><img src="https://img.shields.io/badge/Ultralytics-YOLO26-111F68?style=for-the-badge&logo=ultralytics&logoColor=white" alt="Ultralytics"></a>
  <a href="https://docs.ultralytics.com/models/yolo26/"><img src="https://img.shields.io/badge/YOLO26-Detection-FF6F00?style=for-the-badge" alt="YOLO26"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-MPS-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://github.com/ml-explore/mlx"><img src="https://img.shields.io/badge/MLX-Metal-000000?style=for-the-badge" alt="MLX"></a>
  <a href="https://github.com/thewebAI/yolo-mlx"><img src="https://img.shields.io/badge/yolo--mlx-optional-FF6F00?style=for-the-badge" alt="yolo-mlx"></a>
  <a href="https://ollama.com/"><img src="https://img.shields.io/badge/Ollama-Local%20VLM-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama"></a>
  <a href="https://www.jaided.ai/easyocr/"><img src="https://img.shields.io/badge/EasyOCR-On--device%20text-1A73E8?style=for-the-badge" alt="EasyOCR"></a>
  <a href="https://github.com/SYSTRAN/faster-whisper"><img src="https://img.shields.io/badge/faster--whisper-STT-7C5CFF?style=for-the-badge" alt="faster-whisper"></a>
  <a href="https://python-sounddevice.readthedocs.io/"><img src="https://img.shields.io/badge/sounddevice-Mic%20capture-1f425f?style=for-the-badge" alt="sounddevice"></a>
  <a href="https://github.com/theskumar/python-dotenv"><img src="https://img.shields.io/badge/python--dotenv-Config-ECD53F?style=for-the-badge" alt="python-dotenv"></a>
  <a href="https://docs.pytest.org/"><img src="https://img.shields.io/badge/pytest-Tests-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="pytest"></a>
</p>

|  | Project | Official site | Role in Senti |
| :---: | --- | --- | --- |
| <img src="https://cdn.simpleicons.org/python/3776AB" width="28" alt=""> | **Python** | [python.org](https://www.python.org/) | Runtime (3.11+) |
| <img src="https://cdn.simpleicons.org/apple/000000" width="28" alt=""> | **macOS** | [apple.com/macos](https://www.apple.com/macos/) | Camera, TTS `say`, permissions |
| <img src="https://cdn.simpleicons.org/qt/41CD52" width="28" alt=""> | **PySide6 / Qt** | [doc.qt.io/qtforpython-6](https://doc.qt.io/qtforpython-6/) · [qt.io](https://www.qt.io/) | Native window, AVFoundation capture, TTS |
| <img src="https://cdn.simpleicons.org/numpy/013243" width="28" alt=""> | **NumPy** | [numpy.org](https://numpy.org/) | Frame arrays |
| <img src="https://cdn.simpleicons.org/opencv/5C3EE8" width="28" alt=""> | **OpenCV** | [opencv.org](https://opencv.org/) | Overlays, sharpness, scene diff |
| <img src="https://cdn.simpleicons.org/ultralytics/111F68" width="28" alt=""> | **Ultralytics YOLO26** | [ultralytics.com](https://www.ultralytics.com/) · [YOLO26 docs](https://docs.ultralytics.com/models/yolo26/) | Detection + tracking |
| <img src="https://cdn.simpleicons.org/pytorch/EE4C2C" width="28" alt=""> | **PyTorch** | [pytorch.org](https://pytorch.org/) | Default YOLO path: MPS on Apple Silicon |
| 🍎 | **MLX / yolo-mlx** | [MLX](https://github.com/ml-explore/mlx) · [yolo-mlx](https://github.com/thewebAI/yolo-mlx) | Optional native Metal detector (`YOLO_RUNTIME=mlx`; `pip install "yolo-mlx[tracking,convert]"`) |
| <img src="https://cdn.simpleicons.org/ollama/111111" width="28" alt=""> | **Ollama** | [ollama.com](https://ollama.com/) | Local vision-language model |
| 🔤 | **EasyOCR** | [jaided.ai/easyocr](https://www.jaided.ai/easyocr/) · [GitHub](https://github.com/JaidedAI/EasyOCR) | On-device text recognition |
| 🎙️ | **faster-whisper** | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Push-to-talk transcription |
| 🎚️ | **sounddevice** | [python-sounddevice.readthedocs.io](https://python-sounddevice.readthedocs.io/) | Microphone capture |
| ⚙️ | **python-dotenv** | [GitHub](https://github.com/theskumar/python-dotenv) | `.env` configuration |
| <img src="https://cdn.simpleicons.org/pytest/0A9EDC" width="28" alt=""> | **pytest** | [docs.pytest.org](https://docs.pytest.org/) | Unit tests |

Pinned versions live in [`requirements.txt`](requirements.txt).

## 📋 Requirements

- 🍎 macOS on Apple Silicon (developed on M2 Pro)
- 🐍 Python 3.11 or newer
- 📷 Built-in or Continuity Camera
- 🔐 Camera permission (and microphone if voice input is on)
- 🦙 [Ollama](https://ollama.com) with a vision model for scene descriptions

<a id="quick-start"></a>

## ⚡ Quick start

```bash
git clone https://github.com/TeslaNeuro/Senti.git
cd Senti

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Pull a local vision model, then launch:

```bash
ollama pull gemma4
./scripts/run.sh
```

Or:

```bash
source .venv/bin/activate
python -m app
```

On first launch, macOS will ask for camera access. Grant it to **Terminal** (or your IDE) if you start Senti from the command line. Ultralytics downloads `yolo26n.pt` into `models/` automatically (~6 MB) unless that file is already there.

Optional native Metal path: install [yolo-mlx](https://github.com/thewebAI/yolo-mlx) with `pip install "yolo-mlx[tracking,convert]"`, then set `YOLO_RUNTIME=mlx`.

> 💡 If the preview is black, another app (FaceTime, Zoom, Chrome, …) likely holds the camera. Quit it and relaunch.

## 🖥️ Using the app

| Action | What happens |
| --- | --- |
| 👀 Watch the preview | Live boxes, labels, confidence, track IDs (`#1`, `#2`, …) |
| ⏳ Wait for `READY` | Automatic VLM description of the best buffered frame |
| 💬 Type in **Ask** | Follow-up against scene memory, or a new VLM call when needed |
| 🎯 **Focus** dropdown | Analyze a specific tracked object (cropped when enabled) |
| 🔁 **Analyze** | Force a fresh VLM pass (bypasses cooldown) |
| 🧹 **Clear** | Reset scene memory and the response panel |
| 🔊 **Speak** / *say that* | Replay the current answer (when TTS is on) |
| 🎙️ **Mic** / **Stop** | Push-to-talk; **Esc** cancels an in-progress recording |
| 📖 *read this* | Run OCR on the current scene |

Status bar shows camera, YOLO device, VLM activity, FPS, and inference latency. Hover a pill for the full detail.

Step-by-step walkthrough: **[Usage](docs/usage.md)**.

## ⚙️ Configuration

Copy `.env.example` to `.env`. Important defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | `1280` / `720` | Capture size |
| `YOLO_MODEL` | `yolo26n.pt` | Weights filename; stored in `models/` (`yolo26s.pt` is more accurate) |
| `YOLO_RUNTIME` | `auto` | `auto` (Ultralytics unless `YOLO_DEVICE=mlx`), `ultralytics` (PyTorch MPS), or `mlx` (yolo-mlx Metal) |
| `YOLO_DEVICE` | `auto` | `auto` → PyTorch `mps` on Apple Silicon; `mlx` for yolo-mlx |
| `VLM_MODEL` | `gemma4` | Ollama vision model |
| `VLM_BASE_URL` | `http://localhost:11434` | Local Ollama API |
| `OCR_ENABLED` | `false` | On-device text recognition |
| `TTS_ENABLED` | `false` | Speak answers aloud |
| `VOICE_ENABLED` | `false` | Whisper push-to-talk |

Optional features (OCR, TTS, voice) are off until you turn them on. Full reference: **[Configuration](docs/configuration.md)**.

<details>
<summary>🎛 Optional extras</summary>

```env
OCR_ENABLED=true
TTS_ENABLED=true
TTS_RUNTIME=auto
VOICE_ENABLED=true
VOICE_MODEL=base
```

`TTS_RUNTIME=auto` uses Qt when it exposes your voice, otherwise the macOS `say` command (so names like **Tessa** still work). First OCR or Whisper use downloads models in the background.

</details>

## 🧪 Tests

```bash
source .venv/bin/activate
pytest tests/ -q
```

## 📚 Documentation

| | Guide | Contents |
| :---: | --- | --- |
| 🏗️ | [Architecture](docs/architecture.md) | Layers, threads, scene states, package map |
| 🖱️ | [Usage](docs/usage.md) | Window tour, questions, focus, speech, voice |
| ⚙️ | [Configuration](docs/configuration.md) | Every `.env` setting and sensible ranges |
| 🩺 | [Troubleshooting](docs/troubleshooting.md) | Camera, Qt, Ollama, OCR, TTS, Whisper |
| 🛡️ | [Security](SECURITY.md) | Privacy model and how to report issues |
| 📜 | [License](LICENSE) | MIT License |

<a id="privacy"></a>

## 🔐 Privacy

Senti is **local-first**:

- 🚫 Camera frames are never written to disk
- 🧠 The frame buffer is bounded and in-memory only
- 💻 YOLO, OCR, Whisper, and the VLM run on-device (Ollama on localhost)
- 📡 No telemetry, no cloud uploads, no account

See [SECURITY.md](SECURITY.md) for the threat model and reporting.

## 📁 Project layout

```text
Senti/
├── app/                 Application package
│   ├── camera/          Qt / AVFoundation capture + in-memory buffer
│   ├── detection/       YOLO26 worker
│   ├── tracking/        ByteTrack / BoT-SORT monitor
│   ├── perception/      Scene change, stability, best-frame selection
│   ├── vision/          Local VLM, scheduler, object crops
│   ├── scene/           Conversational scene memory
│   ├── ocr/             EasyOCR worker
│   ├── speech/          Qt TTS + macOS say
│   ├── voice/           Push-to-talk + faster-whisper
│   └── ui/              Native desktop window
├── docs/                Architecture, usage, configuration
│   └── assets/          README graphics (logo, banner, pipeline)
├── models/              YOLO26 weights (gitignored *.pt / *.npz)
├── scripts/run.sh       Create venv, install, launch
├── tests/               Unit tests
└── resources/Info.plist Camera and microphone usage strings
```

<a id="license"></a>

## 📄 License

Senti is released under the [MIT License](LICENSE).

Copyright (c) 2026 **Arshia Keshvari**.

<p align="center">
  <img src="docs/assets/senti-logo.png" width="72" alt="Senti">
  <br>
  <sub>Built on-device. Nothing leaves your Mac.</sub>
</p>
