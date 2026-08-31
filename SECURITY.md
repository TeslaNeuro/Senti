# 🛡️ Security

Copyright (c) 2026 Arshia Keshvari. Licensed under the MIT License.

Senti is a local visual assistant. The security posture is **keep camera, microphone, and model traffic on the machine**.

## Privacy model

- **No cloud API** for frames or transcripts. The VLM talks to Ollama at `VLM_BASE_URL` (default `http://localhost:11434`).
- **No recording to disk.** Live frames live in a bounded in-memory buffer and a short selection buffer. They are discarded when the app exits.
- **Permissions.** macOS camera (always) and microphone (only if `VOICE_ENABLED=true`) are requested through Qt. Usage strings live in `resources/Info.plist`.
- **No telemetry, accounts, or analytics.**

If you point `VLM_BASE_URL` at a remote host, you are choosing to send JPEG frames there. The default is localhost.

## What we do not do

- Silent screen or camera capture in the background
- Upload of images, audio, or transcripts to a Senti-operated service
- Persistence of conversation history across launches (memory is process-local)

## Reporting a vulnerability

Please **do not** open a public GitHub issue for a vulnerability that could expose camera frames, microphone audio, or a local RCE path.

Email or message the author, **Arshia Keshvari**, via the GitHub profile attached to [TeslaNeuro/Senti](https://github.com/TeslaNeuro/Senti), and include:

- Affected version / commit
- What an attacker would need locally
- A minimal description of the issue (not a public exploit)

We will work with you on a fix and a disclosure timeline.

## Supply chain

Install from this repository and `requirements.txt` (or `pyproject.toml` metadata) inside a venv. Review `.env` before enabling OCR, TTS, or voice. YOLO, EasyOCR, and Whisper weights are downloaded by those libraries from their upstreams on first use.

Optional [yolo-mlx](https://github.com/thewebAI/yolo-mlx) is AGPL-3.0 and is not in the default install. Only pull it in with `pip install "yolo-mlx[tracking,convert]"` if you want `YOLO_RUNTIME=mlx`. Do not commit downloaded weights (`*.pt`, `*.npz`, `*.safetensors`).
