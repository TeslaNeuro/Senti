# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Centralized application configuration."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _PROJECT_ROOT
_ENV_PATH = _PROJECT_ROOT / ".env"
_ENV_EXAMPLE_PATH = _PROJECT_ROOT / ".env.example"


def models_directory(project_root: Path | None = None) -> Path:
    """Return ``<root>/models``, creating the directory if needed."""
    path = (project_root or PROJECT_ROOT) / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_env_files(env_path: Path | None = None) -> None:
    """Load .env, falling back to .env.example when .env is missing."""
    path = env_path or _ENV_PATH
    if path.exists():
        warn_duplicate_env_keys(path)
        load_dotenv(path, override=False)
        return
    if _ENV_EXAMPLE_PATH.exists():
        warn_duplicate_env_keys(_ENV_EXAMPLE_PATH)
        load_dotenv(_ENV_EXAMPLE_PATH, override=False)


def find_duplicate_env_keys(path: Path) -> list[str]:
    """Return env variable names that appear more than once in a dotenv file."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if not key:
            continue
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    return duplicates


def warn_duplicate_env_keys(path: Path) -> None:
    """Log a warning when duplicate keys exist (last value wins in dotenv)."""
    duplicates = find_duplicate_env_keys(path)
    if not duplicates:
        return
    joined = ", ".join(duplicates)
    logging.getLogger(__name__).warning(
        "%s defines duplicate keys (%s). The last value in the file wins.",
        path.name,
        joined,
    )


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {key}: {raw!r}") from exc


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {key}: {raw!r}") from exc


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration loaded from environment variables."""

    # Camera
    camera_device: int
    camera_width: int
    camera_height: int
    target_fps: int
    frame_buffer_size: int
    selection_buffer_size: int

    # Detection (YOLO26)
    yolo_model: str
    yolo_confidence: float
    yolo_image_size: int
    yolo_device: str
    yolo_runtime: str

    # Tracking
    tracking_enabled: bool
    tracker_type: str

    # Scene change and VLM
    scene_change_threshold: float
    stability_frames: int
    vlm_cooldown: float
    vlm_auto_analyze: bool
    vlm_model: str
    vlm_runtime: str
    vlm_base_url: str
    conversation_max_turns: int
    context_stale_frames: int
    object_crop_enabled: bool
    object_crop_padding: float

    # Optional features
    ocr_enabled: bool
    ocr_runtime: str
    ocr_languages: str
    ocr_min_confidence: float
    ocr_auto_on_ready: bool
    tts_enabled: bool
    tts_runtime: str
    tts_auto_speak: bool
    tts_rate: float
    tts_volume: float
    tts_voice: str
    tts_interrupt: bool
    voice_enabled: bool
    voice_runtime: str
    voice_model: str
    voice_language: str
    voice_max_seconds: float
    voice_min_seconds: float
    voice_sample_rate: int

    # App
    debug_mode: bool
    log_level: str

    @classmethod
    def load(cls, env_path: Path | None = None) -> AppConfig:
        _load_env_files(env_path)
        return cls(
            camera_device=_env_int("CAMERA_DEVICE", 0),
            camera_width=_env_int("CAMERA_WIDTH", 1280),
            camera_height=_env_int("CAMERA_HEIGHT", 720),
            target_fps=_env_int("TARGET_FPS", 30),
            frame_buffer_size=_env_int("FRAME_BUFFER_SIZE", 8),
            selection_buffer_size=_env_int("SELECTION_BUFFER_SIZE", 12),
            yolo_model=os.getenv("YOLO_MODEL", "yolo26n.pt"),
            yolo_confidence=_env_float("YOLO_CONFIDENCE", 0.5),
            yolo_image_size=_env_int("YOLO_IMAGE_SIZE", 640),
            yolo_device=os.getenv("YOLO_DEVICE", "auto"),
            yolo_runtime=os.getenv("YOLO_RUNTIME", "auto"),
            tracking_enabled=_env_bool("TRACKING_ENABLED", True),
            tracker_type=os.getenv("TRACKER_TYPE", "bytetrack.yaml"),
            scene_change_threshold=_env_float("SCENE_CHANGE_THRESHOLD", 0.15),
            stability_frames=_env_int("STABILITY_FRAMES", 10),
            vlm_cooldown=_env_float("VLM_COOLDOWN", 5.0),
            vlm_auto_analyze=_env_bool("VLM_AUTO_ANALYZE", True),
            vlm_model=os.getenv("VLM_MODEL", "gemma4"),
            vlm_runtime=os.getenv("VLM_RUNTIME", "ollama"),
            vlm_base_url=os.getenv("VLM_BASE_URL", "http://localhost:11434"),
            conversation_max_turns=_env_int("CONVERSATION_MAX_TURNS", 8),
            context_stale_frames=_env_int("CONTEXT_STALE_FRAMES", 45),
            object_crop_enabled=_env_bool("OBJECT_CROP_ENABLED", True),
            object_crop_padding=_env_float("OBJECT_CROP_PADDING", 0.15),
            ocr_enabled=_env_bool("OCR_ENABLED", False),
            ocr_runtime=os.getenv("OCR_RUNTIME", "easyocr"),
            ocr_languages=os.getenv("OCR_LANGUAGES", "en"),
            ocr_min_confidence=_env_float("OCR_MIN_CONFIDENCE", 0.4),
            ocr_auto_on_ready=_env_bool("OCR_AUTO_ON_READY", True),
            tts_enabled=_env_bool("TTS_ENABLED", False),
            tts_runtime=os.getenv("TTS_RUNTIME", "auto"),
            tts_auto_speak=_env_bool("TTS_AUTO_SPEAK", True),
            tts_rate=_env_float("TTS_RATE", 0.0),
            tts_volume=_env_float("TTS_VOLUME", 1.0),
            tts_voice=os.getenv("TTS_VOICE", "").strip(),
            tts_interrupt=_env_bool("TTS_INTERRUPT", True),
            voice_enabled=_env_bool("VOICE_ENABLED", False),
            voice_runtime=os.getenv("VOICE_RUNTIME", "whisper"),
            voice_model=os.getenv("VOICE_MODEL", "base"),
            voice_language=os.getenv("VOICE_LANGUAGE", "en"),
            voice_max_seconds=_env_float("VOICE_MAX_SECONDS", 8.0),
            voice_min_seconds=_env_float("VOICE_MIN_SECONDS", 0.6),
            voice_sample_rate=_env_int("VOICE_SAMPLE_RATE", 16000),
            debug_mode=_env_bool("DEBUG_MODE", False),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )

    def validate(self) -> None:
        """Raise ValueError for invalid configuration."""
        if self.camera_width <= 0 or self.camera_height <= 0:
            raise ValueError("Camera dimensions must be positive.")
        if self.target_fps <= 0:
            raise ValueError("TARGET_FPS must be positive.")
        if self.frame_buffer_size <= 0:
            raise ValueError("FRAME_BUFFER_SIZE must be positive.")
        if self.selection_buffer_size <= 0:
            raise ValueError("SELECTION_BUFFER_SIZE must be positive.")
        if not 0.0 < self.yolo_confidence <= 1.0:
            raise ValueError("YOLO_CONFIDENCE must be in (0, 1].")
        if self.yolo_image_size <= 0:
            raise ValueError("YOLO_IMAGE_SIZE must be positive.")
        if self.yolo_device.strip().lower() not in {"auto", "mps", "mlx", "cpu", "cuda", "0"}:
            raise ValueError("YOLO_DEVICE must be one of: auto, mps, mlx, cpu, cuda, 0.")
        if self.yolo_runtime.strip().lower() not in {"auto", "ultralytics", "mlx"}:
            raise ValueError("YOLO_RUNTIME must be one of: auto, ultralytics, mlx.")
        if self.tracker_type not in {"bytetrack.yaml", "botsort.yaml"}:
            raise ValueError("TRACKER_TYPE must be bytetrack.yaml or botsort.yaml.")
        if not 0.0 < self.scene_change_threshold <= 1.0:
            raise ValueError("SCENE_CHANGE_THRESHOLD must be in (0, 1].")
        if self.stability_frames <= 0:
            raise ValueError("STABILITY_FRAMES must be positive.")
        if self.vlm_cooldown < 0:
            raise ValueError("VLM_COOLDOWN must be non-negative.")
        if self.conversation_max_turns <= 0:
            raise ValueError("CONVERSATION_MAX_TURNS must be positive.")
        if self.context_stale_frames <= 0:
            raise ValueError("CONTEXT_STALE_FRAMES must be positive.")
        if not 0.0 <= self.object_crop_padding <= 1.0:
            raise ValueError("OBJECT_CROP_PADDING must be in [0, 1].")
        if not 0.0 <= self.ocr_min_confidence <= 1.0:
            raise ValueError("OCR_MIN_CONFIDENCE must be in [0, 1].")
        if self.ocr_runtime.strip().lower() not in {"easyocr"}:
            raise ValueError("OCR_RUNTIME must be 'easyocr'.")
        if not -1.0 <= self.tts_rate <= 1.0:
            raise ValueError("TTS_RATE must be in [-1, 1].")
        if not 0.0 <= self.tts_volume <= 1.0:
            raise ValueError("TTS_VOLUME must be in [0, 1].")
        if self.tts_runtime.strip().lower() not in {"auto", "qt", "say"}:
            raise ValueError("TTS_RUNTIME must be one of: auto, qt, say.")
        if self.voice_runtime.strip().lower() not in {"whisper"}:
            raise ValueError("VOICE_RUNTIME must be 'whisper'.")
        if self.voice_max_seconds <= 0:
            raise ValueError("VOICE_MAX_SECONDS must be positive.")
        if self.voice_min_seconds < 0:
            raise ValueError("VOICE_MIN_SECONDS must be non-negative.")
        if self.voice_sample_rate <= 0:
            raise ValueError("VOICE_SAMPLE_RATE must be positive.")
        if self.vlm_runtime.strip().lower() not in {"ollama"}:
            raise ValueError("VLM_RUNTIME must be 'ollama'.")
        if not self.vlm_model.strip():
            raise ValueError("VLM_MODEL must be set.")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Unsupported LOG_LEVEL: {self.log_level}")


def setup_logging(config: AppConfig) -> None:
    """Configure structured console logging."""
    level = getattr(logging, config.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    if config.debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
