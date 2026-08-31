# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Local speech-to-text via faster-whisper."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from app.config import AppConfig
from app.voice.engine import VoiceResult, normalize_transcript

logger = logging.getLogger(__name__)


class SpeechToTextEngine(ABC):
    """Abstract local speech-to-text backend."""

    @abstractmethod
    def check_availability(self) -> tuple[bool, str]:
        """Return whether the backend can run."""

    @abstractmethod
    def load(self) -> None:
        """Load models."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> VoiceResult:
        """Transcribe mono float32 PCM audio."""


class FasterWhisperEngine(SpeechToTextEngine):
    """Offline transcription using faster-whisper."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._model: Optional[object] = None

    def check_availability(self) -> tuple[bool, str]:
        import importlib.util

        if importlib.util.find_spec("faster_whisper") is None:
            return False, "faster-whisper not installed. Run: pip install faster-whisper"
        return True, f"faster-whisper ready ({self._config.voice_model}, loads on first use)"

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        logger.info("Loading faster-whisper model: %s", self._config.voice_model)
        self._model = WhisperModel(
            self._config.voice_model,
            device="auto",
            compute_type="int8",
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> VoiceResult:
        self.load()
        assert self._model is not None
        duration_s = len(audio) / sample_rate if sample_rate else 0.0
        if duration_s <= 0:
            return VoiceResult(text="", latency_ms=0.0, duration_s=0.0, error="No audio captured.")

        started = time.perf_counter()
        language = self._config.voice_language.strip() or None
        try:
            segments, _info = self._model.transcribe(
                audio,
                language=language,
                vad_filter=True,
            )
            text = normalize_transcript(" ".join(segment.text for segment in segments))
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            logger.exception("Voice transcription failed")
            return VoiceResult(
                text="",
                latency_ms=latency_ms,
                duration_s=duration_s,
                error=str(exc),
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        logger.info("Voice transcript ready in %.1fs: %r", latency_ms / 1000.0, text)
        return VoiceResult(text=text, latency_ms=latency_ms, duration_s=duration_s)


def create_stt_engine(config: AppConfig) -> SpeechToTextEngine:
    runtime = config.voice_runtime.strip().lower()
    if runtime == "whisper":
        return FasterWhisperEngine(config)
    raise ValueError(f"Unsupported VOICE_RUNTIME: {config.voice_runtime}")
