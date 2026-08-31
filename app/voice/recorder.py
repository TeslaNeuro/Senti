"""Microphone capture for push-to-talk voice input."""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class MicrophoneRecorder:
    """Records mono PCM audio until stopped or max duration is reached."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self._sample_rate = sample_rate
        self._stop_event = threading.Event()
        self._chunks: list[np.ndarray] = []
        self._stream = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def start(self) -> None:
        import sounddevice as sd

        self._stop_event.clear()
        self._chunks.clear()

        def callback(indata, frames, time, status) -> None:  # noqa: ANN001
            if status:
                logger.debug("Audio input status: %s", status)
            self._chunks.append(indata.copy())
            if self._stop_event.is_set():
                raise sd.CallbackStop()

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()
        logger.info("Microphone recording started")

    def stop(self) -> np.ndarray:
        import sounddevice as sd

        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._chunks:
            return np.array([], dtype=np.float32)
        audio = np.concatenate(self._chunks, axis=0).flatten()
        logger.info("Microphone recording stopped (%.2fs)", len(audio) / self._sample_rate)
        return audio

    @staticmethod
    def check_availability() -> tuple[bool, str]:
        try:
            import sounddevice  # noqa: F401
        except ImportError:
            return False, "sounddevice not installed. Run: pip install sounddevice"
        try:
            import sounddevice as sd

            if sd.query_devices(kind="input") is None:
                return False, "No microphone input device found."
        except Exception as exc:
            return False, f"Microphone unavailable: {exc}"
        return True, "Microphone ready"
