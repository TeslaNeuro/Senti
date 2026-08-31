# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Background voice capture and transcription thread."""

from __future__ import annotations

import logging
import queue
import threading
import time
from enum import Enum, auto
from typing import Optional

from PySide6.QtCore import QThread, Signal

from app.config import AppConfig
from app.voice.engine import VoiceResult, is_usable_transcript
from app.voice.recorder import MicrophoneRecorder
from app.voice.whisper_engine import create_stt_engine

logger = logging.getLogger(__name__)


class VoiceCommand(Enum):
    START = auto()
    STOP = auto()
    SHUTDOWN = auto()


class VoiceThread(QThread):
    """Captures microphone audio and runs local speech-to-text."""

    availability_checked = Signal(bool, str)
    listening_started = Signal()
    listening_stopped = Signal()
    transcription_completed = Signal(object)  # VoiceResult
    transcription_failed = Signal(str)

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._queue: queue.Queue[Optional[VoiceCommand]] = queue.Queue()
        self._recorder = MicrophoneRecorder(sample_rate=config.voice_sample_rate)
        self._engine = create_stt_engine(config)
        self._running = True
        self._listening = False
        self._listen_started_at = 0.0
        self._max_timer: Optional[threading.Timer] = None

    def stop(self) -> None:
        self._running = False
        self._queue.put(VoiceCommand.SHUTDOWN)
        self.wait(5000)

    def start_listening(self) -> None:
        if self._listening:
            return
        self._queue.put(VoiceCommand.START)

    def stop_listening(self) -> None:
        if not self._listening:
            return
        self._queue.put(VoiceCommand.STOP)

    def _cancel_max_timer(self) -> None:
        if self._max_timer is not None:
            self._max_timer.cancel()
            self._max_timer = None

    def _schedule_max_duration_stop(self) -> None:
        self._cancel_max_timer()
        self._max_timer = threading.Timer(self._config.voice_max_seconds, self.stop_listening)
        self._max_timer.daemon = True
        self._max_timer.start()

    def run(self) -> None:
        if not self._config.voice_enabled:
            self.availability_checked.emit(False, "Voice input disabled in config.")
            return

        mic_ok, mic_message = MicrophoneRecorder.check_availability()
        if not mic_ok:
            self.availability_checked.emit(False, mic_message)
            return

        try:
            engine_ok, engine_message = self._engine.check_availability()
            self.availability_checked.emit(engine_ok, engine_message)
            if not engine_ok:
                return
        except Exception as exc:
            self.availability_checked.emit(False, str(exc))
            logger.exception("Failed to initialize voice input")
            return

        while self._running:
            command = self._queue.get()
            if command is None or command is VoiceCommand.SHUTDOWN:
                break
            if command is VoiceCommand.START:
                self._begin_listening()
            elif command is VoiceCommand.STOP:
                self._finish_listening()

        self._cancel_max_timer()
        if self._listening:
            self._recorder.stop()
            self._listening = False

    def _begin_listening(self) -> None:
        if self._listening:
            return
        try:
            self._recorder.start()
        except Exception as exc:
            logger.exception("Failed to start microphone")
            self.transcription_failed.emit(f"Microphone error: {exc}")
            return
        self._listening = True
        self._listen_started_at = time.perf_counter()
        self._schedule_max_duration_stop()
        self.listening_started.emit()

    def _finish_listening(self) -> None:
        if not self._listening:
            return
        self._cancel_max_timer()
        self._listening = False
        self.listening_stopped.emit()
        audio = self._recorder.stop()
        duration_s = len(audio) / self._recorder.sample_rate
        if duration_s < self._config.voice_min_seconds:
            self.transcription_failed.emit("Recording too short. Hold the mic a little longer.")
            return
        try:
            self._engine.load()
        except Exception as exc:
            logger.exception("Failed to load speech model")
            self.transcription_failed.emit(f"Speech model error: {exc}")
            return
        result = self._engine.transcribe(audio, self._recorder.sample_rate)
        if not result.ok:
            self.transcription_failed.emit(result.error or "Could not transcribe audio.")
            return
        if not is_usable_transcript(result.text):
            self.transcription_failed.emit("No speech detected.")
            return
        self.transcription_completed.emit(result)
