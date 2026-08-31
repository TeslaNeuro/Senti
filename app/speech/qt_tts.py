"""macOS speech synthesis via Qt TextToSpeech."""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Literal, Optional

from PySide6.QtCore import QObject, Slot
from PySide6.QtTextToSpeech import QTextToSpeech

from app.config import AppConfig
from app.speech.engine import prepare_speech_text
from app.speech.macos_say import find_macos_voice, map_rate_to_say_wpm, say_available

logger = logging.getLogger(__name__)

Backend = Literal["qt", "say"]


class SpeechController(QObject):
    """Speaks assistant responses using the native macOS engine."""

    def __init__(self, config: AppConfig, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._tts: Optional[QTextToSpeech] = None
        self._backend: Backend = "qt"
        self._say_voice: Optional[str] = None
        self._say_proc: Optional[subprocess.Popen[bytes]] = None
        self._available = False
        self._message = ""
        self._last_spoken = ""

        if not config.tts_enabled:
            return

        engine = self._select_engine()
        self._tts = QTextToSpeech(engine, self) if engine else QTextToSpeech(self)
        self._tts.stateChanged.connect(self._on_state_changed)
        self._tts.errorOccurred.connect(self._on_error)
        self._select_backend()
        available, message = self.check_availability()
        self._available = available
        self._message = message

    @staticmethod
    def _select_engine() -> Optional[str]:
        engines = QTextToSpeech.availableEngines()
        if "darwin" in engines:
            return "darwin"
        if engines:
            return engines[0]
        return None

    def _select_backend(self) -> None:
        if self._tts is None:
            return

        runtime = self._config.tts_runtime.strip().lower()
        self._tts.setRate(self._config.tts_rate)
        self._tts.setVolume(self._config.tts_volume)

        if runtime == "say":
            self._use_say_backend(self._config.tts_voice or None)
            return

        if not self._config.tts_voice:
            self._backend = "qt"
            return

        if self._try_configure_qt_voice(self._config.tts_voice):
            self._backend = "qt"
            return

        mac_voice = find_macos_voice(self._config.tts_voice)
        if mac_voice and say_available():
            self._use_say_backend(mac_voice)
            logger.info(
                "Voice %r is not exposed by Qt; using macOS say backend instead.",
                self._config.tts_voice,
            )
            return

        logger.warning(
            "TTS voice %r not found in Qt or macOS say voices; using Qt default.",
            self._config.tts_voice,
        )
        self._backend = "qt"

    def _try_configure_qt_voice(self, query: str) -> bool:
        if self._tts is None:
            return False
        needle = query.lower()
        for voice in self._tts.availableVoices():
            if needle in voice.name().lower():
                self._tts.setVoice(voice)
                logger.info("TTS voice (Qt): %s", voice.name())
                return True
        return False

    def _use_say_backend(self, voice: Optional[str]) -> None:
        if not say_available():
            self._backend = "qt"
            return
        self._backend = "say"
        self._say_voice = voice

    def check_availability(self) -> tuple[bool, str]:
        if not self._config.tts_enabled:
            return False, "TTS disabled in config."
        if self._backend == "say":
            if not say_available():
                return False, "macOS say command unavailable."
            if self._say_voice:
                return True, f"macOS say ready ({self._say_voice})"
            return True, "macOS say ready"
        if self._tts is None:
            return False, "Speech engine unavailable."
        if sys.platform != "darwin" and self._tts.engine() != "darwin":
            return True, f"Using {self._tts.engine()} speech engine"
        return True, "macOS speech ready"

    @property
    def available(self) -> bool:
        return self._available

    @property
    def message(self) -> str:
        return self._message

    @property
    def last_spoken(self) -> str:
        return self._last_spoken

    @property
    def is_speaking(self) -> bool:
        if self._backend == "say":
            return self._say_proc is not None and self._say_proc.poll() is None
        return self._tts is not None and self._tts.state() == QTextToSpeech.State.Speaking

    def speak(self, text: str, *, interrupt: Optional[bool] = None) -> bool:
        if not self._available:
            return False

        prepared = prepare_speech_text(text)
        if not prepared:
            return False

        should_interrupt = self._config.tts_interrupt if interrupt is None else interrupt
        if should_interrupt and self.is_speaking:
            self.stop()

        self._last_spoken = prepared
        if self._backend == "say":
            return self._speak_with_say(prepared)
        return self._speak_with_qt(prepared)

    def _speak_with_qt(self, prepared: str) -> bool:
        if self._tts is None:
            return False
        self._tts.say(prepared)
        logger.info("Speaking %d characters via Qt", len(prepared))
        return True

    def _speak_with_say(self, prepared: str) -> bool:
        command = ["say", "-r", str(map_rate_to_say_wpm(self._config.tts_rate))]
        if self._say_voice:
            command.extend(["-v", self._say_voice])
        command.append(prepared)
        try:
            self._say_proc = subprocess.Popen(command)
        except OSError as exc:
            logger.warning("Failed to start say: %s", exc)
            return False
        logger.info(
            "Speaking %d characters via say%s",
            len(prepared),
            f" ({self._say_voice})" if self._say_voice else "",
        )
        return True

    def stop(self) -> None:
        if self._backend == "say" and self._say_proc is not None and self._say_proc.poll() is None:
            self._say_proc.terminate()
            self._say_proc = None
            return
        if self._tts is not None:
            self._tts.stop()

    @Slot(QTextToSpeech.State)
    def _on_state_changed(self, state: QTextToSpeech.State) -> None:
        if state == QTextToSpeech.State.Ready:
            logger.debug("TTS ready")

    @Slot(QTextToSpeech.ErrorReason, str)
    def _on_error(self, reason: QTextToSpeech.ErrorReason, message: str) -> None:
        logger.warning("TTS error (%s): %s", reason, message)
