"""Speech synthesis package."""

from app.speech.engine import SpeechRequest, is_speak_command, prepare_speech_text
from app.speech.qt_tts import SpeechController

__all__ = [
    "SpeechController",
    "SpeechRequest",
    "is_speak_command",
    "prepare_speech_text",
]
