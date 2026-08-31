# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Local text-to-speech: Qt voices with a macOS `say` fallback."""

from app.speech.engine import SpeechRequest, is_speak_command, prepare_speech_text
from app.speech.qt_tts import SpeechController

__all__ = [
    "SpeechController",
    "SpeechRequest",
    "is_speak_command",
    "prepare_speech_text",
]
