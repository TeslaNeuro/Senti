"""Speech types and text preparation for local TTS."""

from __future__ import annotations

import re
from dataclasses import dataclass

_MAX_SPEECH_CHARS = 2000

_SPEAK_COMMAND_PATTERNS = (
    re.compile(r"^(say|speak) (that|this|it)\??$"),
    re.compile(r"^read (that|it)\??$"),
    re.compile(r"^read (it )?aloud\??$"),
    re.compile(r"^repeat (that|it)\??$"),
    re.compile(r"^say (it )?again\??$"),
)


def is_speak_command(question: str) -> bool:
    lowered = question.strip().lower()
    return any(pattern.search(lowered) for pattern in _SPEAK_COMMAND_PATTERNS)


def prepare_speech_text(text: str, *, max_chars: int = _MAX_SPEECH_CHARS) -> str:
    """Normalize assistant text for macOS speech synthesis."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    cleaned = re.sub(r"[#*_`]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace("…", ".")
    if len(cleaned) > max_chars:
        trimmed = cleaned[:max_chars].rsplit(" ", 1)[0]
        cleaned = f"{trimmed}."
    return cleaned


@dataclass(frozen=True)
class SpeechRequest:
    """A single speak request."""

    text: str
    interrupt: bool = True
