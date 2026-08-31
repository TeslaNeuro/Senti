# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Voice input types and transcript normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass

_MIN_TRANSCRIPT_CHARS = 2


def normalize_transcript(text: str) -> str:
    """Clean up speech-to-text output before routing to the assistant."""
    cleaned = text.strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" .")
    if not cleaned:
        return ""
    if cleaned[-1] not in ".?!":
        cleaned = f"{cleaned}."
    return cleaned[0].upper() + cleaned[1:] if cleaned else ""


def is_usable_transcript(text: str, *, min_chars: int = _MIN_TRANSCRIPT_CHARS) -> bool:
    return len(text.strip()) >= min_chars


@dataclass(frozen=True)
class VoiceResult:
    """Speech-to-text output for one recording."""

    text: str
    latency_ms: float
    duration_s: float
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text.strip())
