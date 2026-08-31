# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""macOS `say` command helpers for voices Qt does not expose."""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional


def say_available() -> bool:
    return shutil.which("say") is not None


def parse_macos_voice_list(output: str) -> dict[str, str]:
    """Parse `say -v '?'` output into lowercase name -> canonical name."""
    voices: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split()
        if parts:
            voices[parts[0].lower()] = parts[0]
    return voices


def list_macos_voices() -> dict[str, str]:
    if not say_available():
        return {}
    result = subprocess.run(
        ["say", "-v", "?"],
        capture_output=True,
        text=True,
        check=False,
    )
    return parse_macos_voice_list(result.stdout)


def find_macos_voice(query: str) -> Optional[str]:
    if not query.strip():
        return None
    voices = list_macos_voices()
    needle = query.strip().lower()
    if needle in voices:
        return voices[needle]
    for key, canonical in voices.items():
        if needle in key:
            return canonical
    return None


def map_rate_to_say_wpm(rate: float) -> int:
    """Map Qt-style rate (-1..1) to `say -r` words per minute."""
    return int(max(80, min(400, 200 + rate * 80)))
