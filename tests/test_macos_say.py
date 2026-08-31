# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Tests for macOS say voice helpers."""

from app.speech.macos_say import find_macos_voice, map_rate_to_say_wpm, parse_macos_voice_list


def test_parse_macos_voice_list() -> None:
    voices = parse_macos_voice_list("Tessa               en_ZA    # Hi\nDaniel en_GB\n")
    assert voices["tessa"] == "Tessa"
    assert voices["daniel"] == "Daniel"


def test_find_macos_voice_from_parsed_list(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.speech.macos_say.list_macos_voices",
        lambda: {"tessa": "Tessa", "daniel": "Daniel"},
    )
    assert find_macos_voice("Tessa") == "Tessa"
    assert find_macos_voice("dan") == "Daniel"


def test_map_rate_to_say_wpm() -> None:
    assert map_rate_to_say_wpm(0.0) == 200
    assert map_rate_to_say_wpm(1.0) == 280
