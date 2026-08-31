# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Tests for speech text preparation and speak commands."""

from app.speech.engine import is_speak_command, prepare_speech_text


def test_prepare_speech_text_strips_markdown() -> None:
    text = prepare_speech_text("**Hello** world\n\nwith *emphasis*")
    assert text == "Hello world with emphasis"


def test_prepare_speech_text_truncates_long_text() -> None:
    text = prepare_speech_text("word " * 500, max_chars=40)
    assert len(text) <= 41
    assert text.endswith(".")


def test_is_speak_command() -> None:
    assert is_speak_command("say that")
    assert is_speak_command("read aloud")
    assert not is_speak_command("what is this")
