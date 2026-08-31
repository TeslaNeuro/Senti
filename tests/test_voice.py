# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Tests for voice transcript normalization."""

from app.voice.engine import is_usable_transcript, normalize_transcript


def test_normalize_transcript() -> None:
    assert normalize_transcript("  what is this  ") == "What is this."
    assert normalize_transcript("hello world!") == "Hello world!"


def test_is_usable_transcript() -> None:
    assert is_usable_transcript("hi")
    assert not is_usable_transcript(" ")
