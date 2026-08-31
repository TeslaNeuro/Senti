# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Tests for UI helper utilities."""

from app.ui.helpers import short_status_detail


def test_short_status_detail_truncates() -> None:
    detail = short_status_detail("EasyOCR not installed. Run pip install easyocr")
    assert detail.endswith("…)")
    assert len(detail) <= 25


def test_short_status_detail_short_message() -> None:
    assert short_status_detail("ready") == "(ready)"
