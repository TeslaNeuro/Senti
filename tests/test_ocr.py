# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Tests for OCR formatting and read-command routing."""

from app.ocr.engine import OcrResult, format_ocr_results
from app.perception.engine import AssistantState
from app.scene.conversation import QuestionRoute, QuestionRouter, is_read_command


def test_format_ocr_results() -> None:
    text = format_ocr_results(
        [
            OcrResult("STM32F401", 0.94, (1, 2, 3, 4)),
            OcrResult("3.3V", 0.88, (5, 6, 7, 8)),
        ]
    )
    assert "STM32F401" in text
    assert "3.3V" in text


def test_is_read_command() -> None:
    assert is_read_command("read this")
    assert is_read_command("What text do you see?")


def test_router_returns_cached_ocr() -> None:
    router = QuestionRouter()
    plan = router.route(
        "read this",
        has_description=False,
        detections=[],
        analyzed_frame_id=1,
        latest_frame_id=2,
        has_cached_frame=False,
        assistant_state=AssistantState.READY,
        ocr_results=[OcrResult("HELLO", 0.9, (0, 0, 1, 1))],
        ocr_enabled=True,
    )
    assert plan.route == QuestionRoute.CONTEXT_ONLY
    assert plan.answer is not None
    assert "HELLO" in plan.answer


def test_router_requests_ocr_when_enabled() -> None:
    router = QuestionRouter()
    plan = router.route(
        "read this",
        has_description=True,
        detections=[],
        analyzed_frame_id=1,
        latest_frame_id=2,
        has_cached_frame=True,
        assistant_state=AssistantState.READY,
        ocr_results=[],
        ocr_enabled=True,
    )
    assert plan.route == QuestionRoute.OCR_READ
