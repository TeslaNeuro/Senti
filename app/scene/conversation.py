# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Conversation history and question routing for visual follow-ups."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional

from app.detection.detector import Detection
from app.ocr.engine import OcrAnalysis, OcrResult, format_ocr_results
from app.perception.engine import AssistantState
from app.speech.engine import is_speak_command


class QuestionRoute(Enum):
    """How a user question should be answered."""

    CONTEXT_ONLY = auto()
    VLM_CACHED_FRAME = auto()
    VLM_FRESH_FRAME = auto()
    OCR_READ = auto()
    NO_SCENE = auto()


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class QuestionPlan:
    route: QuestionRoute
    answer: Optional[str] = None
    reason: str = ""
    speak_aloud: bool = False


_OBJECT_LIST_PATTERNS = (
    re.compile(r"what objects? do you see"),
    re.compile(r"what do you see"),
    re.compile(r"list (the )?objects"),
    re.compile(r"what(?:'s| is) in (the )?scene"),
)

_DESCRIBE_AGAIN_PATTERNS = (
    re.compile(r"^what am i looking at\??$"),
    re.compile(r"^what is this\??$"),
    re.compile(r"^what(?:'s| is) that\??$"),
    re.compile(r"^describe (this|the scene|it)\??$"),
)

_READ_TEXT_PATTERNS = (
    re.compile(r"^read this\??$"),
    re.compile(r"read (the )?text"),
    re.compile(r"what does (it|this) say"),
    re.compile(r"what text (is there|do you see)"),
)


def is_read_command(question: str) -> bool:
    lowered = question.strip().lower()
    return _matches_any(_READ_TEXT_PATTERNS, lowered)


def format_object_list(detections: list[Detection], description: str = "") -> str:
    if not detections and not description:
        return "I don't see any objects in the current scene yet."

    lines: list[str] = []
    if detections:
        lines.append("I can see:")
        for det in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if det.track_id is not None:
                lines.append(f"- #{det.track_id} {det.class_name} ({det.confidence:.0%})")
            else:
                lines.append(f"- {det.class_name} ({det.confidence:.0%})")
    if description:
        if lines:
            lines.append("")
        lines.append(f"Last description: {description}")
    return "\n".join(lines)


def build_conversation_context(turns: list[ConversationTurn], max_turns: int = 6) -> Optional[str]:
    if not turns:
        return None
    recent = turns[-max_turns:]
    lines: list[str] = ["Recent conversation:"]
    for turn in recent:
        speaker = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{speaker}: {turn.content}")
    return "\n".join(lines)


def _matches_any(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


class QuestionRouter:
    """Decide whether to answer from memory or call the VLM."""

    def __init__(self, stale_frame_gap: int = 45) -> None:
        self._stale_frame_gap = stale_frame_gap

    def route(
        self,
        question: str,
        *,
        has_description: bool,
        detections: list[Detection],
        analyzed_frame_id: int,
        latest_frame_id: int,
        has_cached_frame: bool,
        assistant_state: AssistantState,
        last_description: str = "",
        ocr_results: Optional[list[OcrResult]] = None,
        ocr_enabled: bool = False,
    ) -> QuestionPlan:
        cleaned = question.strip()
        if not cleaned:
            return QuestionPlan(QuestionRoute.NO_SCENE, reason="Empty question.")

        lowered = cleaned.lower()

        if is_read_command(cleaned):
            if ocr_results:
                return QuestionPlan(
                    QuestionRoute.CONTEXT_ONLY,
                    answer=format_ocr_results(ocr_results),
                    reason="Returning cached OCR results.",
                )
            if ocr_enabled:
                return QuestionPlan(
                    QuestionRoute.OCR_READ,
                    reason="Read command requires OCR.",
                )
            return QuestionPlan(
                QuestionRoute.NO_SCENE,
                reason="OCR is disabled. Set OCR_ENABLED=true to read text.",
            )

        if is_speak_command(cleaned):
            if last_description:
                return QuestionPlan(
                    QuestionRoute.CONTEXT_ONLY,
                    answer=last_description,
                    reason="Repeating the current scene description aloud.",
                    speak_aloud=True,
                )
            return QuestionPlan(
                QuestionRoute.NO_SCENE,
                reason="Nothing to speak yet. Wait for a scene description.",
            )

        if not has_description and not detections:
            return QuestionPlan(
                QuestionRoute.NO_SCENE,
                reason="No scene understanding yet. Wait for auto-analyze or click Analyze.",
            )

        if _matches_any(_OBJECT_LIST_PATTERNS, lowered) and detections:
            return QuestionPlan(
                QuestionRoute.CONTEXT_ONLY,
                answer=format_object_list(detections, last_description),
                reason="Object list available from detections.",
            )

        scene_busy = assistant_state in {
            AssistantState.SCENE_CHANGED,
            AssistantState.WAITING_FOR_STABILITY,
        }
        frame_gap = latest_frame_id - analyzed_frame_id
        is_stale = analyzed_frame_id < 0 or frame_gap > self._stale_frame_gap

        if scene_busy or is_stale or not has_cached_frame:
            return QuestionPlan(
                QuestionRoute.VLM_FRESH_FRAME,
                reason="Scene changed or frame is stale; using the latest frame.",
            )

        if _matches_any(_DESCRIBE_AGAIN_PATTERNS, lowered) and last_description:
            return QuestionPlan(
                QuestionRoute.CONTEXT_ONLY,
                answer=last_description,
                reason="Reusing the current scene description.",
            )

        return QuestionPlan(
            QuestionRoute.VLM_CACHED_FRAME,
            reason="Follow-up question against the current analyzed frame.",
        )
