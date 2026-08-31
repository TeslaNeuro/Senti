# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Tests for conversational question routing."""

from app.detection.detector import Detection
from app.perception.engine import AssistantState
from app.scene.conversation import QuestionRoute, QuestionRouter


def _detection(class_name: str = "cup", track_id: int = 1) -> Detection:
    return Detection(0, class_name, 0.9, (1, 2, 3, 4), track_id=track_id)


def test_router_lists_objects_from_context() -> None:
    router = QuestionRouter()
    plan = router.route(
        "What objects do you see?",
        has_description=True,
        detections=[_detection()],
        analyzed_frame_id=10,
        latest_frame_id=12,
        has_cached_frame=True,
        assistant_state=AssistantState.READY,
        last_description="A mug on a desk.",
    )
    assert plan.route == QuestionRoute.CONTEXT_ONLY
    assert plan.answer is not None
    assert "cup" in plan.answer


def test_router_uses_cached_frame_for_follow_up() -> None:
    router = QuestionRouter()
    plan = router.route(
        "What's that connector?",
        has_description=True,
        detections=[_detection("keyboard")],
        analyzed_frame_id=20,
        latest_frame_id=22,
        has_cached_frame=True,
        assistant_state=AssistantState.READY,
        last_description="A keyboard with a USB port.",
    )
    assert plan.route == QuestionRoute.VLM_CACHED_FRAME


def test_router_requests_fresh_frame_when_scene_changed() -> None:
    router = QuestionRouter()
    plan = router.route(
        "What is this?",
        has_description=True,
        detections=[_detection()],
        analyzed_frame_id=20,
        latest_frame_id=22,
        has_cached_frame=True,
        assistant_state=AssistantState.SCENE_CHANGED,
        last_description="Old scene.",
    )
    assert plan.route == QuestionRoute.VLM_FRESH_FRAME


def test_router_reuses_description_for_repeat_question() -> None:
    router = QuestionRouter()
    plan = router.route(
        "What am I looking at?",
        has_description=True,
        detections=[_detection()],
        analyzed_frame_id=20,
        latest_frame_id=22,
        has_cached_frame=True,
        assistant_state=AssistantState.READY,
        last_description="A black ballpoint pen.",
    )
    assert plan.route == QuestionRoute.CONTEXT_ONLY
    assert plan.answer == "A black ballpoint pen."


def test_router_requires_scene_context() -> None:
    router = QuestionRouter()
    plan = router.route(
        "What is this?",
        has_description=False,
        detections=[],
        analyzed_frame_id=-1,
        latest_frame_id=5,
        has_cached_frame=False,
        assistant_state=AssistantState.WATCHING,
    )
    assert plan.route == QuestionRoute.NO_SCENE


def test_router_speak_command_repeats_description() -> None:
    router = QuestionRouter()
    plan = router.route(
        "say that",
        has_description=True,
        detections=[_detection()],
        analyzed_frame_id=20,
        latest_frame_id=22,
        has_cached_frame=True,
        assistant_state=AssistantState.READY,
        last_description="A black ballpoint pen.",
    )
    assert plan.route == QuestionRoute.CONTEXT_ONLY
    assert plan.speak_aloud is True
    assert plan.answer == "A black ballpoint pen."
