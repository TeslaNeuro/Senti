# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Tests for scene context manager."""

from app.detection.detector import Detection
from app.scene.scene_state import SceneContextManager
from app.vision.vision_model import VisionAnalysisResult
import numpy as np


def test_scene_context_keeps_conversation_and_frame() -> None:
    manager = SceneContextManager(max_turns=4)
    manager.add_user_turn("What am I looking at?")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = VisionAnalysisResult(
        description="A green rectangle.",
        latency_ms=10.0,
        model="test",
        frame_id=7,
    )
    detections = [Detection(0, "book", 0.8, (0, 0, 1, 1), track_id=2)]

    manager.apply_analysis(result, detections, image_bgr=image)

    assert manager.scene.description == "A green rectangle."
    assert manager.scene.frame_id == 7
    assert manager.scene.analyzed_frame_bgr is not None
    assert manager.scene.primary_object is not None
    assert len(manager.turns) == 2
    context = manager.previous_context()
    assert context is not None
    assert "Recent conversation:" in context
    assert "green rectangle" in context


def test_reset_scene_clears_memory() -> None:
    manager = SceneContextManager()
    manager.add_user_turn("Hello")
    manager.scene.description = "Something"
    manager.reset_scene()
    assert manager.scene.description == ""
    assert manager.turns == []
