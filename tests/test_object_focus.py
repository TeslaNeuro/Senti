# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Tests for object-focused cropping."""

import numpy as np

from app.detection.detector import Detection
from app.vision.object_focus import (
    crop_detection,
    parse_track_id_from_question,
    select_focus_target,
    should_use_object_crop,
)


def _detection(
    class_name: str = "cup",
    track_id: int = 1,
    bbox: tuple[int, int, int, int] = (40, 40, 120, 120),
    confidence: float = 0.9,
) -> Detection:
    return Detection(0, class_name, confidence, bbox, track_id=track_id)


def test_should_use_object_crop_for_component_question() -> None:
    assert should_use_object_crop("What's that component?")


def test_parse_track_id_from_question() -> None:
    assert parse_track_id_from_question("Tell me about #3") == 3
    assert parse_track_id_from_question("track 2 details") == 2


def test_select_focus_target_by_combo_track_id() -> None:
    detections = [_detection(track_id=1), _detection("phone", track_id=2, bbox=(200, 200, 280, 280))]
    selection = select_focus_target(
        detections,
        focus_track_id=2,
        frame_width=640,
        frame_height=480,
    )
    assert selection is not None
    assert selection.detection.track_id == 2


def test_select_focus_target_auto_for_that_question() -> None:
    detections = [
        _detection("keyboard", track_id=1, bbox=(10, 10, 50, 50), confidence=0.6),
        _detection("cup", track_id=2, bbox=(250, 220, 390, 360), confidence=0.92),
    ]
    selection = select_focus_target(
        detections,
        question="What's that?",
        frame_width=640,
        frame_height=480,
    )
    assert selection is not None
    assert selection.detection.track_id == 2


def test_crop_detection_returns_padded_region() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[40:120, 40:120] = (0, 255, 0)
    crop = crop_detection(image, _detection(), padding_ratio=0.1)
    assert crop.shape[0] > 80
    assert crop.shape[1] > 80
    assert crop.shape[0] < 200
    assert crop.shape[1] < 200
