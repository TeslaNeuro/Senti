# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Tests for best-frame selection."""

import numpy as np

from app.camera.frame_buffer import CameraFrame
from app.detection.detector import Detection, DetectionResult
from app.perception.frame_selector import FrameSelector, score_frame


def _camera_frame(value: int, frame_id: int = 0) -> CameraFrame:
    data = np.full((100, 100, 3), value, dtype=np.uint8)
    return CameraFrame(data=data, timestamp=0.0, frame_id=frame_id, width=100, height=100)


def _result(detections: list[Detection] | None = None) -> DetectionResult:
    return DetectionResult(detections=detections or [], frame_id=0)


def test_sharp_frame_scores_higher_than_blurry() -> None:
    sharp = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
    blurry = np.full((80, 80, 3), 128, dtype=np.uint8)
    sharp_score = score_frame(sharp, [], movement_score=0.0).sharpness
    blurry_score = score_frame(blurry, [], movement_score=0.0).sharpness
    assert sharp_score > blurry_score


def test_stable_frame_scores_higher_than_moving() -> None:
    frame = np.full((80, 80, 3), 100, dtype=np.uint8)
    stable = score_frame(frame, [], movement_score=0.01).stability
    moving = score_frame(frame, [], movement_score=0.5).stability
    assert stable > moving


def test_selector_picks_highest_scoring_candidate() -> None:
    selector = FrameSelector(max_size=5)
    blurry = _camera_frame(50, frame_id=1)
    sharp_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    sharp = CameraFrame(
        data=sharp_data, timestamp=0.0, frame_id=2, width=100, height=100
    )

    selector.add(blurry, _result(), movement_score=0.4)
    selector.add(sharp, _result(), movement_score=0.01)

    selected = selector.select_best()
    assert selected is not None
    assert selected.candidate.camera_frame.frame_id == 2


def test_selector_respects_max_buffer_size() -> None:
    selector = FrameSelector(max_size=3)
    for idx in range(5):
        selector.add(_camera_frame(idx, frame_id=idx), _result(), movement_score=0.0)
    assert selector.size == 3


def test_centered_object_scores_higher() -> None:
    frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    centered = Detection(0, "cup", 0.9, (35, 35, 65, 65))
    edge = Detection(0, "cup", 0.9, (2, 2, 20, 20))
    centered_score = score_frame(frame, [centered], 0.0).position
    edge_score = score_frame(frame, [edge], 0.0).position
    assert centered_score > edge_score
