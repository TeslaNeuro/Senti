# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Rolling buffer and best-frame selection for deep analysis."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.camera.frame_buffer import CameraFrame
from app.detection.detector import Detection, DetectionResult


@dataclass(frozen=True)
class FrameScoreBreakdown:
    """Per-factor scores used to rank candidate frames (0–1 each)."""

    sharpness: float
    stability: float
    object_size: float
    position: float
    confidence: float
    lighting: float
    total: float


@dataclass(frozen=True)
class FrameCandidate:
    """A frame paired with its detections and quality scores."""

    camera_frame: CameraFrame
    result: DetectionResult
    movement_score: float
    score: FrameScoreBreakdown


@dataclass(frozen=True)
class SelectedFrame:
    """The highest-quality frame chosen from recent candidates."""

    candidate: FrameCandidate


class FrameSelector:
    """Maintains a rolling buffer and selects the best frame for analysis."""

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self._max_size = max_size
        self._candidates: deque[FrameCandidate] = deque(maxlen=max_size)
        self._last_selected: Optional[SelectedFrame] = None

    @property
    def size(self) -> int:
        return len(self._candidates)

    @property
    def last_selected(self) -> Optional[SelectedFrame]:
        return self._last_selected

    def reset(self) -> None:
        """Clear candidates when the scene meaningfully changes."""
        self._candidates.clear()
        self._last_selected = None

    def add(
        self,
        camera_frame: CameraFrame,
        result: DetectionResult,
        movement_score: float,
    ) -> None:
        score = score_frame(camera_frame.data, result.detections, movement_score)
        self._candidates.append(
            FrameCandidate(
                camera_frame=camera_frame,
                result=result,
                movement_score=movement_score,
                score=score,
            )
        )

    def select_best(self) -> Optional[SelectedFrame]:
        """Pick the highest-scoring frame from the rolling buffer."""
        if not self._candidates:
            return None
        best = max(self._candidates, key=lambda item: item.score.total)
        selected = SelectedFrame(candidate=best)
        self._last_selected = selected
        return selected


def score_frame(
    frame_bgr: np.ndarray,
    detections: list[Detection],
    movement_score: float,
) -> FrameScoreBreakdown:
    """Score a frame for sharpness, stability, object prominence, and lighting."""
    height, width = frame_bgr.shape[:2]
    sharpness = _sharpness_score(frame_bgr)
    stability = _stability_score(movement_score)
    object_size, position, confidence = _object_scores(detections, width, height)
    lighting = _lighting_score(frame_bgr)

    total = (
        0.30 * sharpness
        + 0.20 * stability
        + 0.15 * object_size
        + 0.10 * position
        + 0.15 * confidence
        + 0.10 * lighting
    )
    return FrameScoreBreakdown(
        sharpness=sharpness,
        stability=stability,
        object_size=object_size,
        position=position,
        confidence=confidence,
        lighting=lighting,
        total=total,
    )


def _sharpness_score(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return min(1.0, variance / 500.0)


def _stability_score(movement_score: float) -> float:
    return max(0.0, 1.0 - min(1.0, movement_score * 5.0))


def _object_scores(
    detections: list[Detection],
    width: int,
    height: int,
) -> tuple[float, float, float]:
    if not detections:
        return 0.0, 0.0, 0.0

    frame_area = float(width * height) or 1.0
    center_x = width / 2.0
    center_y = height / 2.0
    frame_diag = float(np.hypot(width, height)) or 1.0

    best_size = 0.0
    best_position = 0.0
    best_confidence = 0.0

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        area = max(0, x2 - x1) * max(0, y2 - y1) / frame_area
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = float(np.hypot(cx - center_x, cy - center_y) / frame_diag)
        position = max(0.0, 1.0 - dist * 2.0)
        best_size = max(best_size, min(1.0, area * 4.0))
        best_position = max(best_position, position)
        best_confidence = max(best_confidence, det.confidence)

    return best_size, best_position, best_confidence


def _lighting_score(frame_bgr: np.ndarray) -> float:
    mean = float(np.mean(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))) / 255.0
    return max(0.0, 1.0 - min(1.0, abs(mean - 0.5) * 2.0))
