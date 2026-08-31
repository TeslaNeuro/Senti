# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Scene change detection using visual, detection, and spatial signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.detection.detector import Detection
from app.tracking.tracker import TrackUpdate


@dataclass(frozen=True)
class SceneChangeAnalysis:
    """Result of one scene-change evaluation pass."""

    scene_changed: bool
    rising_edge: bool
    visual_score: float
    detection_score: float
    spatial_score: float
    combined_score: float
    movement_score: float
    reasons: tuple[str, ...] = ()


class SceneChangeDetector:
    """Detect meaningful scene changes from multiple complementary signals."""

    def __init__(self, threshold: float, debounce_frames: int = 2) -> None:
        self._threshold = threshold
        self._debounce_frames = debounce_frames
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_track_ids: set[int] = set()
        self._prev_class_names: set[str] = set()
        self._prev_centers: dict[int, tuple[float, float]] = {}
        self._hot_count = 0
        self._was_scene_changed = False

    def analyze(
        self,
        frame_bgr: np.ndarray,
        detections: list[Detection],
        track_update: Optional[TrackUpdate],
    ) -> SceneChangeAnalysis:
        visual_score = self._visual_change(frame_bgr)
        detection_score, detection_reasons = self._detection_change(detections, track_update)
        spatial_score, movement_score = self._spatial_change(detections, frame_bgr.shape)

        combined_score = max(visual_score, detection_score, spatial_score)
        reasons: list[str] = []
        if visual_score >= self._threshold:
            reasons.append("visual")
        reasons.extend(detection_reasons)
        if spatial_score >= self._threshold:
            reasons.append("spatial")

        if combined_score >= self._threshold:
            self._hot_count += 1
        else:
            self._hot_count = 0

        scene_changed = self._hot_count >= self._debounce_frames
        rising_edge = scene_changed and not self._was_scene_changed
        self._was_scene_changed = scene_changed

        return SceneChangeAnalysis(
            scene_changed=scene_changed,
            rising_edge=rising_edge,
            visual_score=visual_score,
            detection_score=detection_score,
            spatial_score=spatial_score,
            combined_score=combined_score,
            movement_score=movement_score,
            reasons=tuple(reasons),
        )

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_track_ids.clear()
        self._prev_class_names.clear()
        self._prev_centers.clear()
        self._hot_count = 0
        self._was_scene_changed = False

    def _visual_change(self, frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        if self._prev_gray is None:
            self._prev_gray = small
            return 0.0

        diff = cv2.absdiff(small, self._prev_gray)
        score = float(np.mean(diff) / 255.0)
        self._prev_gray = small
        return score

    def _detection_change(
        self,
        detections: list[Detection],
        track_update: Optional[TrackUpdate],
    ) -> tuple[float, list[str]]:
        reasons: list[str] = []
        score = 0.0

        current_classes = {det.class_name for det in detections}
        if self._prev_class_names and current_classes != self._prev_class_names:
            score = max(score, 0.5)
            reasons.append("classes")

        prev_count = len(self._prev_class_names) if self._prev_class_names else len(current_classes)
        count_delta = abs(len(detections) - prev_count)
        if count_delta >= 2:
            score = max(score, 0.6)
            reasons.append("object_count")

        if track_update is not None:
            if track_update.new_track_ids:
                score = max(score, 0.7)
                reasons.append("new_object")
            if track_update.lost_track_ids:
                score = max(score, 0.7)
                reasons.append("object_lost")

        self._prev_class_names = current_classes
        if track_update is not None:
            self._prev_track_ids = set(track_update.active_track_ids)
        return score, reasons

    def _spatial_change(
        self,
        detections: list[Detection],
        frame_shape: tuple,
    ) -> tuple[float, float]:
        height, width = frame_shape[:2]
        frame_diag = float(np.hypot(width, height)) or 1.0
        max_shift = 0.0
        current_centers: dict[int, tuple[float, float]] = {}

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            key = det.track_id if det.track_id is not None else det.class_id
            current_centers[key] = (cx, cy)
            if key in self._prev_centers:
                px, py = self._prev_centers[key]
                shift = float(np.hypot(cx - px, cy - py) / frame_diag)
                max_shift = max(max_shift, shift)

        self._prev_centers = current_centers
        spatial_score = min(1.0, max_shift * 4.0)
        return spatial_score, max_shift
