# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Object-focused cropping and focus target selection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.detection.detector import Detection

_OBJECT_FOCUS_PATTERNS = (
    re.compile(r"what(?:'s| is) that (component|part|object|thing)\b"),
    re.compile(r"what is this (component|part|object)\b"),
    re.compile(r"describe (this|that) (component|part|object)\b"),
    re.compile(r"what does (this|that) (component|part|object) do\b"),
    re.compile(r"what(?:'s| is) that\??$"),
    re.compile(r"focus on\b"),
    re.compile(r"zoom in on\b"),
    re.compile(r"that (component|part)\b"),
    re.compile(r"this (component|part)\b"),
)

_TRACK_ID_PATTERNS = (
    re.compile(r"#(\d+)\b"),
    re.compile(r"\btrack(?:ed)?\s*(?:id\s*)?#?(\d+)\b"),
    re.compile(r"\bobject\s*#?(\d+)\b"),
)


@dataclass(frozen=True)
class FocusSelection:
    detection: Detection
    reason: str


def should_use_object_crop(question: Optional[str]) -> bool:
    if not question:
        return False
    lowered = question.strip().lower()
    return any(pattern.search(lowered) for pattern in _OBJECT_FOCUS_PATTERNS)


def parse_track_id_from_question(question: str) -> Optional[int]:
    lowered = question.lower()
    for pattern in _TRACK_ID_PATTERNS:
        match = pattern.search(lowered)
        if match:
            return int(match.group(1))
    return None


def _bbox_area(detection: Detection) -> int:
    x1, y1, x2, y2 = detection.bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _distance_to_center(detection: Detection, frame_width: int, frame_height: int) -> float:
    x1, y1, x2, y2 = detection.bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    frame_cx = frame_width / 2.0
    frame_cy = frame_height / 2.0
    return ((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2) ** 0.5


def _find_by_track_id(detections: list[Detection], track_id: int) -> Optional[Detection]:
    for detection in detections:
        if detection.track_id == track_id:
            return detection
    return None


def _find_by_class_name(question: str, detections: list[Detection]) -> Optional[Detection]:
    lowered = question.lower()
    matches: list[Detection] = []
    for detection in detections:
        name = detection.class_name.lower().replace("_", " ")
        if name in lowered or name.replace(" ", "") in lowered.replace(" ", ""):
            matches.append(detection)
    if not matches:
        return None
    return max(matches, key=lambda det: det.confidence)


def select_focus_target(
    detections: list[Detection],
    *,
    question: Optional[str] = None,
    focus_track_id: Optional[int] = None,
    frame_width: int,
    frame_height: int,
    allow_auto: bool = True,
) -> Optional[FocusSelection]:
    """Pick the most likely object for a focused VLM crop."""
    if not detections:
        return None

    if focus_track_id is not None:
        match = _find_by_track_id(detections, focus_track_id)
        if match is not None:
            return FocusSelection(match, reason=f"User selected track #{focus_track_id}.")

    if question:
        track_id = parse_track_id_from_question(question)
        if track_id is not None:
            match = _find_by_track_id(detections, track_id)
            if match is not None:
                return FocusSelection(match, reason=f"Question referenced track #{track_id}.")

        class_match = _find_by_class_name(question, detections)
        if class_match is not None and should_use_object_crop(question):
            return FocusSelection(class_match, reason=f"Question referenced {class_match.class_name}.")

    if not allow_auto and focus_track_id is None:
        return None

    wants_crop = should_use_object_crop(question) or focus_track_id is not None
    if not wants_crop:
        return None

    # Prefer prominent, centered, confident objects.
    ranked = sorted(
        detections,
        key=lambda det: (
            det.confidence,
            _bbox_area(det),
            -_distance_to_center(det, frame_width, frame_height),
        ),
        reverse=True,
    )
    chosen = ranked[0]
    if chosen.track_id is not None:
        reason = f"Auto-focused track #{chosen.track_id} ({chosen.class_name})."
    else:
        reason = f"Auto-focused {chosen.class_name}."
    return FocusSelection(chosen, reason=reason)


def crop_detection(
    image_bgr: np.ndarray,
    detection: Detection,
    *,
    padding_ratio: float = 0.15,
    min_size: int = 32,
) -> np.ndarray:
    """Crop a detection bbox from a frame with proportional padding."""
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = detection.bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(width, x2 + pad_x)
    crop_y2 = min(height, y2 + pad_y)

    if crop_x2 - crop_x1 < min_size or crop_y2 - crop_y1 < min_size:
        return image_bgr.copy()

    return image_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()
