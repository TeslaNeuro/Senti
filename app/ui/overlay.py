# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Draw detection overlays on camera frames."""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

from app.detection.detector import Detection

_TEXT_COLOR = (255, 255, 255)


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Stable BGR color per track ID."""
    palette = (
        (0, 220, 120),
        (255, 170, 40),
        (80, 180, 255),
        (220, 80, 200),
        (120, 220, 255),
        (180, 255, 80),
    )
    return palette[track_id % len(palette)]


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list[Detection],
    focus_track_id: Optional[int] = None,
) -> np.ndarray:
    """Return a copy of the frame with bounding boxes and labels drawn."""
    if not detections:
        return frame_bgr

    output = frame_bgr.copy()
    height, width = output.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        box_color = _track_color(detection.track_id) if detection.track_id is not None else (0, 220, 120)
        thickness = 4 if detection.track_id is not None and detection.track_id == focus_track_id else 2
        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, thickness)

        if detection.track_id is not None:
            label = f"#{detection.track_id} {detection.class_name} {detection.confidence:.0%}"
        else:
            label = f"{detection.class_name} {detection.confidence:.0%}"

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        text_y = max(y1, text_h + 6)
        cv2.rectangle(
            output,
            (x1, text_y - text_h - 8),
            (x1 + text_w + 8, text_y + baseline),
            box_color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 4, text_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    return output
