# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Normalize YOLO result tensors from Ultralytics (torch) or yolo-mlx (numpy)."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.detection.detector import Detection


def as_numpy(value: Any) -> np.ndarray | None:
    """Convert a torch tensor, MLX array, or numpy array to ``np.ndarray``."""
    if value is None:
        return None
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        value = numpy_fn()
    return np.asarray(value)


def iter_yolo_results(results: Any) -> list[Any]:
    """Normalize predict/track output to a list of result objects."""
    if results is None:
        return []
    if isinstance(results, (list, tuple)):
        return list(results)
    return [results]


def parse_yolo_results(results: Any, names: dict | None = None) -> list[Detection]:
    """Parse Ultralytics-style or yolo-mlx Results into ``Detection`` rows."""
    items = iter_yolo_results(results)
    if not items:
        return []

    result = items[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    class_names = names if names is not None else (getattr(result, "names", None) or {})
    xyxy = as_numpy(getattr(boxes, "xyxy", None))
    if xyxy is None or xyxy.size == 0:
        return []
    confidences = as_numpy(getattr(boxes, "conf", None))
    class_ids = as_numpy(getattr(boxes, "cls", None))
    if confidences is None or class_ids is None:
        return []

    track_ids = None
    raw_ids = getattr(boxes, "id", None)
    if raw_ids is not None:
        track_ids = as_numpy(raw_ids)

    detections: list[Detection] = []
    for idx, (x1, y1, x2, y2) in enumerate(xyxy):
        class_id = int(class_ids[idx])
        track_id = int(track_ids[idx]) if track_ids is not None else None
        name = class_names.get(class_id, class_names.get(str(class_id), f"class_{class_id}"))
        detections.append(
            Detection(
                class_id=class_id,
                class_name=str(name),
                confidence=float(confidences[idx]),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                track_id=track_id,
            )
        )
    return detections
