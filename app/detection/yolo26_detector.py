# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Ultralytics YOLO26 object detector."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import PROJECT_ROOT, AppConfig, models_directory
from app.detection.detector import Detection, DetectionMetrics, DetectionResult, ObjectDetector
from app.detection.parsing import parse_yolo_results
from app.tracking.tracker import TrackMonitor

logger = logging.getLogger(__name__)


def resolve_yolo_device(requested: str) -> str:
    """Resolve configured device to an available runtime device."""
    device = requested.strip().lower()
    if device != "auto":
        return device

    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def resolve_model_path(model_name: str, *, project_root: Path | None = None) -> str:
    """Resolve YOLO weights to an absolute path under ``models/``.

    Ultralytics downloads to whatever path ``YOLO()`` receives. A bare name
    such as ``yolo26s.pt`` would land in the process cwd (the repo root).
    Always passing ``<root>/models/<file>`` keeps weights in ``models/``.

    Weights that were previously downloaded into the project root are moved
    into ``models/`` on first resolve. Absolute paths are used as-is so a
    custom checkpoint can live outside the repo.
    """
    raw = model_name.strip()
    if not raw:
        raise ValueError("YOLO_MODEL must not be empty.")

    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return str(candidate)

    root = project_root or PROJECT_ROOT
    target = models_directory(root) / candidate.name
    legacy = root / candidate.name
    if legacy.is_file() and legacy.resolve() != target.resolve():
        if target.is_file():
            logger.info("Using %s; removing leftover %s", target, legacy)
            legacy.unlink()
        else:
            logger.info("Moving YOLO weights from %s to %s", legacy, target)
            shutil.move(str(legacy), str(target))
    return str(target)


class Yolo26Detector(ObjectDetector):
    """YOLO26 detector backed by Ultralytics."""

    def __init__(self, config: AppConfig, device: str | None = None) -> None:
        self._config = config
        self._device = device or resolve_yolo_device(config.yolo_device)
        self._model = None
        self._ready = False
        self._load_error: Optional[str] = None
        self._timestamps: deque[float] = deque(maxlen=60)
        self._model_path = resolve_model_path(config.yolo_model)
        self._track_monitor = TrackMonitor()

    @property
    def device(self) -> str:
        return self._device

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def is_ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        """Load the YOLO26 model synchronously (call from a worker thread)."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self._load_error = "ultralytics is not installed"
            logger.error(self._load_error)
            raise RuntimeError(self._load_error) from exc

        try:
            logger.info("Loading YOLO26 model: %s (device=%s)", self._model_path, self._device)
            self._model = YOLO(self._model_path)
            # Warm-up inference validates the runtime device.
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self._run_inference(dummy)
            self._ready = True
            self._load_error = None
            logger.info("YOLO26 loaded on %s", self._device)
        except Exception as exc:
            self._ready = False
            self._load_error = str(exc)
            logger.exception("Failed to load YOLO26")
            raise

    async def detect(self, frame: np.ndarray, frame_id: int = -1) -> DetectionResult:
        if not self._ready or self._model is None:
            return DetectionResult(frame_id=frame_id)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.detect_sync, frame, frame_id)

    def detect_sync(self, frame: np.ndarray, frame_id: int = -1) -> DetectionResult:
        return self._detect_sync(frame, frame_id)

    def _run_inference(self, frame: np.ndarray):
        kwargs = {
            "source": frame,
            "device": self._device,
            "imgsz": self._config.yolo_image_size,
            "conf": self._config.yolo_confidence,
            "verbose": False,
        }
        if self._config.tracking_enabled:
            return self._model.track(
                persist=True,
                tracker=self._config.tracker_type,
                **kwargs,
            )
        return self._model.predict(**kwargs)

    def _detect_sync(self, frame: np.ndarray, frame_id: int) -> DetectionResult:
        assert self._model is not None
        started = time.perf_counter()
        try:
            results = self._run_inference(frame)
        except Exception:
            logger.exception("YOLO26 inference failed")
            return DetectionResult(frame_id=frame_id)

        inference_ms = (time.perf_counter() - started) * 1000.0
        self._timestamps.append(time.perf_counter())
        detections = parse_yolo_results(results)
        track_update = None
        if self._config.tracking_enabled:
            track_update = self._track_monitor.update(detections)
            if track_update.new_track_ids:
                logger.info("New tracked objects: %s", list(track_update.new_track_ids))
            if track_update.lost_track_ids:
                logger.info("Lost tracked objects: %s", list(track_update.lost_track_ids))
        metrics = DetectionMetrics(
            inference_ms=inference_ms,
            fps=self._detection_fps(),
            device=self._device,
        )
        logger.info("YOLO26 inference: %.1fms (%d objects)", inference_ms, len(detections))
        return DetectionResult(
            detections=detections,
            frame_id=frame_id,
            metrics=metrics,
            track_update=track_update,
        )

    def _parse_results(self, results) -> list[Detection]:
        return parse_yolo_results(results)

    def _detection_fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
