# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""YOLO26 detector backed by yolo-mlx (native Apple Silicon / Metal)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Optional

import numpy as np

from app.config import AppConfig
from app.detection.backend import MLX_INSTALL_HINT, MLX_REPO, ensure_mlx_weights
from app.detection.detector import DetectionMetrics, DetectionResult, ObjectDetector
from app.detection.parsing import parse_yolo_results
from app.tracking.tracker import TrackMonitor

logger = logging.getLogger(__name__)


class YoloMlxDetector(ObjectDetector):
    """YOLO26 detector using yolo-mlx instead of PyTorch MPS."""

    def __init__(self, config: AppConfig, device: str = "mlx") -> None:
        self._config = config
        self._device = device
        self._model = None
        self._ready = False
        self._load_error: Optional[str] = None
        self._timestamps: deque[float] = deque(maxlen=60)
        self._model_path = ""
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
        """Load yolo-mlx weights, converting from .pt on first use if needed."""
        try:
            from yolo26mlx import YOLO
        except ImportError as exc:
            self._load_error = (
                "yolo-mlx is not installed. Install with: "
                f"{MLX_INSTALL_HINT}  "
                "Or switch to PyTorch MPS: YOLO_RUNTIME=ultralytics and YOLO_DEVICE=mps."
            )
            logger.error(self._load_error)
            raise RuntimeError(self._load_error) from exc

        try:
            if self._config.tracking_enabled:
                try:
                    from yolo26mlx.engine.tracker import TrackerManager  # noqa: F401
                except ImportError as exc:
                    raise RuntimeError(
                        "yolo-mlx [tracking] extras are required for TRACKING_ENABLED "
                        f"(model.track()). Install with: {MLX_INSTALL_HINT}  "
                        f"Official: {MLX_REPO}  "
                        "Or set TRACKING_ENABLED=false."
                    ) from exc

            self._model_path = ensure_mlx_weights(self._config.yolo_model)
            logger.info("Loading YOLO26 MLX model: %s", self._model_path)
            self._model = YOLO(self._model_path, task="detect", verbose=False)
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self._run_inference(dummy)
            self._ready = True
            self._load_error = None
            logger.info("YOLO26 loaded on mlx")
        except Exception as exc:
            self._ready = False
            self._load_error = str(exc)
            logger.exception("Failed to load YOLO26 (mlx)")
            raise

    async def detect(self, frame: np.ndarray, frame_id: int = -1) -> DetectionResult:
        if not self._ready or self._model is None:
            return DetectionResult(frame_id=frame_id)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.detect_sync, frame, frame_id)

    def detect_sync(self, frame: np.ndarray, frame_id: int = -1) -> DetectionResult:
        if not self._ready or self._model is None:
            return DetectionResult(frame_id=frame_id)

        started = time.perf_counter()
        try:
            results = self._run_inference(frame)
        except Exception:
            logger.exception("YOLO26 MLX inference failed")
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
        logger.info("YOLO26 mlx inference: %.1fms (%d objects)", inference_ms, len(detections))
        return DetectionResult(
            detections=detections,
            frame_id=frame_id,
            metrics=metrics,
            track_update=track_update,
        )

    def _run_inference(self, frame_bgr: np.ndarray):
        """Run official yolo26mlx predict/track on an OpenCV BGR frame.

        yolo-mlx treats numpy arrays as VideoCapture-style BGR and converts
        to RGB internally (see Predictor._preprocess_cv2). Do not pre-convert.
        """
        assert self._model is not None
        kwargs = {
            "conf": self._config.yolo_confidence,
            "imgsz": self._config.yolo_image_size,
            "save": False,
        }
        if self._config.tracking_enabled:
            try:
                return self._model.track(
                    frame_bgr,
                    persist=True,
                    tracker=self._config.tracker_type,
                    show=False,
                    **kwargs,
                )
            except ImportError:
                logger.error(
                    "yolo-mlx track() needs [tracking] extras. Install with: %s  "
                    "Falling back to predict() (no track IDs).",
                    MLX_INSTALL_HINT,
                )
            except Exception:
                logger.warning(
                    "yolo-mlx track() failed; falling back to predict().",
                    exc_info=True,
                )
        return self._model.predict(frame_bgr, **kwargs)

    def _detection_fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
