"""Ultralytics YOLO26 object detector."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import AppConfig
from app.detection.detector import Detection, DetectionMetrics, DetectionResult, ObjectDetector
from app.tracking.tracker import TrackMonitor, TrackUpdate

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


def resolve_model_path(model_name: str) -> str:
    """Prefer a local model file in models/ when present."""
    project_root = Path(__file__).resolve().parent.parent.parent
    local_path = project_root / "models" / model_name
    if local_path.is_file():
        return str(local_path)
    return model_name


class Yolo26Detector(ObjectDetector):
    """YOLO26 detector backed by Ultralytics."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._device = resolve_yolo_device(config.yolo_device)
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
        detections = self._parse_results(results)
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
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        names = result.names or {}
        detections: list[Detection] = []
        xyxy = boxes.xyxy.cpu().numpy()
        if xyxy.size == 0:
            return []
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = None
        if getattr(boxes, "id", None) is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)

        for idx, (x1, y1, x2, y2) in enumerate(xyxy):
            class_id = int(class_ids[idx])
            track_id = int(track_ids[idx]) if track_ids is not None else None
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=str(names.get(class_id, f"class_{class_id}")),
                    confidence=float(confidences[idx]),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    track_id=track_id,
                )
            )
        return detections

    def _detection_fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
