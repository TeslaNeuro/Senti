"""Background detection worker that always prefers the latest camera frame."""

from __future__ import annotations

import logging
import time
from typing import Optional

from PySide6.QtCore import QThread, Signal

from app.camera.capture import CameraCapture
from app.camera.frame_buffer import CameraFrame
from app.config import AppConfig
from app.detection.detector import DetectionResult
from app.detection.yolo26_detector import Yolo26Detector
from app.perception.engine import ScenePerceptionEngine, SceneState

logger = logging.getLogger(__name__)


class DetectionThread(QThread):
    """Runs YOLO26 inference off the UI thread."""

    model_loaded = Signal(str)  # device name
    model_failed = Signal(str)  # error message
    result_ready = Signal(object, object, object)  # CameraFrame, DetectionResult, SceneState
    metrics_updated = Signal(object)  # DetectionResult.metrics

    def __init__(self, config: AppConfig, camera: CameraCapture, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._camera = camera
        self._detector: Optional[Yolo26Detector] = None
        self._scene_engine = ScenePerceptionEngine(config)
        self._running = True
        self._last_processed_id = -1
        self._vlm_latest_frame_id = -1

    @property
    def last_selected_frame(self):
        return self._scene_engine.last_selected_frame

    @property
    def latest_frame_id(self) -> int:
        return self._vlm_latest_frame_id

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    def run(self) -> None:
        self._detector = Yolo26Detector(self._config)
        try:
            self._detector.load()
            self.model_loaded.emit(self._detector.device)
        except Exception as exc:
            self.model_failed.emit(str(exc))
            return

        while self._running:
            frame = self._camera.latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            if frame.frame_id == self._last_processed_id:
                time.sleep(0.005)
                continue

            # Always prefer the newest frame; skip stale queued work.
            latest = frame
            while True:
                newer = self._camera.latest_frame()
                if newer is None or newer.frame_id == latest.frame_id:
                    break
                latest = newer

            self._last_processed_id = latest.frame_id
            assert self._detector is not None
            result = self._detector.detect_sync(latest.data, latest.frame_id)
            scene_state = self._scene_engine.update(latest.data, result, latest)
            self.result_ready.emit(latest, result, scene_state)
            self._vlm_latest_frame_id = latest.frame_id
            if result.metrics is not None:
                self.metrics_updated.emit(result.metrics)

            # Brief yield so we don't spin if inference is very fast.
            time.sleep(0.001)
