"""Background VLM worker thread."""

from __future__ import annotations

import logging
import queue
from typing import Optional

from PySide6.QtCore import QThread, Signal

from app.config import AppConfig
from app.vision.local_vlm import create_vision_model
from app.vision.vision_model import VisionAnalysisRequest, VisionAnalysisResult, VisionModel
from app.vision.vlm_scheduler import VlmRequestDecision, VlmScheduler

logger = logging.getLogger(__name__)


class VlmThread(QThread):
    """Runs local VLM inference off the UI thread."""

    availability_checked = Signal(bool, str)
    analysis_started = Signal(int, bool)  # frame_id, manual
    analysis_completed = Signal(object)  # VisionAnalysisResult
    analysis_rejected = Signal(str)

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._scheduler = VlmScheduler(config.vlm_cooldown)
        self._queue: queue.Queue[Optional[VisionAnalysisRequest]] = queue.Queue()
        self._model: Optional[VisionModel] = None
        self._running = True

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)
        self.wait(5000)

    def update_latest_frame_id(self, frame_id: int) -> None:
        self._scheduler.update_latest_frame(frame_id)

    def request_analysis(self, request: VisionAnalysisRequest) -> None:
        decision = self._scheduler.schedule(request)
        if decision.decision != VlmRequestDecision.ACCEPT:
            if decision.decision == VlmRequestDecision.REJECT_BUSY and request.manual:
                self._queue.put(request)
            self.analysis_rejected.emit(decision.message)
            return
        self._queue.put(request)

    def run(self) -> None:
        try:
            self._model = create_vision_model(self._config)
            available, message = self._model.check_availability()
            self.availability_checked.emit(available, message)
            if not available:
                logger.warning("VLM unavailable: %s", message)
        except Exception as exc:
            self.availability_checked.emit(False, str(exc))
            logger.exception("Failed to initialize VLM")
            return

        while self._running:
            request = self._queue.get()
            if request is None:
                break
            assert self._model is not None
            self._scheduler.begin(request.frame_id)
            self.analysis_started.emit(request.frame_id, request.manual)
            mode = "manual" if request.manual else "auto"
            logger.info("VLM %s request started for frame #%d", mode, request.frame_id)
            result = self._model.analyze_sync(request)
            self._scheduler.complete(request.frame_id)
            self.analysis_completed.emit(result)
