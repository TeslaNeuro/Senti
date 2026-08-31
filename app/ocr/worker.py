# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Background OCR worker thread."""

from __future__ import annotations

import logging
import queue
import time
from typing import Optional

from PySide6.QtCore import QThread, Signal

from app.config import AppConfig
from app.ocr.easyocr_engine import create_ocr_engine
from app.ocr.engine import OcrAnalysis, OcrEngine, OcrRequest

logger = logging.getLogger(__name__)


class OcrThread(QThread):
    """Runs local OCR off the UI thread."""

    availability_checked = Signal(bool, str)
    recognition_started = Signal(int)
    recognition_completed = Signal(object)  # OcrAnalysis
    recognition_rejected = Signal(str)

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._queue: queue.Queue[Optional[OcrRequest]] = queue.Queue()
        self._engine: Optional[OcrEngine] = None
        self._running = True
        self._busy = False

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)
        self.wait(5000)

    def request_recognition(self, request: OcrRequest) -> None:
        if self._busy:
            self.recognition_rejected.emit("OCR is already running.")
            return
        self._queue.put(request)

    def run(self) -> None:
        if not self._config.ocr_enabled:
            self.availability_checked.emit(False, "OCR disabled in config.")
            return

        try:
            self._engine = create_ocr_engine(self._config)
            available, message = self._engine.check_availability()
            self.availability_checked.emit(available, message)
            if not available:
                logger.warning("OCR unavailable: %s", message)
                return
        except Exception as exc:
            self.availability_checked.emit(False, str(exc))
            logger.exception("Failed to initialize OCR")
            return

        while self._running:
            request = self._queue.get()
            if request is None:
                break
            assert self._engine is not None
            self._busy = True
            self.recognition_started.emit(request.frame_id)
            logger.info("OCR started for frame #%d", request.frame_id)
            started = time.perf_counter()
            try:
                results = self._engine.recognize(request.image_bgr)
                latency_ms = (time.perf_counter() - started) * 1000.0
                analysis = OcrAnalysis(
                    results=results,
                    frame_id=request.frame_id,
                    latency_ms=latency_ms,
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1000.0
                logger.exception("OCR failed")
                analysis = OcrAnalysis(
                    results=[],
                    frame_id=request.frame_id,
                    latency_ms=latency_ms,
                    error=str(exc),
                )
            self._busy = False
            self.recognition_completed.emit(analysis)
