# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""EasyOCR-backed local text recognition."""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from app.config import AppConfig
from app.ocr.engine import OcrEngine, OcrResult

logger = logging.getLogger(__name__)


class EasyOcrEngine(OcrEngine):
    """Local OCR via EasyOCR (CPU by default on Apple Silicon)."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._reader: Optional[object] = None

    def load(self) -> None:
        if self._reader is not None:
            return
        import easyocr

        languages = [lang.strip() for lang in self._config.ocr_languages.split(",") if lang.strip()]
        logger.info("Loading EasyOCR (%s)", ", ".join(languages))
        self._reader = easyocr.Reader(languages, gpu=False)

    def check_availability(self) -> tuple[bool, str]:
        import importlib.util

        if importlib.util.find_spec("easyocr") is None:
            return False, "EasyOCR not installed. Run: pip install easyocr"
        return True, "EasyOCR ready (loads on first use)"

    def recognize(self, image_bgr: np.ndarray) -> list[OcrResult]:
        self.load()
        assert self._reader is not None
        started = time.perf_counter()
        raw_results = self._reader.readtext(image_bgr)
        latency_ms = (time.perf_counter() - started) * 1000.0
        logger.info("OCR completed: %d lines in %.1fs", len(raw_results), latency_ms / 1000.0)

        results: list[OcrResult] = []
        for bbox, text, confidence in raw_results:
            if confidence < self._config.ocr_min_confidence:
                continue
            cleaned = str(text).strip()
            if not cleaned:
                continue
            xs = [int(point[0]) for point in bbox]
            ys = [int(point[1]) for point in bbox]
            results.append(
                OcrResult(
                    text=cleaned,
                    confidence=float(confidence),
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                )
            )
        return results


def create_ocr_engine(config: AppConfig) -> OcrEngine:
    runtime = config.ocr_runtime.strip().lower()
    if runtime == "easyocr":
        return EasyOcrEngine(config)
    raise ValueError(f"Unsupported OCR_RUNTIME: {config.ocr_runtime}")
