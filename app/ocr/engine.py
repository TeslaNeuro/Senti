# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""OCR types and engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class OcrResult:
    """Structured OCR detection."""

    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass(frozen=True)
class OcrAnalysis:
    """OCR output for a single frame."""

    results: list[OcrResult]
    frame_id: int
    latency_ms: float
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def text_lines(self) -> list[str]:
        return [result.text for result in self.results if result.text.strip()]


@dataclass(frozen=True)
class OcrRequest:
    """Input for a single OCR pass."""

    image_bgr: np.ndarray
    frame_id: int
    respond_in_ui: bool = False


def format_ocr_results(results: list[OcrResult]) -> str:
    if not results:
        return "No readable text detected."

    lines = ["Detected text:"]
    for result in results:
        lines.append(f'- "{result.text}" ({result.confidence:.0%})')
    return "\n".join(lines)


class OcrEngine(ABC):
    """Abstract local OCR backend."""

    @abstractmethod
    def load(self) -> None:
        """Load OCR models."""

    @abstractmethod
    def recognize(self, image_bgr: np.ndarray) -> list[OcrResult]:
        """Run OCR on a BGR image."""

    @abstractmethod
    def check_availability(self) -> tuple[bool, str]:
        """Return whether the backend is usable."""
