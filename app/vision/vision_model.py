# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Vision-language model types and interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from app.detection.detector import Detection

VLM_SYSTEM_PROMPT = """You are a local visual assistant running on a user's Mac.

Your job is to understand what the user is currently looking at.

Identify the main object or scene first.

Then describe the most useful visible details.

Use visual evidence only.

Do not invent details that cannot be supported by the image.

If you are uncertain, say that you are uncertain.

YOLO detections and OCR results may be provided as hints.
They are not guaranteed to be correct.

When the user asks a specific question, answer that question directly.

Keep responses concise by default.
Provide more detail when the user asks for it."""


@dataclass(frozen=True)
class VisionAnalysisResult:
    """Structured VLM response."""

    description: str
    latency_ms: float
    model: str
    frame_id: int = -1
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class VisionAnalysisRequest:
    """Input bundle for a single VLM analysis."""

    image_bgr: np.ndarray
    frame_id: int
    detections: list[Detection] = field(default_factory=list)
    user_question: Optional[str] = None
    ocr_text: Optional[list[str]] = None
    previous_context: Optional[str] = None
    manual: bool = True
    focused_detection: Optional[Detection] = None
    is_object_crop: bool = False


def build_vlm_user_prompt(request: VisionAnalysisRequest) -> str:
    """Build the text context sent alongside the image."""
    parts: list[str] = []

    if request.is_object_crop and request.focused_detection is not None:
        det = request.focused_detection
        label = det.class_name
        if det.track_id is not None:
            label = f"#{det.track_id} {label}"
        parts.append(
            "The image is a cropped close-up of one detected object from the scene. "
            f"Focused object: {label} ({det.confidence:.0%})."
        )
        parts.append("")

    if request.detections:
        parts.append("Detected objects:")
        for det in request.detections:
            label = f"- {det.class_name} — {det.confidence:.0%}"
            if det.track_id is not None:
                label = f"- #{det.track_id} {det.class_name} — {det.confidence:.0%}"
            parts.append(label)
        parts.append("")

    if request.ocr_text:
        parts.append("OCR:")
        parts.extend(f'"{line}"' for line in request.ocr_text)
        parts.append("")

    if request.previous_context:
        parts.append(request.previous_context)
        parts.append("")

    if request.user_question:
        parts.append(f"User question: {request.user_question.strip()}")
    else:
        parts.append("What am I looking at? Describe the main object or scene.")

    return "\n".join(parts).strip()


class VisionModel(ABC):
    """Abstract local vision-language model."""

    @abstractmethod
    async def analyze(self, request: VisionAnalysisRequest) -> VisionAnalysisResult:
        """Analyze an image with optional structured context."""

    @abstractmethod
    def analyze_sync(self, request: VisionAnalysisRequest) -> VisionAnalysisResult:
        """Synchronous analysis for worker threads."""

    @abstractmethod
    def check_availability(self) -> tuple[bool, str]:
        """Return whether the backend is reachable and a status message."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Configured model identifier."""
