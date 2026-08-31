"""Object detection types and abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from app.tracking.tracker import TrackUpdate


@dataclass(frozen=True)
class Detection:
    """A single detected object."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    track_id: Optional[int] = None


@dataclass(frozen=True)
class DetectionMetrics:
    """Performance metrics for a detection pass."""

    inference_ms: float
    fps: float
    device: str


@dataclass(frozen=True)
class DetectionResult:
    """Structured output from an object detector."""

    detections: list[Detection] = field(default_factory=list)
    frame_id: int = -1
    metrics: Optional[DetectionMetrics] = None
    track_update: Optional[TrackUpdate] = None


class ObjectDetector(ABC):
    """Abstract real-time object detector."""

    @abstractmethod
    async def detect(self, frame: np.ndarray, frame_id: int = -1) -> DetectionResult:
        """Run detection on a BGR frame."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True when the model is loaded and usable."""

    @property
    @abstractmethod
    def device(self) -> str:
        """Inference device label (e.g. mps, cpu)."""
