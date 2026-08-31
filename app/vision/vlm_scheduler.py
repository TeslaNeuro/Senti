"""VLM request scheduling and duplicate prevention."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from app.vision.vision_model import VisionAnalysisRequest


class VlmRequestDecision(Enum):
    ACCEPT = auto()
    REJECT_BUSY = auto()
    REJECT_COOLDOWN = auto()
    REJECT_STALE = auto()
    REJECT_DUPLICATE = auto()


@dataclass(frozen=True)
class VlmScheduleResult:
    decision: VlmRequestDecision
    message: str = ""


class VlmScheduler:
    """Prevents excessive or duplicate VLM inference."""

    def __init__(self, cooldown_seconds: float) -> None:
        self._cooldown_seconds = cooldown_seconds
        self._running = False
        self._last_completed_at = 0.0
        self._latest_frame_id = -1
        self._pending_frame_id: Optional[int] = None
        self._last_analyzed_frame_id: Optional[int] = None

    @property
    def is_running(self) -> bool:
        return self._running

    def update_latest_frame(self, frame_id: int) -> None:
        self._latest_frame_id = frame_id

    def schedule(self, request: VisionAnalysisRequest) -> VlmScheduleResult:
        if self._running:
            if request.manual:
                return VlmScheduleResult(
                    VlmRequestDecision.REJECT_BUSY,
                    "VLM is already running. Please wait for the current analysis to finish.",
                )
            return VlmScheduleResult(VlmRequestDecision.REJECT_BUSY, "VLM is already running.")

        if not request.manual:
            elapsed = time.monotonic() - self._last_completed_at
            if elapsed < self._cooldown_seconds:
                return VlmScheduleResult(
                    VlmRequestDecision.REJECT_COOLDOWN,
                    f"VLM cooldown active ({elapsed:.1f}s / {self._cooldown_seconds:.1f}s).",
                )
            if self._last_analyzed_frame_id == request.frame_id:
                return VlmScheduleResult(
                    VlmRequestDecision.REJECT_DUPLICATE,
                    f"Frame #{request.frame_id} was already analyzed.",
                )

        if request.frame_id < self._latest_frame_id and not request.manual:
            return VlmScheduleResult(VlmRequestDecision.REJECT_STALE, "Frame is stale.")

        return VlmScheduleResult(VlmRequestDecision.ACCEPT)

    def begin(self, frame_id: int) -> None:
        self._running = True
        self._pending_frame_id = frame_id

    def complete(self, frame_id: int) -> None:
        self._running = False
        self._last_completed_at = time.monotonic()
        self._last_analyzed_frame_id = frame_id
        self._pending_frame_id = None

    def take_pending_frame_id(self) -> Optional[int]:
        frame_id = self._pending_frame_id
        self._pending_frame_id = None
        return frame_id
