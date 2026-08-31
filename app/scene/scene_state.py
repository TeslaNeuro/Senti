"""Current visual scene context for conversational interaction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from app.detection.detector import Detection
from app.ocr.engine import OcrAnalysis, OcrResult
from app.scene.conversation import ConversationTurn, build_conversation_context
from app.vision.vision_model import VisionAnalysisResult


@dataclass
class CurrentScene:
    """In-memory understanding of the current visual scene."""

    objects: list[Detection] = field(default_factory=list)
    primary_object: Optional[Detection] = None
    focused_object: Optional[Detection] = None
    description: str = ""
    ocr_text: list[str] = field(default_factory=list)
    ocr_results: list[OcrResult] = field(default_factory=list)
    ocr_frame_id: int = -1
    frame_id: int = -1
    last_analysis_time: Optional[datetime] = None
    confidence: float = 0.0
    analyzed_frame_bgr: Optional[np.ndarray] = None

    def has_context(self) -> bool:
        return bool(self.description or self.objects or self.ocr_text)

    def apply_ocr(self, analysis: OcrAnalysis) -> None:
        if not analysis.ok:
            return
        self.ocr_results = list(analysis.results)
        self.ocr_text = analysis.text_lines
        self.ocr_frame_id = analysis.frame_id

    def is_stale(self, latest_frame_id: int, max_gap: int = 45) -> bool:
        if self.frame_id < 0:
            return True
        return latest_frame_id - self.frame_id > max_gap

    def update_from_analysis(
        self,
        result: VisionAnalysisResult,
        detections: list[Detection],
        image_bgr: Optional[np.ndarray] = None,
    ) -> None:
        if not result.ok:
            return
        self.objects = list(detections)
        self.primary_object = _pick_primary_object(detections)
        self.focused_object = None
        self.description = result.description
        self.frame_id = result.frame_id
        self.last_analysis_time = datetime.now()
        self.confidence = max((det.confidence for det in detections), default=0.0)
        if image_bgr is not None:
            self.analyzed_frame_bgr = image_bgr.copy()

    def clear(self) -> None:
        self.objects = []
        self.primary_object = None
        self.focused_object = None
        self.description = ""
        self.ocr_text = []
        self.ocr_results = []
        self.ocr_frame_id = -1
        self.frame_id = -1
        self.last_analysis_time = None
        self.confidence = 0.0
        self.analyzed_frame_bgr = None


def _pick_primary_object(detections: list[Detection]) -> Optional[Detection]:
    if not detections:
        return None
    return max(detections, key=lambda det: det.confidence)


class SceneContextManager:
    """Stores scene understanding and short conversation history."""

    def __init__(self, max_turns: int = 8) -> None:
        self._scene = CurrentScene()
        self._turns: list[ConversationTurn] = []
        self._max_turns = max_turns

    @property
    def scene(self) -> CurrentScene:
        return self._scene

    @property
    def turns(self) -> list[ConversationTurn]:
        return list(self._turns)

    def apply_analysis(
        self,
        result: VisionAnalysisResult,
        detections: list[Detection],
        image_bgr: Optional[np.ndarray] = None,
    ) -> CurrentScene:
        self._scene.update_from_analysis(result, detections, image_bgr=image_bgr)
        if result.ok and result.description:
            self.add_assistant_turn(result.description)
        return self._scene

    def apply_ocr(self, analysis: OcrAnalysis) -> CurrentScene:
        self._scene.apply_ocr(analysis)
        return self._scene

    def add_user_turn(self, content: str) -> None:
        self._append_turn("user", content)

    def add_assistant_turn(self, content: str) -> None:
        self._append_turn("assistant", content)

    def _append_turn(self, role: str, content: str) -> None:
        text = content.strip()
        if not text:
            return
        self._turns.append(ConversationTurn(role=role, content=text))
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns :]

    def reset_conversation(self) -> None:
        self._turns.clear()

    def reset_scene(self) -> None:
        self._scene.clear()
        self.reset_conversation()

    def previous_context(self) -> Optional[str]:
        parts: list[str] = []
        if self._scene.description:
            parts.append(f"Current scene: {self._scene.description}")
        if self._scene.primary_object is not None:
            primary = self._scene.primary_object
            label = primary.class_name
            if primary.track_id is not None:
                label = f"#{primary.track_id} {label}"
            parts.append(f"Primary object: {label} ({primary.confidence:.0%})")
        if self._scene.ocr_text:
            parts.append("OCR text:\n" + "\n".join(f'- "{line}"' for line in self._scene.ocr_text))
        conversation = build_conversation_context(self._turns, max_turns=self._max_turns)
        if conversation:
            parts.append(conversation)
        if not parts:
            return None
        return "\n\n".join(parts)
