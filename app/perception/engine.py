# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Combined scene perception engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from app.config import AppConfig
from app.camera.frame_buffer import CameraFrame
from app.detection.detector import DetectionResult
from app.perception.scene_change import SceneChangeAnalysis, SceneChangeDetector
from app.perception.scene_stability import SceneStabilityMonitor, StabilityPhase, StabilityState
from app.perception.frame_selector import FrameSelector, SelectedFrame

logger = logging.getLogger(__name__)


class AssistantState(Enum):
    INITIALIZING = auto()
    CAMERA_READY = auto()
    WATCHING = auto()
    SCENE_CHANGED = auto()
    WAITING_FOR_STABILITY = auto()
    READY = auto()
    ERROR = auto()


@dataclass(frozen=True)
class SceneState:
    """Full scene perception output for one processed frame."""

    assistant_state: AssistantState
    scene_changed: bool
    change: SceneChangeAnalysis
    stability: StabilityState
    selected_frame: Optional[SelectedFrame] = None
    frame_selection_count: int = 0


class ScenePerceptionEngine:
    """Coordinates scene-change detection and stability monitoring."""

    def __init__(self, config: AppConfig) -> None:
        self._change_detector = SceneChangeDetector(config.scene_change_threshold)
        self._stability_monitor = SceneStabilityMonitor(config.stability_frames)
        self._frame_selector = FrameSelector(config.selection_buffer_size)
        self._last_phase = StabilityPhase.WATCHING

    def update(
        self,
        frame_bgr: np.ndarray,
        result: DetectionResult,
        camera_frame: CameraFrame,
    ) -> SceneState:
        change = self._change_detector.analyze(
            frame_bgr,
            result.detections,
            result.track_update,
        )
        stability = self._stability_monitor.update(change.rising_edge, change.movement_score)
        assistant_state = self._map_assistant_state(stability.phase, change.rising_edge)

        if change.rising_edge:
            logger.info(
                "Scene change detected (score=%.2f, reasons=%s)",
                change.combined_score,
                ", ".join(change.reasons) or "unknown",
            )
            self._frame_selector.reset()

        self._frame_selector.add(camera_frame, result, change.movement_score)

        selected: Optional[SelectedFrame] = None
        if (
            stability.phase == StabilityPhase.READY
            and self._last_phase != StabilityPhase.READY
        ):
            selected = self._frame_selector.select_best()
            if selected is not None:
                candidate = selected.candidate
                logger.info(
                    "Best frame selected: #%d score=%.2f (sharp=%.2f stable=%.2f)",
                    candidate.camera_frame.frame_id,
                    candidate.score.total,
                    candidate.score.sharpness,
                    candidate.score.stability,
                )

        if stability.is_stable and stability.phase == StabilityPhase.READY:
            if self._last_phase != StabilityPhase.READY:
                logger.info("Object stabilized (%d frames)", stability.stable_frames)

        self._last_phase = stability.phase

        active_selected = selected
        if active_selected is None and stability.phase == StabilityPhase.READY:
            active_selected = self._frame_selector.last_selected

        return SceneState(
            assistant_state=assistant_state,
            scene_changed=change.scene_changed,
            change=change,
            stability=stability,
            selected_frame=active_selected,
            frame_selection_count=self._frame_selector.size,
        )

    def _map_assistant_state(
        self,
        phase: StabilityPhase,
        rising_edge: bool,
    ) -> AssistantState:
        if rising_edge:
            return AssistantState.SCENE_CHANGED
        mapping = {
            StabilityPhase.WATCHING: AssistantState.WATCHING,
            StabilityPhase.SCENE_CHANGED: AssistantState.SCENE_CHANGED,
            StabilityPhase.WAITING_FOR_STABILITY: AssistantState.WAITING_FOR_STABILITY,
            StabilityPhase.READY: AssistantState.READY,
        }
        return mapping.get(phase, AssistantState.WATCHING)

    def reset(self) -> None:
        self._change_detector.reset()
        self._stability_monitor.reset()
        self._frame_selector.reset()
        self._last_phase = StabilityPhase.WATCHING

    @property
    def last_selected_frame(self):
        return self._frame_selector.last_selected
