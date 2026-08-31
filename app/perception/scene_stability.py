# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Scene stability monitoring after a detected change."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class StabilityPhase(Enum):
    WATCHING = auto()
    SCENE_CHANGED = auto()
    WAITING_FOR_STABILITY = auto()
    READY = auto()


@dataclass(frozen=True)
class StabilityState:
    """Current stability status for the visual scene."""

    phase: StabilityPhase
    stable_frames: int
    required_frames: int
    is_stable: bool


class SceneStabilityMonitor:
    """Wait for the scene to settle before deep analysis."""

    def __init__(self, required_frames: int, movement_threshold: float = 0.04) -> None:
        self._required_frames = required_frames
        self._movement_threshold = movement_threshold
        self._stable_frames = 0
        self._phase = StabilityPhase.WATCHING

    @property
    def phase(self) -> StabilityPhase:
        return self._phase

    def update(self, rising_edge: bool, movement_score: float) -> StabilityState:
        if rising_edge:
            self._phase = StabilityPhase.SCENE_CHANGED
            self._stable_frames = 0

        if self._phase == StabilityPhase.SCENE_CHANGED:
            self._phase = StabilityPhase.WAITING_FOR_STABILITY

        if self._phase in {
            StabilityPhase.WAITING_FOR_STABILITY,
            StabilityPhase.READY,
        }:
            if movement_score <= self._movement_threshold:
                self._stable_frames += 1
            else:
                self._stable_frames = 0

            if self._stable_frames >= self._required_frames:
                self._phase = StabilityPhase.READY
        elif self._phase == StabilityPhase.WATCHING and rising_edge:
            pass  # handled above

        is_stable = self._stable_frames >= self._required_frames
        return StabilityState(
            phase=self._phase,
            stable_frames=self._stable_frames,
            required_frames=self._required_frames,
            is_stable=is_stable,
        )

    def reset(self) -> None:
        self._stable_frames = 0
        self._phase = StabilityPhase.WATCHING
