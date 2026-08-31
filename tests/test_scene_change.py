"""Tests for scene change detection."""

import numpy as np

from app.detection.detector import Detection
from app.perception.scene_change import SceneChangeDetector
from app.perception.scene_stability import SceneStabilityMonitor, StabilityPhase
from app.tracking.tracker import TrackUpdate


def _frame(value: int) -> np.ndarray:
    return np.full((120, 160, 3), value, dtype=np.uint8)


def test_identical_frames_do_not_trigger_scene_change() -> None:
    detector = SceneChangeDetector(threshold=0.15, debounce_frames=1)
    frame = _frame(100)
    first = detector.analyze(frame, [], None)
    second = detector.analyze(frame.copy(), [], None)

    assert first.visual_score == 0.0
    assert second.scene_changed is False


def test_substantially_different_frames_trigger_scene_change() -> None:
    detector = SceneChangeDetector(threshold=0.15, debounce_frames=1)
    detector.analyze(_frame(20), [], None)
    result = detector.analyze(_frame(220), [], None)

    assert result.visual_score > 0.15
    assert result.scene_changed is True
    assert result.rising_edge is True


def test_new_tracked_object_triggers_detection_change() -> None:
    detector = SceneChangeDetector(threshold=0.15, debounce_frames=1)
    detector.analyze(_frame(50), [], TrackUpdate(active_track_ids=(1,)))
    update = TrackUpdate(new_track_ids=(2,), active_track_ids=(1, 2))
    detection = Detection(0, "cup", 0.9, (10, 10, 40, 40), track_id=2)
    result = detector.analyze(_frame(50), [detection], update)

    assert result.detection_score >= 0.7
    assert "new_object" in result.reasons


def test_stability_requires_consecutive_stable_frames() -> None:
    monitor = SceneStabilityMonitor(required_frames=3)

    state = monitor.update(rising_edge=True, movement_score=0.2)
    assert state.phase == StabilityPhase.WAITING_FOR_STABILITY

    state = monitor.update(rising_edge=False, movement_score=0.01)
    assert state.stable_frames == 1
    assert state.is_stable is False

    state = monitor.update(rising_edge=False, movement_score=0.01)
    state = monitor.update(rising_edge=False, movement_score=0.01)
    assert state.stable_frames == 3
    assert state.is_stable is True
    assert state.phase == StabilityPhase.READY
