# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Tests for object tracking."""

from app.detection.detector import Detection
from app.tracking.tracker import TrackMonitor


def _det(track_id: int, name: str = "cup") -> Detection:
    return Detection(
        class_id=0,
        class_name=name,
        confidence=0.9,
        bbox=(10, 10, 50, 50),
        track_id=track_id,
    )


def test_track_monitor_keeps_stable_ids() -> None:
    monitor = TrackMonitor()
    first = monitor.update([_det(1), _det(2)])
    second = monitor.update([_det(1), _det(2)])

    assert first.new_track_ids == (1, 2)
    assert first.lost_track_ids == ()
    assert second.new_track_ids == ()
    assert second.lost_track_ids == ()
    assert set(second.active_track_ids) == {1, 2}
    assert monitor.active_objects[1].frames_seen == 2


def test_track_monitor_detects_new_and_lost_objects() -> None:
    monitor = TrackMonitor()
    monitor.update([_det(1)])
    update = monitor.update([_det(2)])

    assert update.new_track_ids == (2,)
    assert update.lost_track_ids == (1,)
    assert update.active_track_ids == (2,)


def test_track_monitor_reset_clears_state() -> None:
    monitor = TrackMonitor()
    monitor.update([_det(1)])
    monitor.reset()
    assert monitor.active_objects == {}
    assert monitor.last_update.active_track_ids == ()
