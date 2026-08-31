# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Tests for bounded frame buffer."""

import numpy as np
import pytest

from app.camera.frame_buffer import BoundedFrameBuffer, CameraFrameBuffer


def test_bounded_buffer_respects_max_size() -> None:
  buffer: BoundedFrameBuffer[int] = BoundedFrameBuffer(max_size=3)
  for value in range(5):
    buffer.push(value)
  assert buffer.size == 3
  assert buffer.latest() == 4
  assert buffer.dropped_count == 2


def test_bounded_buffer_latest_empty() -> None:
  buffer: BoundedFrameBuffer[int] = BoundedFrameBuffer(max_size=2)
  assert buffer.latest() is None


def test_bounded_buffer_snapshot() -> None:
  buffer: BoundedFrameBuffer[int] = BoundedFrameBuffer(max_size=4)
  for value in (1, 2, 3):
    buffer.push(value)
  assert buffer.snapshot() == [1, 2, 3]


def test_camera_frame_buffer_assigns_ids() -> None:
  buffer = CameraFrameBuffer(max_size=4)
  frame_a = buffer.push_frame(np.zeros((10, 10, 3), dtype=np.uint8))
  frame_b = buffer.push_frame(np.ones((10, 10, 3), dtype=np.uint8))
  assert frame_a.frame_id == 0
  assert frame_b.frame_id == 1
  assert buffer.latest() is frame_b


def test_invalid_buffer_size_raises() -> None:
  with pytest.raises(ValueError, match="max_size must be positive"):
    BoundedFrameBuffer(max_size=0)
