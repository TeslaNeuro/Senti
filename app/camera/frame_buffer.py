# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Bounded in-memory frame buffer for the vision pipeline."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Generic, Optional, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class CameraFrame:
    """A single captured camera frame with metadata."""

    data: np.ndarray
    timestamp: float
    frame_id: int
    width: int
    height: int


class BoundedFrameBuffer(Generic[T]):
    """Thread-safe bounded buffer that drops oldest items when full."""

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self._max_size = max_size
        self._items: Deque[T] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._dropped_count = 0

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    def push(self, item: T) -> None:
        with self._lock:
            if len(self._items) == self._max_size:
                self._dropped_count += 1
            self._items.append(item)

    def latest(self) -> Optional[T]:
        with self._lock:
            if not self._items:
                return None
            return self._items[-1]

    def snapshot(self) -> list[T]:
        with self._lock:
            return list(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


class CameraFrameBuffer(BoundedFrameBuffer[CameraFrame]):
    """Bounded buffer specialized for camera frames."""

    def __init__(self, max_size: int) -> None:
        super().__init__(max_size)
        self._next_frame_id = 0
        self._frame_id_lock = threading.Lock()

    def push_frame(self, data: np.ndarray) -> CameraFrame:
        with self._frame_id_lock:
            frame_id = self._next_frame_id
            self._next_frame_id += 1

        height, width = data.shape[:2]
        frame = CameraFrame(
            data=data,
            timestamp=time.monotonic(),
            frame_id=frame_id,
            width=width,
            height=height,
        )
        self.push(frame)
        return frame
