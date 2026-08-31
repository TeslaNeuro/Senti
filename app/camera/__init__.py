# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Camera capture and bounded in-memory frame buffering.

Frames stay in RAM only. Nothing in this package writes video to disk.
"""

from app.camera.frame_buffer import BoundedFrameBuffer, CameraFrame, CameraFrameBuffer

__all__ = [
    "BoundedFrameBuffer",
    "CameraFrame",
    "CameraFrameBuffer",
    "CameraCapture",
    "CameraStatus",
]


def __getattr__(name: str):
    if name in {"CameraCapture", "CameraStatus"}:
        from app.camera.capture import CameraCapture, CameraStatus

        return CameraCapture if name == "CameraCapture" else CameraStatus
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
