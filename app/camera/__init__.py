"""Camera package."""

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
