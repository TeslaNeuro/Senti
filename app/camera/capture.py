"""macOS camera capture using Qt Multimedia (AVFoundation backend)."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtGui import QImage
from PySide6.QtMultimedia import (
    QCamera,
    QCameraDevice,
    QCameraFormat,
    QMediaCaptureSession,
    QMediaDevices,
    QVideoFrame,
    QVideoFrameFormat,
    QVideoSink,
)

from app.camera.frame_buffer import CameraFrame, CameraFrameBuffer
from app.config import AppConfig

logger = logging.getLogger(__name__)


class CameraStatus(Enum):
  INITIALIZING = auto()
  READY = auto()
  RUNNING = auto()
  STOPPED = auto()
  ERROR = auto()
  PERMISSION_DENIED = auto()


@dataclass(frozen=True)
class CameraMetrics:
  capture_fps: float
  frame_count: int
  dropped_frames: int
  width: int
  height: int


def qvideo_frame_to_bgr(frame: QVideoFrame) -> Optional[np.ndarray]:
  """Convert a QVideoFrame to a BGR numpy array."""
  if not frame.isValid():
    return None

  image = frame.toImage()
  if image.isNull():
    return None

  image = image.convertToFormat(QImage.Format.Format_RGB888)
  width = image.width()
  height = image.height()
  bytes_per_line = image.bytesPerLine()
  bits = image.bits()
  arr = np.frombuffer(bits, dtype=np.uint8, count=bytes_per_line * height)
  rgb = arr.reshape(height, bytes_per_line)[:, : width * 3].reshape(height, width, 3)
  return np.ascontiguousarray(rgb[:, :, ::-1])


def bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
  """Convert a BGR numpy array to QImage for display."""
  rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
  height, width, _ = rgb.shape
  bytes_per_line = 3 * width
  return QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()


class CameraCapture(QObject):
  """Asynchronous MacBook camera capture with bounded buffering."""

  frame_ready = Signal(object)  # CameraFrame
  status_changed = Signal(object)  # CameraStatus
  metrics_updated = Signal(object)  # CameraMetrics
  error_occurred = Signal(str)

  def __init__(self, config: AppConfig, parent: Optional[QObject] = None) -> None:
    super().__init__(parent)
    self._config = config
    self._status = CameraStatus.INITIALIZING
    self._frame_buffer = CameraFrameBuffer(config.frame_buffer_size)
    self._frame_count = 0
    self._frame_timestamps: deque[float] = deque(maxlen=120)
    self._recovery_attempts = 0
    self._max_recovery_attempts = 5

    self._capture_session: Optional[QMediaCaptureSession] = None
    self._camera: Optional[QCamera] = None
    self._video_sink: Optional[QVideoSink] = None
    self._recovery_timer = QTimer(self)
    self._recovery_timer.setSingleShot(True)
    self._recovery_timer.timeout.connect(self._attempt_recovery)

    self._metrics_timer = QTimer(self)
    self._metrics_timer.timeout.connect(self._emit_metrics)
    self._metrics_timer.start(500)

  @property
  def status(self) -> CameraStatus:
    return self._status

  @property
  def frame_buffer(self) -> CameraFrameBuffer:
    return self._frame_buffer

  def latest_frame(self) -> Optional[CameraFrame]:
    return self._frame_buffer.latest()

  def start(self) -> None:
    logger.info("Starting camera capture")
    self._set_status(CameraStatus.INITIALIZING)
    try:
      self._open_camera()
      logger.info(
        "Camera initialized (target %dx%d @ %d FPS)",
        self._config.camera_width,
        self._config.camera_height,
        self._config.target_fps,
      )
    except PermissionError as exc:
      self._set_status(CameraStatus.PERMISSION_DENIED)
      self.error_occurred.emit(str(exc))
      logger.error("Camera permission denied: %s", exc)
    except Exception as exc:
      self._set_status(CameraStatus.ERROR)
      self.error_occurred.emit(str(exc))
      logger.exception("Failed to start camera")
      self._schedule_recovery()

  def stop(self) -> None:
    logger.info("Stopping camera capture")
    self._recovery_timer.stop()
    if self._camera is not None:
      self._camera.stop()
    self._camera = None
    self._capture_session = None
    self._video_sink = None
    self._set_status(CameraStatus.STOPPED)

  def _open_camera(self) -> None:
    devices = QMediaDevices.videoInputs()
    if not devices:
      raise RuntimeError("No camera devices found.")

    device = self._select_device(devices)
    logger.info("Using camera: %s", device.description())

    camera_format = self._select_format(device)
    self._camera = QCamera(device)
    if camera_format.isNull():
      logger.warning("Requested resolution unavailable; using camera default format.")
    else:
      self._camera.setCameraFormat(camera_format)

    self._capture_session = QMediaCaptureSession()
    self._capture_session.setCamera(self._camera)

    self._video_sink = QVideoSink()
    self._video_sink.videoFrameChanged.connect(self._on_video_frame)
    self._capture_session.setVideoSink(self._video_sink)

    self._camera.errorOccurred.connect(self._on_camera_error)
    self._camera.start()
    self._recovery_attempts = 0
    self._set_status(CameraStatus.READY)

  def _select_device(self, devices: list[QCameraDevice]) -> QCameraDevice:
    index = self._config.camera_device
    if 0 <= index < len(devices):
      return devices[index]
    logger.warning("Camera device %d not found; using default.", index)
    return QMediaDevices.defaultVideoInput()

  def _select_format(self, device: QCameraDevice) -> QCameraFormat:
    target_width = self._config.camera_width
    target_height = self._config.camera_height
    target_fps = float(self._config.target_fps)

    formats = device.videoFormats()
    if not formats:
      return QCameraFormat()

    def score(fmt: QCameraFormat) -> tuple[float, float, float]:
      width_delta = abs(fmt.resolution().width() - target_width)
      height_delta = abs(fmt.resolution().height() - target_height)
      fps_delta = abs(fmt.maxFrameRate() - target_fps)
      return (width_delta + height_delta, fps_delta, -fmt.maxFrameRate())

    best = min(formats, key=score)
    return best

  @Slot(QVideoFrame)
  def _on_video_frame(self, frame: QVideoFrame) -> None:
    if self._status not in {CameraStatus.READY, CameraStatus.RUNNING}:
      return

    bgr = qvideo_frame_to_bgr(frame)
    if bgr is None:
      return

    camera_frame = self._frame_buffer.push_frame(bgr)
    self._frame_count += 1
    self._frame_timestamps.append(time.monotonic())

    if self._status == CameraStatus.READY:
      self._set_status(CameraStatus.RUNNING)

    self.frame_ready.emit(camera_frame)

  @Slot()
  def _on_camera_error(self) -> None:
    if self._camera is None:
      return
    message = self._camera.errorString() or "Unknown camera error"
    logger.error("Camera error: %s", message)
    self._set_status(CameraStatus.ERROR)
    self.error_occurred.emit(message)
    self._schedule_recovery()

  def _schedule_recovery(self) -> None:
    if self._recovery_attempts >= self._max_recovery_attempts:
      logger.error("Camera recovery attempts exhausted.")
      return
    delay_ms = min(1000 * (2 ** self._recovery_attempts), 10000)
    self._recovery_attempts += 1
    logger.info("Scheduling camera recovery in %d ms (attempt %d)", delay_ms, self._recovery_attempts)
    self._recovery_timer.start(delay_ms)

  def _attempt_recovery(self) -> None:
    logger.info("Attempting camera recovery")
    self.stop()
    self.start()

  def _capture_fps(self) -> float:
    if len(self._frame_timestamps) < 2:
      return 0.0
    elapsed = self._frame_timestamps[-1] - self._frame_timestamps[0]
    if elapsed <= 0:
      return 0.0
    return (len(self._frame_timestamps) - 1) / elapsed

  def _emit_metrics(self) -> None:
    latest = self.latest_frame()
    metrics = CameraMetrics(
      capture_fps=self._capture_fps(),
      frame_count=self._frame_count,
      dropped_frames=self._frame_buffer.dropped_count,
      width=latest.width if latest else self._config.camera_width,
      height=latest.height if latest else self._config.camera_height,
    )
    self.metrics_updated.emit(metrics)

  def _set_status(self, status: CameraStatus) -> None:
    if self._status != status:
      self._status = status
      self.status_changed.emit(status)
