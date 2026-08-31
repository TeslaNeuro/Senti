"""Tests for YOLO26 detection parsing and helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.detection.detector import Detection
from app.detection.yolo26_detector import Yolo26Detector, resolve_yolo_device
from app.ui.overlay import draw_detections


def test_resolve_yolo_device_auto_cpu_when_no_mps() -> None:
    with patch("torch.backends.mps.is_available", return_value=False):
        assert resolve_yolo_device("auto") == "cpu"


def test_resolve_yolo_device_auto_mps_when_available() -> None:
    with patch("torch.backends.mps.is_available", return_value=True):
        assert resolve_yolo_device("auto") == "mps"


def test_resolve_yolo_device_explicit() -> None:
    assert resolve_yolo_device("cpu") == "cpu"


def test_parse_results_builds_detections() -> None:
    config = MagicMock(
        yolo_model="yolo26n.pt",
        yolo_confidence=0.5,
        yolo_image_size=640,
        yolo_device="cpu",
        tracking_enabled=False,
    )
    detector = Yolo26Detector(config)
    boxes = SimpleNamespace(
        xyxy=MagicMock(),
        conf=MagicMock(),
        cls=MagicMock(),
    )
    boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 20, 110, 120]])
    boxes.conf.cpu.return_value.numpy.return_value = np.array([0.93])
    boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
    result = SimpleNamespace(boxes=boxes, names={0: "cup"})
    detections = detector._parse_results([result])
    assert len(detections) == 1
    assert detections[0] == Detection(
        class_id=0,
        class_name="cup",
        confidence=0.93,
        bbox=(10, 20, 110, 120),
    )


def test_parse_results_includes_track_ids() -> None:
    config = MagicMock(
        yolo_model="yolo26n.pt",
        yolo_confidence=0.5,
        yolo_image_size=640,
        yolo_device="cpu",
        tracking_enabled=True,
        tracker_type="bytetrack.yaml",
    )
    detector = Yolo26Detector(config)
    boxes = SimpleNamespace(
        xyxy=MagicMock(),
        conf=MagicMock(),
        cls=MagicMock(),
        id=MagicMock(),
    )
    boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 20, 110, 120]])
    boxes.conf.cpu.return_value.numpy.return_value = np.array([0.93])
    boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
    boxes.id.cpu.return_value.numpy.return_value = np.array([7])
    result = SimpleNamespace(boxes=boxes, names={0: "cup"})
    detections = detector._parse_results([result])
    assert detections[0].track_id == 7


def test_draw_detections_returns_copy_with_boxes() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        Detection(class_id=0, class_name="pen", confidence=0.9, bbox=(10, 10, 40, 40)),
    ]
    output = draw_detections(frame, detections)
    assert output.shape == frame.shape
    assert not np.array_equal(output, frame)
