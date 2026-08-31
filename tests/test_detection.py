# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Tests for YOLO26 detection parsing and helpers."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.detection.detector import Detection
from app.detection.backend import create_detector, ensure_mlx_weights, resolve_yolo_backend
from app.detection.yolo26_detector import Yolo26Detector, resolve_model_path, resolve_yolo_device
from app.ui.overlay import draw_detections


def test_resolve_yolo_device_auto_cpu_when_no_mps() -> None:
    with patch("torch.backends.mps.is_available", return_value=False):
        assert resolve_yolo_device("auto") == "cpu"


def test_resolve_yolo_device_auto_mps_when_available() -> None:
    with patch("torch.backends.mps.is_available", return_value=True):
        assert resolve_yolo_device("auto") == "mps"


def test_resolve_yolo_device_explicit() -> None:
    assert resolve_yolo_device("cpu") == "cpu"


def test_resolve_yolo_backend_mlx_via_runtime() -> None:
    assert resolve_yolo_backend("mlx", "auto") == ("mlx", "mlx")


def test_resolve_yolo_backend_mlx_via_device() -> None:
    assert resolve_yolo_backend("auto", "mlx") == ("mlx", "mlx")


def test_resolve_yolo_backend_explicit_mps() -> None:
    assert resolve_yolo_backend("ultralytics", "mps") == ("ultralytics", "mps")


def test_resolve_yolo_backend_auto_uses_pytorch_device() -> None:
    with patch("torch.backends.mps.is_available", return_value=True):
        assert resolve_yolo_backend("auto", "auto") == ("ultralytics", "mps")


def test_ensure_mlx_weights_uses_existing_npz(tmp_path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    npz = models / "yolo26n.npz"
    npz.write_bytes(b"mlx")
    assert Path(ensure_mlx_weights("yolo26n.pt", project_root=tmp_path)) == npz


def test_create_detector_mlx_without_package() -> None:
    config = MagicMock(yolo_runtime="mlx", yolo_device="auto")
    with patch("app.detection.backend.mlx_available", return_value=False):
        with pytest.raises(RuntimeError, match="yolo-mlx is not installed"):
            create_detector(config)


def test_ensure_mlx_weights_accepts_npz_name(tmp_path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    npz = models / "yolo26s.npz"
    npz.write_bytes(b"mlx")
    assert Path(ensure_mlx_weights("yolo26s.npz", project_root=tmp_path)) == npz


def test_ensure_mlx_weights_uses_official_converter(tmp_path, monkeypatch) -> None:
    import sys
    from types import ModuleType

    models = tmp_path / "models"
    models.mkdir()
    pt = models / "yolo26n.pt"
    pt.write_bytes(b"pt")
    calls: dict[str, object] = {}

    def convert_yolo26_weights(pt_path: str, output_path: str, verbose: bool = True):
        Path(output_path).write_bytes(b"npz")
        calls["convert"] = (pt_path, output_path, verbose)
        return [("layer.weight", object())]

    def verify_conversion(pt_path: str, mlx_weights: list) -> bool:
        calls["verify"] = (pt_path, mlx_weights)
        return True

    convert_mod = ModuleType("yolo26mlx.converters.convert")
    convert_mod.convert_yolo26_weights = convert_yolo26_weights
    convert_mod.verify_conversion = verify_conversion
    converters = ModuleType("yolo26mlx.converters")
    converters.convert = convert_mod
    pkg = ModuleType("yolo26mlx")
    pkg.converters = converters
    monkeypatch.setitem(sys.modules, "yolo26mlx", pkg)
    monkeypatch.setitem(sys.modules, "yolo26mlx.converters", converters)
    monkeypatch.setitem(sys.modules, "yolo26mlx.converters.convert", convert_mod)

    result = Path(ensure_mlx_weights("yolo26n.pt", project_root=tmp_path))
    assert result == models / "yolo26n.npz"
    assert calls["convert"] == (str(pt), str(result), True)
    assert calls["verify"][0] == str(pt)


def test_resolve_model_path_uses_models_directory(tmp_path) -> None:
    models = tmp_path / "models"
    resolved = Path(resolve_model_path("yolo26s.pt", project_root=tmp_path))
    assert resolved == models / "yolo26s.pt"
    assert models.is_dir()
    assert not resolved.exists()


def test_resolve_model_path_strips_relative_directory(tmp_path) -> None:
    resolved = Path(resolve_model_path("models/yolo26n.pt", project_root=tmp_path))
    assert resolved == tmp_path / "models" / "yolo26n.pt"


def test_resolve_model_path_moves_legacy_root_weights(tmp_path) -> None:
    legacy = tmp_path / "yolo26s.pt"
    legacy.write_bytes(b"weights")
    resolved = Path(resolve_model_path("yolo26s.pt", project_root=tmp_path))
    assert resolved == tmp_path / "models" / "yolo26s.pt"
    assert resolved.read_bytes() == b"weights"
    assert not legacy.exists()


def test_resolve_model_path_removes_root_copy_when_models_has_file(tmp_path) -> None:
    target = tmp_path / "models"
    target.mkdir()
    (target / "yolo26n.pt").write_bytes(b"canonical")
    leftover = tmp_path / "yolo26n.pt"
    leftover.write_bytes(b"stale")
    resolved = Path(resolve_model_path("yolo26n.pt", project_root=tmp_path))
    assert resolved.read_bytes() == b"canonical"
    assert not leftover.exists()


def test_resolve_model_path_keeps_existing_absolute_file(tmp_path) -> None:
    custom = tmp_path / "custom" / "detector.pt"
    custom.parent.mkdir()
    custom.write_bytes(b"ok")
    assert resolve_model_path(str(custom), project_root=tmp_path) == str(custom)


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


def test_parse_results_numpy_boxes_without_torch() -> None:
    config = MagicMock(
        yolo_model="yolo26n.pt",
        yolo_confidence=0.5,
        yolo_image_size=640,
        yolo_device="mlx",
        tracking_enabled=False,
    )
    detector = Yolo26Detector(config)
    boxes = SimpleNamespace(
        xyxy=np.array([[10, 20, 110, 120]], dtype=float),
        conf=np.array([0.88], dtype=float),
        cls=np.array([0], dtype=float),
        id=None,
    )
    result = SimpleNamespace(boxes=boxes, names={0: "cup"})
    detections = detector._parse_results([result])
    assert detections[0].class_name == "cup"
    assert detections[0].confidence == pytest.approx(0.88)


def test_draw_detections_returns_copy_with_boxes() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        Detection(class_id=0, class_name="pen", confidence=0.9, bbox=(10, 10, 40, 40)),
    ]
    output = draw_detections(frame, detections)
    assert output.shape == frame.shape
    assert not np.array_equal(output, frame)
