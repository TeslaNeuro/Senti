# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""YOLO backend selection: Ultralytics (PyTorch MPS) or yolo-mlx (Metal)."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from app.detection.yolo26_detector import resolve_model_path, resolve_yolo_device

logger = logging.getLogger(__name__)

BACKEND_ULTRALYTICS = "ultralytics"
BACKEND_MLX = "mlx"

# Official package: https://github.com/thewebAI/yolo-mlx  (import yolo26mlx)
# [tracking] is required for model.track(); [convert] is required for .pt → .npz.
MLX_REPO = "https://github.com/thewebAI/yolo-mlx"
MLX_INSTALL_HINT = 'pip install "yolo-mlx[tracking,convert]"'


def mlx_available() -> bool:
    """Return True when the yolo-mlx package (import yolo26mlx) is installed."""
    return importlib.util.find_spec("yolo26mlx") is not None


def resolve_yolo_backend(runtime: str, device: str) -> tuple[str, str]:
    """Pick detection backend and the label shown in the status bar.

    ``YOLO_RUNTIME=mlx`` or ``YOLO_DEVICE=mlx`` selects yolo-mlx (Metal).
    Anything else uses Ultralytics + PyTorch, with ``YOLO_DEVICE`` as today
    (``auto`` → MPS on Apple Silicon, else CPU).
    """
    runtime_key = runtime.strip().lower()
    device_key = device.strip().lower()
    wants_mlx = runtime_key == "mlx" or device_key == "mlx"
    if wants_mlx:
        return BACKEND_MLX, "mlx"
    pytorch_device = device_key if device_key != "mlx" else "auto"
    return BACKEND_ULTRALYTICS, resolve_yolo_device(pytorch_device)


def ensure_mlx_weights(model_name: str, *, project_root: Path | None = None) -> str:
    """Return a ``.npz`` path under ``models/``, converting from ``.pt`` if needed."""
    resolved = Path(resolve_model_path(model_name, project_root=project_root))
    if resolved.suffix.lower() in {".npz", ".safetensors"}:
        if not resolved.is_file():
            raise FileNotFoundError(
                f"MLX weights not found: {resolved}. "
                f"Convert a .pt checkpoint or place the .npz in models/."
            )
        return str(resolved)

    npz_path = resolved.with_suffix(".npz")
    if npz_path.is_file():
        return str(npz_path)

    pt_path = resolved if resolved.suffix.lower() == ".pt" else resolved.with_suffix(".pt")
    if not pt_path.is_file():
        _download_ultralytics_weights(str(pt_path))
    if not pt_path.is_file():
        raise FileNotFoundError(
            f"Cannot convert to MLX: {pt_path} is missing. "
            "Set YOLO_MODEL to a .pt file Senti can download, or place a .npz in models/."
        )

    logger.info("Converting YOLO weights for MLX: %s → %s", pt_path, npz_path)
    try:
        from yolo26mlx.converters.convert import convert_yolo26_weights, verify_conversion
    except ImportError as exc:
        raise RuntimeError(
            "yolo-mlx [convert] extras are required to turn .pt into .npz. "
            f"Install with: {MLX_INSTALL_HINT}  Official: {MLX_REPO}"
        ) from exc

    # Same API as: yolo-mlx converters convert model.pt -o model.npz --verify
    mlx_weights = convert_yolo26_weights(str(pt_path), str(npz_path), verbose=True)
    if not verify_conversion(str(pt_path), mlx_weights):
        logger.warning(
            "yolo-mlx weight verification reported issues for %s (npz kept). "
            "Re-run: yolo-mlx converters convert %s -o %s --verify",
            pt_path,
            pt_path,
            npz_path,
        )
    if not npz_path.is_file():
        raise RuntimeError(f"MLX conversion did not produce {npz_path}")
    return str(npz_path)


def _download_ultralytics_weights(model_path: str) -> None:
    """Let Ultralytics fetch a missing .pt into models/ before MLX conversion."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning("ultralytics is not installed; cannot download %s", model_path)
        return
    logger.info("Downloading Ultralytics weights to %s", model_path)
    YOLO(model_path)


def create_detector(config):
    """Build the configured YOLO detector (Ultralytics or yolo-mlx)."""
    from app.detection.yolo26_detector import Yolo26Detector

    backend, device = resolve_yolo_backend(config.yolo_runtime, config.yolo_device)
    if backend == BACKEND_MLX:
        if not mlx_available():
            raise RuntimeError(
                "YOLO_RUNTIME/YOLO_DEVICE is mlx but yolo-mlx is not installed. "
                f"Install with: {MLX_INSTALL_HINT}  "
                "Or set YOLO_RUNTIME=ultralytics and YOLO_DEVICE=mps for PyTorch MPS."
            )
        from app.detection.yolo_mlx_detector import YoloMlxDetector

        return YoloMlxDetector(config, device=device)
    return Yolo26Detector(config, device=device)
