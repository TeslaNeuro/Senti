# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Configure Qt environment before any Qt modules are imported."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _pyside6_root() -> Path | None:
    spec = importlib.util.find_spec("PySide6")
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def configure_qt_environment() -> None:
    """Ensure Qt can locate bundled platform plugins (fixes cocoa plugin errors)."""
    project_root = Path(__file__).resolve().parent.parent
    info_plist = project_root / "resources" / "Info.plist"
    if info_plist.is_file():
        os.environ.setdefault("QT_INFO_PLIST", str(info_plist))

    pyside_root = _pyside6_root()
    if pyside_root is None:
        return

    qt_plugins = pyside_root / "Qt" / "plugins"
    qt_platforms = qt_plugins / "platforms"
    if qt_platforms.is_dir():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(qt_platforms))
    if qt_plugins.is_dir():
        os.environ.setdefault("QT_PLUGIN_PATH", str(qt_plugins))
