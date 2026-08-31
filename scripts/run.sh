#!/usr/bin/env bash
# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

# This is a simple custom script to run the application. You may customise it however you want.
# Always adapt to your machine and requirements. Don't use blindly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt

# Optional: app also sets these automatically via app/qt_bootstrap.py
export QT_INFO_PLIST="$ROOT/resources/Info.plist"
PYSIDE_PLUGINS="$(python -c "import importlib.util; s=importlib.util.find_spec('PySide6'); import pathlib; print(pathlib.Path(s.origin).parent / 'Qt' / 'plugins')")"
export QT_PLUGIN_PATH="$PYSIDE_PLUGINS"
export QT_QPA_PLATFORM_PLUGIN_PATH="$PYSIDE_PLUGINS/platforms"

python -m app "$@"
