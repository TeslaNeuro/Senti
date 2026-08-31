# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Application entry point."""

from app.qt_bootstrap import configure_qt_environment

configure_qt_environment()

from app.main import main

if __name__ == "__main__":
    raise SystemExit(main())
