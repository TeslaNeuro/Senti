# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""On-device OCR via EasyOCR, run off the UI thread."""

from app.ocr.engine import OcrAnalysis, OcrRequest, OcrResult, format_ocr_results
from app.ocr.worker import OcrThread

__all__ = [
    "OcrAnalysis",
    "OcrRequest",
    "OcrResult",
    "OcrThread",
    "format_ocr_results",
]
