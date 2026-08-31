# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Tests for VLM scheduler."""

import numpy as np

from app.vision.vision_model import VisionAnalysisRequest
from app.vision.vlm_scheduler import VlmRequestDecision, VlmScheduler


def _request(frame_id: int = 1, manual: bool = False) -> VisionAnalysisRequest:
    return VisionAnalysisRequest(
        image_bgr=np.zeros((32, 32, 3), dtype=np.uint8),
        frame_id=frame_id,
        manual=manual,
    )


def test_scheduler_accepts_manual_request() -> None:
    scheduler = VlmScheduler(cooldown_seconds=5.0)
    result = scheduler.schedule(_request(manual=True))
    assert result.decision == VlmRequestDecision.ACCEPT


def test_scheduler_blocks_duplicate_requests() -> None:
    scheduler = VlmScheduler(cooldown_seconds=0.0)
    scheduler.begin(1)
    result = scheduler.schedule(_request(manual=True))
    assert result.decision == VlmRequestDecision.REJECT_BUSY


def test_scheduler_enforces_cooldown_for_automatic_requests() -> None:
    scheduler = VlmScheduler(cooldown_seconds=1.0)
    scheduler.begin(1)
    scheduler.complete(1)
    result = scheduler.schedule(_request(manual=False))
    assert result.decision == VlmRequestDecision.REJECT_COOLDOWN


def test_manual_request_overrides_cooldown() -> None:
    scheduler = VlmScheduler(cooldown_seconds=10.0)
    scheduler.begin(1)
    scheduler.complete(1)
    result = scheduler.schedule(_request(manual=True))
    assert result.decision == VlmRequestDecision.ACCEPT


def test_scheduler_rejects_stale_automatic_frame() -> None:
    scheduler = VlmScheduler(cooldown_seconds=0.0)
    scheduler.update_latest_frame(10)
    result = scheduler.schedule(_request(frame_id=3, manual=False))
    assert result.decision == VlmRequestDecision.REJECT_STALE


def test_scheduler_rejects_duplicate_automatic_frame() -> None:
    scheduler = VlmScheduler(cooldown_seconds=0.0)
    scheduler.begin(7)
    scheduler.complete(7)
    result = scheduler.schedule(_request(frame_id=7, manual=False))
    assert result.decision == VlmRequestDecision.REJECT_DUPLICATE
