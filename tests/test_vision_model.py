"""Tests for VLM prompt building."""

from app.detection.detector import Detection
from app.vision.vision_model import VisionAnalysisRequest, build_vlm_user_prompt
import numpy as np


def test_build_vlm_user_prompt_includes_detections() -> None:
    request = VisionAnalysisRequest(
        image_bgr=np.zeros((10, 10, 3), dtype=np.uint8),
        frame_id=1,
        detections=[
            Detection(0, "cup", 0.91, (1, 2, 3, 4), track_id=7),
        ],
        user_question="What is this?",
    )
    prompt = build_vlm_user_prompt(request)
    assert "Detected objects:" in prompt
    assert "#7 cup" in prompt
    assert "What is this?" in prompt


def test_build_vlm_user_prompt_marks_object_crop() -> None:
    detection = Detection(0, "keyboard", 0.88, (1, 2, 3, 4), track_id=2)
    request = VisionAnalysisRequest(
        image_bgr=np.zeros((10, 10, 3), dtype=np.uint8),
        frame_id=1,
        detections=[detection],
        focused_detection=detection,
        is_object_crop=True,
        user_question="What's that component?",
    )
    prompt = build_vlm_user_prompt(request)
    assert "cropped close-up" in prompt
    assert "#2 keyboard" in prompt
