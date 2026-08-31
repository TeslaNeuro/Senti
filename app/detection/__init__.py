"""Detection package."""

from app.detection.detector import Detection, DetectionMetrics, DetectionResult, ObjectDetector
from app.detection.yolo26_detector import Yolo26Detector

__all__ = [
    "Detection",
    "DetectionMetrics",
    "DetectionResult",
    "ObjectDetector",
    "Yolo26Detector",
]
