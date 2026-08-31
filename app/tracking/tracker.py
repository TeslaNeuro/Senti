"""Object tracking state and change detection."""

from __future__ import annotations

from dataclasses import dataclass

from app.detection.detector import Detection


@dataclass(frozen=True)
class TrackUpdate:
    """Tracks that appeared, disappeared, or remain active between frames."""

    new_track_ids: tuple[int, ...] = ()
    lost_track_ids: tuple[int, ...] = ()
    active_track_ids: tuple[int, ...] = ()


@dataclass
class TrackedObject:
    """In-memory state for a single tracked object."""

    track_id: int
    class_name: str
    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]
    frames_seen: int = 1
    frames_missing: int = 0

    @classmethod
    def from_detection(cls, detection: Detection) -> TrackedObject:
        if detection.track_id is None:
            raise ValueError("Detection has no track_id")
        return cls(
            track_id=detection.track_id,
            class_name=detection.class_name,
            class_id=detection.class_id,
            confidence=detection.confidence,
            bbox=detection.bbox,
        )


class TrackMonitor:
    """Maintains track identities and reports new/lost objects between frames."""

    def __init__(self) -> None:
        self._active: dict[int, TrackedObject] = {}
        self._last_update: TrackUpdate = TrackUpdate()

    @property
    def active_objects(self) -> dict[int, TrackedObject]:
        return dict(self._active)

    @property
    def last_update(self) -> TrackUpdate:
        return self._last_update

    def update(self, detections: list[Detection]) -> TrackUpdate:
        previous_ids = set(self._active.keys())
        current_ids: set[int] = set()

        for detection in detections:
            if detection.track_id is None:
                continue
            track_id = detection.track_id
            current_ids.add(track_id)
            if track_id in self._active:
                existing = self._active[track_id]
                self._active[track_id] = TrackedObject(
                    track_id=track_id,
                    class_name=detection.class_name,
                    class_id=detection.class_id,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    frames_seen=existing.frames_seen + 1,
                    frames_missing=0,
                )
            else:
                self._active[track_id] = TrackedObject.from_detection(detection)

        lost_ids = previous_ids - current_ids
        for track_id in lost_ids:
            del self._active[track_id]

        new_ids = current_ids - previous_ids
        self._last_update = TrackUpdate(
            new_track_ids=tuple(sorted(new_ids)),
            lost_track_ids=tuple(sorted(lost_ids)),
            active_track_ids=tuple(sorted(current_ids)),
        )
        return self._last_update

    def reset(self) -> None:
        self._active.clear()
        self._last_update = TrackUpdate()
