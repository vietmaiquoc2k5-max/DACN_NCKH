"""Temporal visual features for fire flicker and smoke expansion analysis."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class TemporalFeatureResult:
    """Normalized temporal descriptors for one tracked detection."""

    area_growth_rate: float
    centroid_speed: float
    roi_intensity_variance: float
    flicker_score: float
    smoke_expansion_score: float


@dataclass
class _Observation:
    """Internal feature history sample."""

    timestamp_sec: float
    area_ratio: float
    center_x: float
    center_y: float
    roi_mean_intensity: float


class TemporalFeatureExtractor:
    """Extract temporal geometry and ROI appearance features per track.

    The descriptors are heuristic inputs for temporal verification, not a
    standalone fire classifier. ROI intensity variance can be affected by
    camera auto-exposure, headlights and moving objects.
    """

    def __init__(self, history_size: int = 12) -> None:
        """Initialize bounded per-track feature histories.

        Args:
            history_size: Maximum number of observations retained per track.
        """
        if history_size < 3:
            raise ValueError("history_size phải >= 3")
        self.history_size = history_size
        self._history: dict[int, Deque[_Observation]] = {}

    def update(
        self,
        track_id: int,
        frame_bgr: np.ndarray,
        xyxy: tuple[int, int, int, int],
        timestamp_sec: float,
    ) -> TemporalFeatureResult:
        """Update a track and calculate temporal descriptors.

        Args:
            track_id: Stable tracker ID for the object.
            frame_bgr: Current BGR frame.
            xyxy: Bounding box in pixel coordinates.
            timestamp_sec: Video timestamp in seconds.

        Returns:
            Geometry and appearance descriptors for this track.
        """
        if frame_bgr is None or frame_bgr.ndim != 3:
            raise ValueError("frame_bgr phải là ảnh BGR hợp lệ")
        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = xyxy
        x1, x2 = max(0, min(width - 1, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height - 1, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding box rỗng hoặc nằm ngoài ảnh")

        roi = frame_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        area_ratio = ((x2 - x1) * (y2 - y1)) / float(width * height)
        center_x = ((x1 + x2) * 0.5) / width
        center_y = ((y1 + y2) * 0.5) / height
        mean_intensity = float(np.mean(gray))

        history = self._history.setdefault(track_id, deque(maxlen=self.history_size))
        previous = list(history)
        history.append(_Observation(
            timestamp_sec, area_ratio, center_x, center_y, mean_intensity
        ))

        area_growth = 0.0
        centroid_speed = 0.0
        flicker_score = 0.0
        smoke_expansion = 0.0
        if previous:
            last = previous[-1]
            dt = max(timestamp_sec - last.timestamp_sec, 1e-6)
            area_growth = (area_ratio - last.area_ratio) / dt
            dx = center_x - last.center_x
            dy = center_y - last.center_y
            centroid_speed = float(np.hypot(dx, dy) / dt)

        if len(history) >= 3:
            intensities = np.asarray(
                [sample.roi_mean_intensity for sample in history],
                dtype=np.float32,
            )
            # Coefficient of variation is scale-normalized and approximates
            # temporal brightness fluctuation inside the tracked ROI.
            flicker_score = float(
                np.std(intensities) / max(float(np.mean(intensities)), 1.0)
            )
            areas = np.asarray(
                [sample.area_ratio for sample in history], dtype=np.float32
            )
            positive_growth = np.maximum(np.diff(areas), 0.0)
            smoke_expansion = float(np.sum(positive_growth))

        return TemporalFeatureResult(
            area_growth_rate=float(area_growth),
            centroid_speed=float(centroid_speed),
            roi_intensity_variance=float(np.var(
                [sample.roi_mean_intensity for sample in history]
            )) if len(history) > 1 else 0.0,
            flicker_score=flicker_score,
            smoke_expansion_score=smoke_expansion,
        )

    def reset_track(self, track_id: int) -> None:
        """Clear history for a track that has expired.

        Args:
            track_id: Tracker ID to remove.
        """
        self._history.pop(track_id, None)

    def reset(self) -> None:
        """Clear all track histories, e.g. when a camera restarts."""
        self._history.clear()
