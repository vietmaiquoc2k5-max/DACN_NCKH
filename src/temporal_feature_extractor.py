"""Bộ trích xuất và phân tích biến động hình học của đối tượng lửa/khói (Temporal Feature Extractor).

Mục tiêu (Mục tiêu MT3):
- Theo dõi độ biến thiên diện tích (Area Delta), tỉ lệ co giãn (Aspect Ratio), chu vi và dịch chuyển tâm (Centroid Drift).
- Ngọn lửa thật có tính chất động: diện tích dao động co giãn nhấp nháy, tâm bốc lên trên.
- Ngược lại, nguồn sáng tĩnh (đèn đường, đèn xe đứng yên, biển LED) có diện tích và vị trí gần như không đổi.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import numpy as np


@dataclass
class GeometricHistory:
    """Lịch sử hình học của một Track ID."""
    track_id: int
    label: str
    areas: deque = field(default_factory=lambda: deque(maxlen=20))
    centroids: deque = field(default_factory=lambda: deque(maxlen=20))
    aspect_ratios: deque = field(default_factory=lambda: deque(maxlen=20))
    last_seen_frame: int = 0


class TemporalFeatureExtractor:
    """Trích xuất và đánh giá đặc trưng động học không-thời gian cho từng track."""

    def __init__(self, history_len: int = 15):
        self.history_len = history_len
        self.track_histories: Dict[int, GeometricHistory] = {}
        self.frame_idx = 0

    def reset(self):
        self.track_histories.clear()
        self.frame_idx = 0

    def update(self, track_id: int, label: str, xyxy: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Cập nhật tọa độ và tính toán các chỉ số biến động hình học."""
        self.frame_idx += 1
        x1, y1, x2, y2 = xyxy
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = float(w * h)
        cx = float(x1 + x2) / 2.0
        cy = float(y1 + y2) / 2.0
        ar = float(h) / float(w)

        if track_id not in self.track_histories:
            self.track_histories[track_id] = GeometricHistory(
                track_id=track_id,
                label=label,
                areas=deque(maxlen=self.history_len),
                centroids=deque(maxlen=self.history_len),
                aspect_ratios=deque(maxlen=self.history_len),
                last_seen_frame=self.frame_idx,
            )

        hist = self.track_histories[track_id]
        hist.areas.append(area)
        hist.centroids.append((cx, cy))
        hist.aspect_ratios.append(ar)
        hist.last_seen_frame = self.frame_idx

        # Dọn dẹp tracks cũ
        stale = [tid for tid, h_obj in self.track_histories.items() if (self.frame_idx - h_obj.last_seen_frame) > 45]
        for tid in stale:
            del self.track_histories[tid]

        # Phân tích biến thiên
        if len(hist.areas) >= 3:
            areas_arr = np.array(hist.areas, dtype=np.float32)
            area_std = float(np.std(areas_arr))
            area_mean = float(np.mean(areas_arr))
            area_variation = float(area_std / (area_mean + 1e-5))

            cents = np.array(hist.centroids, dtype=np.float32)
            centroid_drift = float(np.mean(np.linalg.norm(np.diff(cents, axis=0), axis=1))) if len(cents) > 1 else 0.0
        else:
            area_variation = 0.0
            centroid_drift = 0.0

        return {
            "area": area,
            "area_variation": round(area_variation, 4),
            "centroid_drift": round(centroid_drift, 2),
            "history_count": len(hist.areas),
        }

