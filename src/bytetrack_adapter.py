"""Adapter tích hợp bộ theo dõi ByteTrack cho các phát hiện Lửa (FIRE) và Khói (SMOKE).

Cung cấp ByteTrackAdapter:
- Sử dụng thuật toán ByteTrack chuẩn: liên kết hai bước (High confidence detections và Low confidence detections qua IoU/Kalman Filter).
- Tách biệt độc lập không gian ID theo dõi giữa các camera (mỗi CameraStreamWorker sở hữu 1 ByteTrackAdapter riêng).
- Hỗ trợ gán track_id ổn định, khôi phục ID khi đối tượng bị che khuất ngắn hạn (lost tracks).
- Cơ chế dự phòng (Fallback IoU Tracker) mượt mà nếu thiếu thư viện phụ trợ.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from src.temporal_verifier import DetectionBox

try:
    from types import SimpleNamespace
    from ultralytics.trackers.byte_tracker import BYTETracker

    HAS_BYTE_TRACKER = True
except Exception:
    HAS_BYTE_TRACKER = False


class ByteTrackAdapter:
    """Bộ theo dõi ByteTrack chuẩn cho từng luồng Camera."""

    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.7,
        frame_rate: int = 25,
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.tracker: Optional[Any] = None

        if HAS_BYTE_TRACKER:
            try:
                args = SimpleNamespace(
                    track_thresh=self.track_thresh,
                    track_buffer=self.track_buffer,
                    match_thresh=self.match_thresh,
                    frame_rate=self.frame_rate,
                )
                self.tracker = BYTETracker(args, frame_rate=self.frame_rate)
            except Exception:
                self.tracker = None

        # Fallback IoU tracker state nếu không nạp được ByteTracker
        self.fallback_next_id = 1
        self.fallback_tracks = {}  # tid -> (xyxy, label, last_frame)
        self.frame_idx = 0

    def reset(self):
        """Khởi tạo lại trạng thái của tracker."""
        self.frame_idx = 0
        if self.tracker is not None:
            try:
                args = SimpleNamespace(
                    track_thresh=self.track_thresh,
                    track_buffer=self.track_buffer,
                    match_thresh=self.match_thresh,
                    frame_rate=self.frame_rate,
                )
                self.tracker = BYTETracker(args, frame_rate=self.frame_rate)
            except Exception:
                pass
        self.fallback_next_id = 1
        self.fallback_tracks.clear()

    def update(self, boxes: List[DetectionBox], img_shape: Optional[Tuple[int, int]] = None) -> List[DetectionBox]:
        """Cập nhật các bounding box qua ByteTrack và gán track_id cho từng DetectionBox."""
        self.frame_idx += 1
        if not boxes:
            if self.tracker is not None:
                try:
                    dummy = np.empty((0, 6), dtype=np.float32)
                    self.tracker.update(dummy)
                except Exception:
                    pass
            return []

        # 1. Thử dùng ByteTracker của Ultralytics
        if self.tracker is not None:
            try:
                # Format: [x1, y1, x2, y2, score, cls_id]
                dets = []
                for b in boxes:
                    x1, y1, x2, y2 = b.xyxy
                    dets.append([float(x1), float(y1), float(x2), float(y2), float(b.confidence), float(b.cls_id)])
                dets_arr = np.array(dets, dtype=np.float32)
                
                # Gọi update BYTETracker
                online_targets = self.tracker.update(dets_arr)
                
                # Ánh xạ kết quả online_targets ngược lại vào boxes
                for t in online_targets:
                    # Target có tlbr (x1, y1, x2, y2), track_id, cls
                    t_tlbr = t.tlbr
                    tid = int(t.track_id)
                    t_cls = int(t.cls) if hasattr(t, "cls") else None

                    # Tìm box gốc có IoU cao nhất với t_tlbr
                    best_iou = 0.0
                    best_box = None
                    for b in boxes:
                        if t_cls is not None and b.cls_id != t_cls:
                            continue
                        # Tính IoU
                        xA = max(b.xyxy[0], t_tlbr[0])
                        yA = max(b.xyxy[1], t_tlbr[1])
                        xB = min(b.xyxy[2], t_tlbr[2])
                        yB = min(b.xyxy[3], t_tlbr[3])
                        interArea = max(0, xB - xA) * max(0, yB - yA)
                        if interArea > 0:
                            boxAArea = (b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1])
                            boxBArea = (t_tlbr[2] - t_tlbr[0]) * (t_tlbr[3] - t_tlbr[1])
                            iou = interArea / float(boxAArea + boxBArea - interArea)
                            if iou > best_iou:
                                best_iou = iou
                                best_box = b
                    if best_box is not None and best_iou > 0.3:
                        best_box.track_id = tid

                # Gán ID dự phòng cho các box chưa được match
                for b in boxes:
                    if b.track_id <= 0:
                        b.track_id = self.fallback_next_id
                        self.fallback_next_id += 1
                return boxes
            except Exception:
                pass

        # 2. Fallback IoU Tracker mượt mà
        used_ids = set()
        for b in boxes:
            best_iou = 0.0
            best_tid = None
            for tid, (last_xyxy, last_label, last_seen) in self.fallback_tracks.items():
                if tid in used_ids or last_label != b.label:
                    continue
                # Compute IoU
                xA = max(b.xyxy[0], last_xyxy[0])
                yA = max(b.xyxy[1], last_xyxy[1])
                xB = min(b.xyxy[2], last_xyxy[2])
                yB = min(b.xyxy[3], last_xyxy[3])
                inter = max(0, xB - xA) * max(0, yB - yA)
                if inter > 0:
                    areaA = (b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1])
                    areaB = (last_xyxy[2] - last_xyxy[0]) * (last_xyxy[3] - last_xyxy[1])
                    iou = inter / float(areaA + areaB - inter)
                    if iou > best_iou:
                        best_iou = iou
                        best_tid = tid

            if best_tid is not None and best_iou >= 0.25:
                b.track_id = best_tid
                used_ids.add(best_tid)
                self.fallback_tracks[best_tid] = (b.xyxy, b.label, self.frame_idx)
            else:
                new_id = self.fallback_next_id
                self.fallback_next_id += 1
                b.track_id = new_id
                used_ids.add(new_id)
                self.fallback_tracks[new_id] = (b.xyxy, b.label, self.frame_idx)

        # Xóa track cũ quá track_buffer frame
        stale = [tid for tid, (_, _, f_seen) in self.fallback_tracks.items() if (self.frame_idx - f_seen) > self.track_buffer]
        for tid in stale:
            del self.fallback_tracks[tid]

        return boxes

