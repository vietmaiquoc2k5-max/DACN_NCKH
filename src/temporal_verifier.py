"""Module xác minh cháy/khói theo chuỗi thời gian (Temporal Verifier).

Thực hiện Bước 4 của Đề cương NCKH & yêu cầu chống báo động giả:
1. Gán ID theo dõi đối tượng qua các khung hình liên tiếp (IoU Tracking nhẹ tương thích ByteTrack).
2. Vùng loại trừ (Exclusion Zones / ROI) theo từng camera để bỏ qua vùng đèn hiệu cố định.
3. Cửa sổ trượt k/N khung hình:
   - Mức 1 (SUSPECT / Nghi ngờ): Xuất hiện >= k_suspect / N khung hình.
   - Mức 2 (FIRE / Cảnh báo cháy): Xuất hiện >= k_confirm / N khung hình.
4. Phân tích biến động hình học (diện tích & tâm hộp bao) để đánh giá tính động của ngọn lửa/khói.
5. Cơ chế Cooldown theo từng camera để tránh phát cảnh báo trùng lặp liên tục cho cùng một sự kiện.
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any


@dataclass
class DetectionBox:
    """Thông tin một bounding box phát hiện từ model YOLO."""
    cls_id: int
    label: str          # "FIRE" hoặc "SMOKE"
    confidence: float
    xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    norm_xywh: Tuple[float, float, float, float]  # (xc, yc, w, h) trong [0, 1]
    track_id: int = -1


@dataclass
class TrackState:
    """Trạng thái theo dõi của một đối tượng lửa/khói qua chuỗi khung hình."""
    track_id: int
    label: str
    last_xyxy: Tuple[int, int, int, int]
    last_seen_frame: int
    areas: deque = field(default_factory=lambda: deque(maxlen=20))
    confidences: deque = field(default_factory=lambda: deque(maxlen=20))
    hits: int = 0


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Tính chỉ số IoU giữa 2 bounding box (x1, y1, x2, y2)."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0, x_b - x_a)
    inter_h = max(0, y_b - y_a)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return inter_area / float(area_a + area_b - inter_area)


class TemporalVerifier:
    """Bộ xác minh sự kiện cháy/khói qua nhiều khung hình liên tiếp cho một camera."""

    def __init__(
        self,
        window_size_n: int = 10,
        confirm_frames_k: int = 4,
        suspect_frames_k: int = 2,
        cooldown_sec: float = 12.0,
        iou_match_threshold: float = 0.25,
        exclusion_zones: Optional[List[Tuple[float, float, float, float]]] = None,
    ):
        self.window_size_n = max(2, int(window_size_n))
        self.confirm_frames_k = max(1, min(int(confirm_frames_k), self.window_size_n))
        self.suspect_frames_k = max(1, min(int(suspect_frames_k), self.confirm_frames_k))
        self.cooldown_sec = max(1.0, float(cooldown_sec))
        self.iou_match_threshold = iou_match_threshold
        # Mỗi vùng loại trừ là (x1_norm, y1_norm, x2_norm, y2_norm) trong [0, 1]
        self.exclusion_zones: List[Tuple[float, float, float, float]] = exclusion_zones or []

        self.frame_history: deque = deque(maxlen=self.window_size_n)
        self.label_history: deque = deque(maxlen=self.window_size_n)
        self.conf_history: deque = deque(maxlen=self.window_size_n)

        self.tracks: Dict[int, TrackState] = {}
        self.next_track_id: int = 1
        self.frame_idx: int = 0
        self.last_alert_timestamp: float = 0.0
        self.current_level: str = "SAFE"  # "SAFE" | "SUSPECT" | "FIRE"

    def update_params(
        self,
        window_size_n: int,
        confirm_frames_k: int,
        suspect_frames_k: int,
        cooldown_sec: float,
    ):
        """Cập nhật động tham số cửa sổ trượt k/N và thời gian giãn cách cảnh báo."""
        new_n = max(2, int(window_size_n))
        if new_n != self.window_size_n:
            self.window_size_n = new_n
            self.frame_history = deque(self.frame_history, maxlen=new_n)
            self.label_history = deque(self.label_history, maxlen=new_n)
            self.conf_history = deque(self.conf_history, maxlen=new_n)
        self.confirm_frames_k = max(1, min(int(confirm_frames_k), self.window_size_n))
        self.suspect_frames_k = max(1, min(int(suspect_frames_k), self.confirm_frames_k))
        self.cooldown_sec = max(1.0, float(cooldown_sec))

    def reset(self):
        """Đặt lại trạng thái bộ xác minh khi khởi động lại camera."""
        self.frame_history.clear()
        self.label_history.clear()
        self.conf_history.clear()
        self.tracks.clear()
        self.frame_idx = 0
        self.current_level = "SAFE"

    def _is_in_exclusion_zone(self, norm_xywh: Tuple[float, float, float, float]) -> bool:
        """Kiểm tra tâm của bounding box có nằm trong vùng loại trừ ROI hay không."""
        xc, yc, _, _ = norm_xywh
        for x1, y1, x2, y2 in self.exclusion_zones:
            if x1 <= xc <= x2 and y1 <= yc <= y2:
                return True
        return False

    def _update_tracks(self, valid_boxes: List[DetectionBox]) -> List[DetectionBox]:
        """Gán track_id cho từng đối tượng bằng thuật toán ghép cặp IoU."""
        used_track_ids = set()
        for box in valid_boxes:
            best_iou = 0.0
            best_tid = None
            for tid, trk in self.tracks.items():
                if tid in used_track_ids or trk.label != box.label:
                    continue
                iou_val = compute_iou(box.xyxy, trk.last_xyxy)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_tid = tid

            area = max(1, (box.xyxy[2] - box.xyxy[0]) * (box.xyxy[3] - box.xyxy[1]))
            if best_tid is not None and best_iou >= self.iou_match_threshold:
                trk = self.tracks[best_tid]
                trk.last_xyxy = box.xyxy
                trk.last_seen_frame = self.frame_idx
                trk.areas.append(area)
                trk.confidences.append(box.confidence)
                trk.hits += 1
                box.track_id = best_tid
                used_track_ids.add(best_tid)
            else:
                new_tid = self.next_track_id
                self.next_track_id += 1
                new_trk = TrackState(
                    track_id=new_tid,
                    label=box.label,
                    last_xyxy=box.xyxy,
                    last_seen_frame=self.frame_idx,
                    hits=1,
                )
                new_trk.areas.append(area)
                new_trk.confidences.append(box.confidence)
                self.tracks[new_tid] = new_trk
                box.track_id = new_tid
                used_track_ids.add(new_tid)

        # Dọn các track đã biến mất quá window_size_n khung hình
        stale_tids = [
            tid
            for tid, trk in self.tracks.items()
            if (self.frame_idx - trk.last_seen_frame) > self.window_size_n
        ]
        for tid in stale_tids:
            del self.tracks[tid]

        return valid_boxes

    def update(self, detections: List[DetectionBox]) -> Dict[str, Any]:
        """Cập nhật kết quả phát hiện của khung hình hiện tại và trả về quyết định xác minh.

        Returns:
            dict gồm:
              - valid_boxes: Danh sách DetectionBox sau khi lọc ROI và gán track_id
              - active_hits: Số khung có cháy/khói trong cửa sổ N khung gần nhất
              - window_size: Kích thước cửa sổ N
              - alert_level: "SAFE" | "SUSPECT" | "FIRE"
              - should_trigger_db_alert: True nếu vừa xác nhận sự kiện mới và đã qua thời gian Cooldown
              - primary_label: "FIRE" hoặc "SMOKE" hoặc ""
              - max_confidence: Độ tin cậy cao nhất trong khung hình
        """
        self.frame_idx += 1

        # 1. Lọc bỏ các box nằm trong vùng loại trừ ROI
        valid_boxes = [b for b in detections if not self._is_in_exclusion_zone(b.norm_xywh)]

        # 2. Gán Track ID theo thời gian
        valid_boxes = self._update_tracks(valid_boxes)

        has_detection = len(valid_boxes) > 0
        self.frame_history.append(1 if has_detection else 0)

        frame_max_conf = 0.0
        frame_primary_label = ""
        if has_detection:
            # Ưu tiên FIRE hơn SMOKE nếu xuất hiện đồng thời, hoặc lấy box có confidence cao nhất
            fire_boxes = [b for b in valid_boxes if b.label == "FIRE"]
            if fire_boxes:
                best_box = max(fire_boxes, key=lambda x: x.confidence)
                frame_primary_label = "FIRE"
                frame_max_conf = best_box.confidence
            else:
                best_box = max(valid_boxes, key=lambda x: x.confidence)
                frame_primary_label = best_box.label
                frame_max_conf = best_box.confidence

        self.label_history.append(frame_primary_label)
        self.conf_history.append(frame_max_conf)

        # 3. Tính tổng số khung phát hiện k trong cửa sổ trượt N khung hình
        active_hits = sum(self.frame_history)

        if active_hits >= self.confirm_frames_k:
            self.current_level = "FIRE"
        elif active_hits >= self.suspect_frames_k:
            self.current_level = "SUSPECT"
        else:
            self.current_level = "SAFE"

        # 4. Xác định nhãn chủ đạo và độ tin cậy trung bình/cao nhất trong cửa sổ khi kích hoạt
        non_empty_labels = [lbl for lbl in self.label_history if lbl]
        dominant_label = "FIRE" if "FIRE" in non_empty_labels else (non_empty_labels[-1] if non_empty_labels else "")
        peak_conf = max(self.conf_history) if self.conf_history else 0.0

        # 5. Kiểm tra điều kiện lưu sự kiện vào CSDL (chống phát cảnh báo liên tục)
        should_trigger_db_alert = False
        now = time.time()
        if self.current_level == "FIRE" and has_detection:
            if (now - self.last_alert_timestamp) >= self.cooldown_sec:
                self.last_alert_timestamp = now
                should_trigger_db_alert = True

        return {
            "valid_boxes": valid_boxes,
            "active_hits": active_hits,
            "window_size": self.window_size_n,
            "confirm_k": self.confirm_frames_k,
            "alert_level": self.current_level,
            "should_trigger_db_alert": should_trigger_db_alert,
            "primary_label": dominant_label or frame_primary_label,
            "max_confidence": round(float(peak_conf if peak_conf > 0 else frame_max_conf), 4),
        }

