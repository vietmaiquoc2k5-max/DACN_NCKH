"""Module xác minh cháy/khói theo chuỗi thời gian (Temporal Verifier).

Thực hiện Bước 4 của Đề cương NCKH & yêu cầu chống báo động giả:
1. Gán ID theo dõi đối tượng qua các khung hình liên tiếp (ByteTrack hoặc IoU Tracking).
2. Vùng loại trừ (Exclusion Zones / ROI) theo từng camera để bỏ qua vùng đèn hiệu cố định.
3. Cửa sổ trượt k/N khung hình theo từng Track ID độc lập và phân tách lớp FIRE / SMOKE:
   - Mức 1 (SUSPECT / Nghi ngờ): Xuất hiện >= k_suspect / N khung hình.
   - Mức 2 (FIRE / Cảnh báo xác nhận): Xuất hiện >= k_confirm / N khung hình và đạt ngưỡng biến động.
4. Phân tích biến động đặc trưng thời gian (Temporal Features: flicker, expansion, area growth, centroid speed)
   để loại trừ nguồn sáng tĩnh / phản chiếu cố định.
5. Cơ chế Cooldown nhân quả (causal timestamp) theo từng camera để tránh phát cảnh báo trùng lặp liên tục
   mà không làm mất sự kiện mới sau khi hết thời gian Cooldown.
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
    seen_frames: deque = field(default_factory=lambda: deque(maxlen=120))
    areas: deque = field(default_factory=lambda: deque(maxlen=20))
    confidences: deque = field(default_factory=lambda: deque(maxlen=20))
    feature_scores: deque = field(default_factory=lambda: deque(maxlen=20))
    hits: int = 0


def compute_dynamic_feature_score(feat: Any) -> float:
    """Tính điểm biến động đặc trưng thời gian tổng hợp từ TemporalFeatureResult."""
    if feat is None:
        return 0.0
    flicker = float(getattr(feat, "flicker_score", 0.0))
    smoke_exp = float(getattr(feat, "smoke_expansion_score", 0.0))
    area_growth = max(0.0, float(getattr(feat, "area_growth_rate", 0.0)))
    speed = float(getattr(feat, "centroid_speed", 0.0))
    return flicker + smoke_exp + area_growth + 0.1 * speed


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
        feature_threshold: float = 0.0,
        max_missing_frames: Optional[int] = None,
    ):
        self.window_size_n = max(2, int(window_size_n))
        self.confirm_frames_k = max(1, min(int(confirm_frames_k), self.window_size_n))
        self.suspect_frames_k = max(1, min(int(suspect_frames_k), self.confirm_frames_k))
        self.cooldown_sec = max(0.0, float(cooldown_sec))
        self.iou_match_threshold = iou_match_threshold
        # Mỗi vùng loại trừ là (x1_norm, y1_norm, x2_norm, y2_norm) trong [0, 1]
        self.exclusion_zones: List[Tuple[float, float, float, float]] = exclusion_zones or []
        self.feature_threshold = max(0.0, float(feature_threshold))
        self.max_missing_frames = int(max_missing_frames) if max_missing_frames is not None else self.window_size_n

        self.frame_history: deque = deque(maxlen=self.window_size_n)
        self.label_history: deque = deque(maxlen=self.window_size_n)
        self.conf_history: deque = deque(maxlen=self.window_size_n)

        self.tracks: Dict[int, TrackState] = {}
        self.next_track_id: int = 1
        self.frame_idx: int = 0
        self.last_alert_timestamp: Optional[float] = None
        self.current_level: str = "SAFE"  # "SAFE" | "SUSPECT" | "FIRE"

    def update_params(
        self,
        window_size_n: int,
        confirm_frames_k: int,
        suspect_frames_k: int,
        cooldown_sec: float,
        feature_threshold: Optional[float] = None,
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
        self.cooldown_sec = max(0.0, float(cooldown_sec))
        if feature_threshold is not None:
            self.feature_threshold = max(0.0, float(feature_threshold))

    def reset(self):
        """Đặt lại hoàn toàn trạng thái bộ xác minh khi khởi động lại hoặc kết nối lại camera."""
        self.frame_history.clear()
        self.label_history.clear()
        self.conf_history.clear()
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_idx = 0
        self.last_alert_timestamp = None
        self.current_level = "SAFE"

    def _is_in_exclusion_zone(self, norm_xywh: Tuple[float, float, float, float]) -> bool:
        """Kiểm tra tâm của bounding box có nằm trong vùng loại trừ ROI hay không."""
        xc, yc, _, _ = norm_xywh
        for x1, y1, x2, y2 in self.exclusion_zones:
            if x1 <= xc <= x2 and y1 <= yc <= y2:
                return True
        return False

    def _prune_track_windows(self) -> None:
        """Làm sạch các frame cũ ngoài cửa sổ N và xóa các track mất dấu quá max_missing_frames."""
        min_valid_frame = self.frame_idx - self.window_size_n + 1
        expire_limit = max(self.window_size_n, self.max_missing_frames)
        stale_tids = []
        for tid, trk in self.tracks.items():
            while trk.seen_frames and trk.seen_frames[0] < min_valid_frame:
                trk.seen_frames.popleft()
            if (self.frame_idx - trk.last_seen_frame) > expire_limit:
                stale_tids.append(tid)
        for tid in stale_tids:
            del self.tracks[tid]

    def _update_tracks(
        self,
        valid_boxes: List[DetectionBox],
        feature_map: Optional[Dict[int, Any]] = None,
    ) -> List[DetectionBox]:
        """Gán track_id cho từng đối tượng bằng thuật toán ghép cặp IoU nội bộ (chế độ legacy)."""
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
                if not trk.seen_frames or trk.seen_frames[-1] != self.frame_idx:
                    trk.seen_frames.append(self.frame_idx)
                trk.areas.append(area)
                trk.confidences.append(box.confidence)
                if feature_map and best_tid in feature_map:
                    trk.feature_scores.append(compute_dynamic_feature_score(feature_map[best_tid]))
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
                new_trk.seen_frames.append(self.frame_idx)
                new_trk.areas.append(area)
                new_trk.confidences.append(box.confidence)
                if feature_map and new_tid in feature_map:
                    new_trk.feature_scores.append(compute_dynamic_feature_score(feature_map[new_tid]))
                self.tracks[new_tid] = new_trk
                box.track_id = new_tid
                used_track_ids.add(new_tid)

        self._prune_track_windows()
        return valid_boxes

    def _record_external_tracks(
        self,
        valid_boxes: List[DetectionBox],
        feature_map: Optional[Dict[int, Any]] = None,
    ) -> List[DetectionBox]:
        """Ghi nhận và cập nhật lịch sử từng track khi track_id đã được gán bởi ByteTrack bên ngoài."""
        for box in valid_boxes:
            tid = box.track_id
            if tid <= 0:
                continue
            area = max(1, (box.xyxy[2] - box.xyxy[0]) * (box.xyxy[3] - box.xyxy[1]))
            # Đảm bảo không gộp nhầm FIRE và SMOKE nếu cùng track_id
            if tid in self.tracks and self.tracks[tid].label == box.label:
                trk = self.tracks[tid]
                trk.last_xyxy = box.xyxy
                trk.last_seen_frame = self.frame_idx
                if not trk.seen_frames or trk.seen_frames[-1] != self.frame_idx:
                    trk.seen_frames.append(self.frame_idx)
                trk.areas.append(area)
                trk.confidences.append(box.confidence)
                if feature_map and tid in feature_map:
                    trk.feature_scores.append(compute_dynamic_feature_score(feature_map[tid]))
                trk.hits += 1
            else:
                new_trk = TrackState(
                    track_id=tid,
                    label=box.label,
                    last_xyxy=box.xyxy,
                    last_seen_frame=self.frame_idx,
                    hits=1,
                )
                new_trk.seen_frames.append(self.frame_idx)
                new_trk.areas.append(area)
                new_trk.confidences.append(box.confidence)
                if feature_map and tid in feature_map:
                    new_trk.feature_scores.append(compute_dynamic_feature_score(feature_map[tid]))
                self.tracks[tid] = new_trk

        self._prune_track_windows()
        return valid_boxes

    def _track_temporal_score(self, trk: TrackState) -> float:
        """Tính điểm biến động thời gian của một track từ lịch sử feature hoặc biến thiên diện tích."""
        if trk.feature_scores:
            return float(sum(trk.feature_scores) / len(trk.feature_scores))
        if len(trk.areas) >= 3:
            mean_area = sum(trk.areas) / float(len(trk.areas))
            if mean_area > 0:
                var_area = sum((a - mean_area) ** 2 for a in trk.areas) / float(len(trk.areas))
                return float((var_area ** 0.5) / mean_area)
        return 0.0

    def _passes_feature_gate(self, trk: TrackState, has_external_features: bool) -> bool:
        """Kiểm tra track có vượt qua ngưỡng biến động thời gian để xác nhận cảnh báo hay không."""
        if self.feature_threshold <= 0.0:
            return True
        if trk.feature_scores:
            return self._track_temporal_score(trk) >= self.feature_threshold
        if has_external_features:
            # Nếu pipeline yêu cầu feature_threshold > 0 và có truyền feature_map nhưng track không có điểm
            return False
        if len(trk.areas) >= 3:
            return self._track_temporal_score(trk) >= self.feature_threshold
        return False

    def update(
        self,
        detections: List[DetectionBox],
        tracks_assigned: bool = False,
        feature_map: Optional[Dict[int, Any]] = None,
        timestamp_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Cập nhật kết quả phát hiện của khung hình hiện tại và trả về quyết định xác minh.

        Chỉ sử dụng thông tin từ các khung hình quá khứ và hiện tại (causal), tuyệt đối không
        dùng khung hình tương lai.

        Args:
            detections: Danh sách DetectionBox từ detector hoặc ByteTrack.
            tracks_assigned: Nếu True, sử dụng track_id do ByteTrack bên ngoài cung cấp
                thay vì chạy bộ ghép cặp IoU nội bộ.
            feature_map: Từ điển {track_id: TemporalFeatureResult} trích xuất tại khung hiện tại.
            timestamp_sec: Mốc thời gian (giây) của khung hình hiện tại để tính Cooldown chính xác.

        Returns:
            dict gồm:
              - valid_boxes: Danh sách DetectionBox sau khi lọc ROI và gán track_id
              - active_hits: Số khung phát hiện lớn nhất của một track trong cửa sổ N khung gần nhất
              - window_size: Kích thước cửa sổ N
              - confirm_k: Ngưỡng số khung xác nhận K
              - alert_level: "SAFE" | "SUSPECT" | "FIRE"
              - should_trigger_db_alert: True nếu vừa xác nhận sự kiện mới và đã qua thời gian Cooldown
              - primary_label: "FIRE" hoặc "SMOKE" hoặc ""
              - max_confidence: Độ tin cậy cao nhất của track kích hoạt hoặc trong khung hình
              - temporal_score: Điểm biến động đặc trưng thời gian cao nhất của các track đang hoạt động
              - confirmed_track_ids: Danh sách track_id đã đạt ngưỡng K/N và vượt qua feature gate
        """
        self.frame_idx += 1

        # 1. Lọc bỏ các box nằm trong vùng loại trừ ROI
        valid_boxes = [b for b in detections if not self._is_in_exclusion_zone(b.norm_xywh)]

        # 2. Gán Track ID theo thời gian (dùng ByteTrack nếu tracks_assigned=True)
        if tracks_assigned:
            valid_boxes = self._record_external_tracks(valid_boxes, feature_map=feature_map)
        else:
            valid_boxes = self._update_tracks(valid_boxes, feature_map=feature_map)

        has_detection = any((b.track_id > 0) for b in valid_boxes) if tracks_assigned else (len(valid_boxes) > 0)
        self.frame_history.append(1 if has_detection else 0)

        frame_max_conf = 0.0
        frame_primary_label = ""
        if valid_boxes:
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

        # 3. Đánh giá cửa sổ trượt k/N độc lập theo từng Track ID (không gộp chéo track hoặc gộp FIRE/SMOKE)
        has_ext_feat = feature_map is not None
        active_tracks = [trk for trk in self.tracks.values() if len(trk.seen_frames) > 0]
        active_hits = max((len(trk.seen_frames) for trk in active_tracks), default=0)

        confirmed_tracks: List[TrackState] = []
        suspect_tracks: List[TrackState] = []
        for trk in active_tracks:
            k_hits = len(trk.seen_frames)
            if k_hits >= self.confirm_frames_k and self._passes_feature_gate(trk, has_ext_feat):
                confirmed_tracks.append(trk)
            elif k_hits >= self.suspect_frames_k:
                suspect_tracks.append(trk)

        if confirmed_tracks:
            self.current_level = "FIRE"
            fire_conf_trks = [t for t in confirmed_tracks if t.label == "FIRE"]
            chosen_pool = fire_conf_trks if fire_conf_trks else confirmed_tracks
            best_trk = max(chosen_pool, key=lambda t: max(t.confidences) if t.confidences else 0.0)
            dominant_label = best_trk.label
            peak_conf = max(best_trk.confidences) if best_trk.confidences else frame_max_conf
            best_temporal_score = max((self._track_temporal_score(t) for t in confirmed_tracks), default=0.0)
        elif suspect_tracks:
            self.current_level = "SUSPECT"
            fire_susp_trks = [t for t in suspect_tracks if t.label == "FIRE"]
            chosen_pool = fire_susp_trks if fire_susp_trks else suspect_tracks
            best_trk = max(chosen_pool, key=lambda t: max(t.confidences) if t.confidences else 0.0)
            dominant_label = best_trk.label
            peak_conf = max(best_trk.confidences) if best_trk.confidences else frame_max_conf
            best_temporal_score = max((self._track_temporal_score(t) for t in suspect_tracks), default=0.0)
        else:
            self.current_level = "SAFE"
            dominant_label = frame_primary_label
            peak_conf = frame_max_conf
            best_temporal_score = max((self._track_temporal_score(t) for t in active_tracks), default=0.0)

        # 4. Kiểm tra điều kiện lưu sự kiện vào CSDL (Cooldown nhân quả, không làm mất sự kiện mới sau Cooldown)
        should_trigger_db_alert = False
        now = float(timestamp_sec) if timestamp_sec is not None else time.time()
        confirmed_in_current_frame = any(
            t.last_seen_frame == self.frame_idx for t in confirmed_tracks
        )
        if self.current_level == "FIRE" and confirmed_in_current_frame:
            if self.last_alert_timestamp is None or (now - self.last_alert_timestamp) >= self.cooldown_sec:
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
            "temporal_score": round(float(best_temporal_score), 4),
            "confirmed_track_id": confirmed_tracks[0].track_id if confirmed_tracks else None,
            "confirmed_track_ids": [t.track_id for t in confirmed_tracks],
        }
