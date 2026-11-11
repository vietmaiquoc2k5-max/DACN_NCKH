"""Module điều hợp (adapter) cho thuật toán ByteTrack chính thức từ Ultralytics.

Module này bọc lớp `ultralytics.trackers.byte_tracker.BYTETracker` để cung cấp giao diện
chuẩn hóa, an toàn và tối ưu cho luồng xử lý khung hình camera giám sát cháy/khói:
- Nhận danh sách `DetectionBox` từ bộ phát hiện YOLO.
- Thực hiện liên kết theo dõi 2 giai đoạn (high-confidence và low-confidence).
- Quản lý trạng thái theo dõi độc lập theo từng luồng camera hoặc từng lớp (FIRE/SMOKE).
- Bảo toàn định dạng `DetectionBox` và gán `track_id` ổn định qua thời gian.
"""
from __future__ import annotations

import itertools
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.trackers.basetrack import BaseTrack
except ImportError as err:  # pragma: no cover
    BYTETracker = None  # type: ignore
    BaseTrack = None  # type: ignore

from src.config import ByteTrackSettings
from src.temporal_verifier import DetectionBox

LOGGER = logging.getLogger("bytetrack_adapter")


class _DetectionBatch:
    """Đối tượng đóng gói danh sách phát hiện đáp ứng giao diện duck-typing của Ultralytics BYTETracker.

    BYTETracker yêu cầu đầu vào có các thuộc tính `xywh`, `conf`, `cls` dạng NumPy array
    và hỗ trợ chỉ mục boolean / mảng thông qua phương thức `__getitem__`.
    """

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: Any) -> _DetectionBatch:
        return _DetectionBatch(self.xywh[mask], self.conf[mask], self.cls[mask])


class ByteTrackAdapter:
    """Bộ điều hợp tích hợp BYTETracker chính thức của Ultralytics cho hệ thống phát hiện cháy/khói.

    Attributes:
        settings: Cấu hình tham số ByteTrack (ngưỡng conf, buffer, match threshold).
        per_class: Nếu True, duy trì tracker riêng biệt cho từng lớp đối tượng (FIRE và SMOKE)
            để ngăn ngừa hiện tượng nhầm lẫn track khi khói bốc lên từ ngọn lửa.
        id_counter: Bộ sinh ID tăng dần đảm bảo tính duy nhất của track_id trên toàn bộ camera.
    """

    def __init__(
        self,
        settings: Optional[ByteTrackSettings] = None,
        per_class: bool = True,
        **kwargs: Any,
    ) -> None:
        """Khởi tạo ByteTrackAdapter với cấu hình được chỉ định.

        Args:
            settings: Cấu hình ByteTrackSettings (nếu None sẽ dùng giá trị mặc định).
            per_class: Có tách biệt tracker theo từng class ID hay không. Mặc định là True.
        """
        if BYTETracker is None:
            raise ImportError(
                "Không thể import BYTETracker từ 'ultralytics.trackers.byte_tracker'. "
                "Vui lòng kiểm tra cài đặt ultralytics và thư viện lap."
            )

        self.settings = settings or ByteTrackSettings()
        self.per_class = per_class
        self._id_counter = itertools.count(1)
        self._trackers: Dict[int, BYTETracker] = {}
        self._track_id_map: Dict[Tuple[int, int], int] = {}
        LOGGER.info(
            "Khởi tạo ByteTrackAdapter (high_thresh=%.2f, low_thresh=%.2f, new_thresh=%.2f, buffer=%d, per_class=%s)",
            self.settings.track_high_thresh,
            self.settings.track_low_thresh,
            self.settings.new_track_thresh,
            self.settings.track_buffer,
            self.per_class,
        )

    def _create_tracker_instance(self) -> BYTETracker:
        """Tạo một thực thể BYTETracker với bộ tham số chuẩn mà không làm reset BaseTrack._count toàn cục."""
        args = SimpleNamespace(
            track_high_thresh=float(self.settings.track_high_thresh),
            track_low_thresh=float(self.settings.track_low_thresh),
            new_track_thresh=float(self.settings.new_track_thresh),
            track_buffer=int(self.settings.track_buffer),
            match_thresh=float(self.settings.match_thresh),
            fuse_score=bool(self.settings.fuse_score),
        )
        saved_count = getattr(BaseTrack, "_count", 0) if BaseTrack is not None else 0
        tracker = BYTETracker(args)
        if BaseTrack is not None and getattr(BaseTrack, "_count", 0) < saved_count:
            BaseTrack._count = saved_count
        tracker._ids = itertools.count(1)
        return tracker

    def _get_tracker(self, cls_id: int) -> BYTETracker:
        """Lấy hoặc tạo mới tracker cho một class ID cụ thể."""
        key = cls_id if self.per_class else 0
        if key not in self._trackers:
            self._trackers[key] = self._create_tracker_instance()
        return self._trackers[key]

    def reset(self) -> None:
        """Xóa toàn bộ trạng thái tracking hiện tại và đặt lại bộ sinh ID.

        Được gọi khi camera khởi động lại, kết nối lại hoặc đổi nguồn video.
        """
        self._trackers.clear()
        self._track_id_map.clear()
        self._id_counter = itertools.count(1)
        LOGGER.debug("ByteTrackAdapter đã được reset hoàn toàn trạng thái.")

    def active_tracks_count(self) -> int:
        """Trả về tổng số track đang hoạt động (confirmed/tracked) trên tất cả các tracker."""
        count = 0
        for tracker in self._trackers.values():
            count += len([t for t in tracker.tracked_stracks if getattr(t, "is_activated", False)])
        return count

    def update(
        self,
        detections: List[DetectionBox],
        frame: Optional[np.ndarray] = None,
    ) -> List[DetectionBox]:
        """Cập nhật trạng thái theo dõi cho danh sách DetectionBox của khung hình hiện tại.

        Args:
            detections: Danh sách các DetectionBox từ bộ phát hiện YOLO.
            frame: Khung hình BGR hiện tại (tùy chọn, dùng cho ước lượng chuyển động camera GMC nếu có).

        Returns:
            Danh sách DetectionBox ban đầu với thuộc tính `track_id` được cập nhật:
            - `track_id > 0`: Đối tượng đã được ByteTrack gán ID xác nhận.
            - `track_id == -1`: Đối tượng chưa đủ điều kiện tạo track hoặc tạm thời chưa liên kết.
        """
        # Đảm bảo tất cả detection ban đầu có track_id mặc định là -1
        for box in detections:
            box.track_id = -1

        # Xử lý trường hợp không có detection trong khung hình
        if not detections:
            # Vẫn gọi update rỗng cho các tracker đang hoạt động để tiến bước thời gian Kalman
            for tracker in list(self._trackers.values()):
                try:
                    empty_batch = _DetectionBatch(
                        xywh=np.empty((0, 4), dtype=np.float32),
                        conf=np.empty((0,), dtype=np.float32),
                        cls=np.empty((0,), dtype=np.float32),
                    )
                    tracker.update(empty_batch, img=frame)
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning("Lỗi cập nhật rỗng trong BYTETracker: %s", exc)
            return detections

        try:
            # Nhóm các detection theo tracker (theo class nếu per_class=True, hoặc nhóm chung nếu per_class=False)
            grouped: Dict[int, List[Tuple[int, DetectionBox]]] = {}
            for orig_idx, box in enumerate(detections):
                cls_key = box.cls_id if self.per_class else 0
                grouped.setdefault(cls_key, []).append((orig_idx, box))

            # Cho những tracker nào đã tồn tại nhưng khung này không có detection, cập nhật rỗng để giữ thời gian
            active_keys = set(self._trackers.keys())
            for key in active_keys:
                if key not in grouped:
                    try:
                        empty_batch = _DetectionBatch(
                            xywh=np.empty((0, 4), dtype=np.float32),
                            conf=np.empty((0,), dtype=np.float32),
                            cls=np.empty((0,), dtype=np.float32),
                        )
                        self._trackers[key].update(empty_batch, img=frame)
                    except Exception as exc:  # pragma: no cover
                        LOGGER.warning("Lỗi cập nhật tracker rỗng cho class %d: %s", key, exc)

            # Cập nhật từng nhóm detection vào tracker tương ứng
            for key, indexed_boxes in grouped.items():
                tracker = self._get_tracker(key)

                xywh_list = []
                conf_list = []
                cls_list = []

                for _, box in indexed_boxes:
                    x1, y1, x2, y2 = box.xyxy
                    w = max(1.0, float(x2 - x1))
                    h = max(1.0, float(y2 - y1))
                    cx = float(x1) + w * 0.5
                    cy = float(y1) + h * 0.5
                    xywh_list.append([cx, cy, w, h])
                    conf_list.append(float(box.confidence))
                    cls_list.append(float(box.cls_id))

                batch = _DetectionBatch(
                    xywh=np.array(xywh_list, dtype=np.float32),
                    conf=np.array(conf_list, dtype=np.float32),
                    cls=np.array(cls_list, dtype=np.float32),
                )

                # Gọi ByteTrack update: trả về mảng shape (N, 8) với [x1, y1, x2, y2, track_id, score, cls, idx]
                tracks_out = tracker.update(batch, img=frame)

                if tracks_out is not None and len(tracks_out) > 0:
                    for row in tracks_out:
                        if len(row) < 8:
                            continue
                        raw_tid = int(row[4])
                        sub_idx = int(row[7])
                        if 0 <= sub_idx < len(indexed_boxes):
                            map_key = (key, raw_tid)
                            if map_key not in self._track_id_map:
                                self._track_id_map[map_key] = next(self._id_counter)
                            tid = self._track_id_map[map_key]
                            orig_idx, box_obj = indexed_boxes[sub_idx]
                            box_obj.track_id = tid

        except Exception as exc:
            LOGGER.error("Ngoại lệ khi thực thi ByteTrackAdapter.update: %s", exc, exc_info=True)

        return detections
