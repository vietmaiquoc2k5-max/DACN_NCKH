"""Bộ kiểm thử đơn vị cho module ByteTrackAdapter (`src/bytetrack_adapter.py`).

Kiểm tra:
1. Duy trì Track ID ổn định khi bounding box di chuyển nhẹ.
2. Khôi phục Track ID khi đối tượng biến mất 1-2 khung hình (occlusion recovery).
3. Không tạo track mới từ detection có confidence quá thấp.
4. Khung hình không có detection không gây crash.
5. Khung hình không hợp lệ (None, kích thước bất thường) không gây lỗi.
6. Tách biệt lớp FIRE và SMOKE không bị nhầm lẫn track ID.
7. Reset xóa toàn bộ trạng thái tracking.
"""
import unittest
import numpy as np

from src.config import ByteTrackSettings
from src.temporal_verifier import DetectionBox
from src.bytetrack_adapter import ByteTrackAdapter


class TestByteTrackAdapter(unittest.TestCase):
    """Kiểm thử hoạt động của ByteTrackAdapter."""

    def setUp(self):
        self.settings = ByteTrackSettings(
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=False,
        )
        self.adapter = ByteTrackAdapter(settings=self.settings, per_class=True)

    def test_track_id_stability_slight_motion(self):
        """1. Kiểm tra Track ID ổn định khi bounding box di chuyển nhẹ qua các khung hình liên tiếp."""
        box_f1 = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.88,
            xyxy=(100, 100, 150, 150),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
        )
        res1 = self.adapter.update([box_f1])
        tid1 = res1[0].track_id
        self.assertGreater(tid1, 0, "Khung 1: Track ID phải là số dương hợp lệ.")

        # Khung 2: Bounding box di chuyển nhẹ 2 pixel
        box_f2 = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.85,
            xyxy=(102, 101, 152, 151),
            norm_xywh=(0.204, 0.202, 0.1, 0.1),
        )
        res2 = self.adapter.update([box_f2])
        tid2 = res2[0].track_id
        self.assertEqual(tid1, tid2, "Khung 2: Track ID phải được duy trì ổn định khi di chuyển nhẹ.")

    def test_track_recovery_short_occlusion(self):
        """2. Kiểm tra Track ID được duy trì khi detection bị mất trong thời gian ngắn (1-2 khung)."""
        box_f1 = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.89,
            xyxy=(120, 120, 180, 180),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
        )
        self.adapter.update([box_f1])

        box_f2 = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.86,
            xyxy=(122, 121, 182, 181),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
        )
        res2 = self.adapter.update([box_f2])
        initial_tid = res2[0].track_id

        # Khung 3: Khung hình trống (bị che khuất hoặc detector miss)
        res3 = self.adapter.update([])
        self.assertEqual(len(res3), 0, "Khung trống phải trả về danh sách rỗng.")

        # Khung 4: Đối tượng xuất hiện lại với độ dịch chuyển nhỏ
        box_f4 = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.84,
            xyxy=(125, 123, 185, 183),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
        )
        res4 = self.adapter.update([box_f4])
        self.assertEqual(
            res4[0].track_id,
            initial_tid,
            "Khung 4: Track ID phải được khôi phục sau khi bị mất tạm thời 1 khung.",
        )

    def test_low_confidence_rejection(self):
        """3. Kiểm tra detection có confidence quá thấp không được tạo track mới."""
        low_box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.05,  # Dưới cả track_low_thresh (0.1)
            xyxy=(200, 200, 260, 260),
            norm_xywh=(0.3, 0.3, 0.1, 0.1),
        )
        res = self.adapter.update([low_box])
        self.assertEqual(
            res[0].track_id,
            -1,
            "Detection confidence quá thấp (<0.1) không được gán track_id dương.",
        )

    def test_empty_detections_no_crash(self):
        """4. Kiểm tra cập nhật danh sách rỗng liên tục không gây ngoại lệ và đếm đúng số track."""
        for _ in range(5):
            out = self.adapter.update([])
            self.assertEqual(out, [])
        self.assertEqual(self.adapter.active_tracks_count(), 0)

    def test_invalid_frame_resilience(self):
        """5. Kiểm tra hệ thống không crash khi frame là None hoặc định dạng đặc biệt."""
        box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.85,
            xyxy=(100, 100, 140, 140),
            norm_xywh=(0.2, 0.2, 0.08, 0.08),
        )
        # Truyền frame=None
        res_none = self.adapter.update([box], frame=None)
        self.assertGreater(res_none[0].track_id, 0)

        # Truyền frame với kích thước nhỏ (1x1)
        res_weird = self.adapter.update([box], frame=np.zeros((1, 1, 3), dtype=np.uint8))
        self.assertGreater(res_weird[0].track_id, 0)

    def test_class_separation_fire_and_smoke(self):
        """6. Kiểm tra hai lớp FIRE và SMOKE xuất hiện đồng thời không bị gán nhầm track hoặc nhầm class."""
        fire_box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.90,
            xyxy=(100, 100, 150, 150),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
        )
        smoke_box = DetectionBox(
            cls_id=0,
            label="SMOKE",
            confidence=0.85,
            xyxy=(100, 50, 160, 110),
            norm_xywh=(0.2, 0.1, 0.1, 0.1),
        )

        res = self.adapter.update([fire_box, smoke_box])
        self.assertEqual(len(res), 2)
        fire_res = next(b for b in res if b.label == "FIRE")
        smoke_res = next(b for b in res if b.label == "SMOKE")

        self.assertGreater(fire_res.track_id, 0)
        self.assertGreater(smoke_res.track_id, 0)
        self.assertNotEqual(
            fire_res.track_id,
            smoke_res.track_id,
            "FIRE và SMOKE phải có track_id khác nhau.",
        )

    def test_reset_clears_state(self):
        """7. Kiểm tra reset() xóa sạch toàn bộ track và khởi tạo lại bộ đếm ID."""
        box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.92,
            xyxy=(80, 80, 140, 140),
            norm_xywh=(0.1, 0.1, 0.1, 0.1),
        )
        self.adapter.update([box])
        self.assertGreaterEqual(self.adapter.active_tracks_count(), 1)

        self.adapter.reset()
        self.assertEqual(self.adapter.active_tracks_count(), 0)

        # Sau khi reset, track đầu tiên tiếp tục bắt đầu từ ID 1
        res_after = self.adapter.update([box])
        self.assertEqual(res_after[0].track_id, 1)


if __name__ == "__main__":
    unittest.main()
