"""Bộ kiểm thử cho TemporalVerifier khi tích hợp ByteTrack (`tests/test_temporal_verifier_bytetrack.py`).

Kiểm tra:
1. Không ghi đè track_id khi `tracks_assigned=True`.
2. Cơ chế cửa sổ trượt k/N (SAFE -> SUSPECT -> FIRE).
3. Lọc vùng loại trừ (Exclusion Zones / ROI).
4. Cooldown ngăn cảnh báo trùng lặp.
5. Ưu tiên nhãn FIRE so với SMOKE khi phát hiện đồng thời.
6. Tính tương thích ngược với luồng không dùng tracks_assigned.
"""
import time
import unittest

from src.temporal_verifier import TemporalVerifier, DetectionBox


class TestTemporalVerifierByteTrack(unittest.TestCase):
    """Kiểm thử xác minh theo thời gian kết hợp ByteTrack."""

    def setUp(self):
        self.verifier = TemporalVerifier(
            window_size_n=5,
            confirm_frames_k=3,
            suspect_frames_k=2,
            cooldown_sec=2.0,
            exclusion_zones=[(0.7, 0.7, 1.0, 1.0)],  # Góc dưới phải
        )

    def test_preserves_bytetrack_id(self):
        """1. Kiểm tra TemporalVerifier không ghi đè track_id của ByteTrack khi tracks_assigned=True."""
        box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.88,
            xyxy=(100, 100, 150, 150),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
            track_id=101,  # ID do ByteTrack cấp
        )
        res = self.verifier.update([box], tracks_assigned=True)
        self.assertEqual(len(res["valid_boxes"]), 1)
        self.assertEqual(
            res["valid_boxes"][0].track_id,
            101,
            "track_id 101 phải được giữ nguyên vẹn, không bị ghi đè.",
        )

    def test_k_of_n_sliding_window(self):
        """2. Kiểm tra cửa sổ trượt k/N xác nhận đúng các mức SAFE -> SUSPECT -> FIRE."""
        # Khung 1: 1 hit -> SAFE (1/5)
        box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.85,
            xyxy=(100, 100, 150, 150),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
            track_id=1,
        )
        r1 = self.verifier.update([box], tracks_assigned=True)
        self.assertEqual(r1["alert_level"], "SAFE")
        self.assertEqual(r1["active_hits"], 1)

        # Khung 2: 2 hit -> SUSPECT (2/5)
        r2 = self.verifier.update([box], tracks_assigned=True)
        self.assertEqual(r2["alert_level"], "SUSPECT")
        self.assertEqual(r2["active_hits"], 2)

        # Khung 3: 3 hit -> FIRE (3/5)
        r3 = self.verifier.update([box], tracks_assigned=True)
        self.assertEqual(r3["alert_level"], "FIRE")
        self.assertEqual(r3["active_hits"], 3)
        self.assertTrue(r3["should_trigger_db_alert"])

    def test_exclusion_zone_filtering(self):
        """3. Kiểm tra vùng loại trừ ROI loại bỏ đúng các detection nằm trong tọa độ cấm."""
        inside_roi_box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.95,
            xyxy=(750, 750, 850, 850),
            norm_xywh=(0.80, 0.80, 0.1, 0.1),  # Nằm trong ROI [(0.7, 0.7, 1.0, 1.0)]
            track_id=99,
        )
        outside_roi_box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.85,
            xyxy=(100, 100, 150, 150),
            norm_xywh=(0.20, 0.20, 0.1, 0.1),
            track_id=100,
        )

        res = self.verifier.update([inside_roi_box, outside_roi_box], tracks_assigned=True)
        self.assertEqual(len(res["valid_boxes"]), 1)
        self.assertEqual(res["valid_boxes"][0].track_id, 100)

    def test_alert_cooldown_behavior(self):
        """4. Kiểm tra Cooldown ngăn phát cảnh báo trùng lặp liên tục cho cùng một sự kiện."""
        box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.88,
            xyxy=(100, 100, 150, 150),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
            track_id=5,
        )
        # Kích hoạt đủ 3 khung để đạt FIRE
        self.verifier.update([box], tracks_assigned=True)
        self.verifier.update([box], tracks_assigned=True)
        r3 = self.verifier.update([box], tracks_assigned=True)
        self.assertTrue(r3["should_trigger_db_alert"], "Lần đầu đạt FIRE phải trigger alert.")

        # Khung ngay tiếp theo (chưa qua cooldown_sec = 2.0s)
        r4 = self.verifier.update([box], tracks_assigned=True)
        self.assertEqual(r4["alert_level"], "FIRE")
        self.assertFalse(r4["should_trigger_db_alert"], "Chưa hết Cooldown không được trigger lại alert.")

    def test_fire_priority_over_smoke(self):
        """5. Kiểm tra ưu tiên nhãn FIRE hơn SMOKE khi cả hai xuất hiện đồng thời."""
        fire_box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.75,
            xyxy=(100, 100, 150, 150),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
            track_id=1,
        )
        smoke_box = DetectionBox(
            cls_id=0,
            label="SMOKE",
            confidence=0.85,
            xyxy=(200, 200, 250, 250),
            norm_xywh=(0.4, 0.4, 0.1, 0.1),
            track_id=2,
        )
        res = self.verifier.update([smoke_box, fire_box], tracks_assigned=True)
        self.assertEqual(res["primary_label"], "FIRE")

    def test_backward_compatibility_without_tracks_assigned(self):
        """6. Kiểm tra gọi update không truyền tracks_assigned vẫn hoạt động tự gán track bằng IoU."""
        box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.85,
            xyxy=(100, 100, 150, 150),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
            track_id=-1,
        )
        res = self.verifier.update([box])
        self.assertGreater(res["valid_boxes"][0].track_id, 0)


if __name__ == "__main__":
    unittest.main()
