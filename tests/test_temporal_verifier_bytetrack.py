"""Bộ kiểm thử toàn diện cho TemporalVerifier khi tích hợp ByteTrack và Temporal Features (`tests/test_temporal_verifier_bytetrack.py`).

Kiểm tra:
1. Không ghi đè track_id khi `tracks_assigned=True`.
2. Cơ chế cửa sổ trượt k/N theo từng track_id (SAFE -> SUSPECT -> FIRE).
3. Một khung hình đơn lẻ không bao giờ kích hoạt FIRE khi K > 1.
4. Không gộp hits của các track_id rời rạc (nhiễu lốm đốm) thành FIRE.
5. Không gộp hits giữa class FIRE và SMOKE.
6. Khả năng hồi phục khi mất dấu ngắn hạn trong cửa sổ N vs hết hạn sau max_missing_frames.
7. Lọc vùng loại trừ (Exclusion Zones / ROI).
8. Cooldown ngăn cảnh báo trùng lặp nhưng cho phép cảnh báo sự kiện mới sau khi hết cooldown_sec.
9. Temporal features (`feature_map` + `feature_threshold`) thực sự ảnh hưởng tới điểm xác nhận và quyết định cảnh báo.
10. Ưu tiên nhãn FIRE so với SMOKE khi phát hiện đồng thời.
11. Tính tương thích ngược với luồng không dùng tracks_assigned.
"""
import unittest

from src.temporal_features import TemporalFeatureResult
from src.temporal_verifier import DetectionBox, TemporalVerifier, compute_dynamic_feature_score


class TestTemporalVerifierByteTrack(unittest.TestCase):
    """Kiểm thử xác minh theo thời gian kết hợp ByteTrack và đặc trưng động học."""

    def setUp(self):
        self.verifier = TemporalVerifier(
            window_size_n=5,
            confirm_frames_k=3,
            suspect_frames_k=2,
            cooldown_sec=2.0,
            exclusion_zones=[(0.7, 0.7, 1.0, 1.0)],  # Góc dưới phải
            feature_threshold=0.0,
            max_missing_frames=5,
        )

    def test_preserves_bytetrack_id(self):
        """1. Kiểm tra TemporalVerifier không ghi đè track_id của ByteTrack khi tracks_assigned=True."""
        box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.88,
            xyxy=(100, 100, 150, 150),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
            track_id=101,
        )
        res = self.verifier.update([box], tracks_assigned=True, timestamp_sec=1.0)
        self.assertEqual(len(res["valid_boxes"]), 1)
        self.assertEqual(
            res["valid_boxes"][0].track_id,
            101,
            "track_id 101 phải được giữ nguyên vẹn, không bị ghi đè.",
        )

    def test_single_frame_never_triggers_fire_when_k_gt_1(self):
        """2. Một khung hình đơn lẻ không bao giờ kích hoạt FIRE khi K > 1."""
        box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.99,
            xyxy=(100, 100, 150, 150),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
            track_id=7,
        )
        res = self.verifier.update([box], tracks_assigned=True, timestamp_sec=0.5)
        self.assertEqual(res["alert_level"], "SAFE")
        self.assertFalse(res["should_trigger_db_alert"])

    def test_k_of_n_sliding_window(self):
        """3. Kiểm tra cửa sổ trượt k/N xác nhận đúng các mức SAFE -> SUSPECT -> FIRE trên cùng track_id."""
        box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.85,
            xyxy=(100, 100, 150, 150),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
            track_id=1,
        )
        r1 = self.verifier.update([box], tracks_assigned=True, timestamp_sec=0.2)
        self.assertEqual(r1["alert_level"], "SAFE")
        self.assertEqual(r1["active_hits"], 1)

        r2 = self.verifier.update([box], tracks_assigned=True, timestamp_sec=0.4)
        self.assertEqual(r2["alert_level"], "SUSPECT")
        self.assertEqual(r2["active_hits"], 2)

        r3 = self.verifier.update([box], tracks_assigned=True, timestamp_sec=0.6)
        self.assertEqual(r3["alert_level"], "FIRE")
        self.assertEqual(r3["active_hits"], 3)
        self.assertEqual(r3["confirmed_track_id"], 1)
        self.assertTrue(r3["should_trigger_db_alert"])

    def test_unrelated_track_ids_do_not_merge_hits(self):
        """4. Các detection rời rạc mang track_id khác nhau ở các khung khác nhau không được cộng dồn hits thành FIRE."""
        for idx, tid in enumerate([10, 20, 30, 40], start=1):
            box = DetectionBox(
                cls_id=1,
                label="FIRE",
                confidence=0.90,
                xyxy=(50 * idx, 50 * idx, 50 * idx + 40, 50 * idx + 40),
                norm_xywh=(0.1 * idx, 0.1 * idx, 0.05, 0.05),
                track_id=tid,
            )
            res = self.verifier.update([box], tracks_assigned=True, timestamp_sec=float(idx))
            self.assertEqual(res["active_hits"], 1)
            self.assertEqual(res["alert_level"], "SAFE")
            self.assertFalse(res["should_trigger_db_alert"])

    def test_fire_and_smoke_do_not_merge_hits(self):
        """5. Tách biệt class FIRE và SMOKE: 2 frame FIRE + 1 frame SMOKE không được gộp thành 3 hits để báo FIRE."""
        fire_box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.85,
            xyxy=(100, 100, 150, 150),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
            track_id=11,
        )
        smoke_box = DetectionBox(
            cls_id=0,
            label="SMOKE",
            confidence=0.85,
            xyxy=(100, 100, 150, 150),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
            track_id=22,
        )
        self.verifier.update([fire_box], tracks_assigned=True, timestamp_sec=1.0)
        self.verifier.update([fire_box], tracks_assigned=True, timestamp_sec=1.2)
        res = self.verifier.update([smoke_box], tracks_assigned=True, timestamp_sec=1.4)
        self.assertNotEqual(res["alert_level"], "FIRE")
        self.assertFalse(res["should_trigger_db_alert"])

    def test_short_missing_frames_recovery_within_window_n(self):
        """6. Mất dấu ngắn hạn 1-2 frame trong cửa sổ N=5 vẫn cho phép track đạt K=3 khi xuất hiện lại."""
        box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.88,
            xyxy=(100, 100, 150, 150),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
            track_id=55,
        )
        # Frame 1: hit 1
        self.verifier.update([box], tracks_assigned=True, timestamp_sec=1.0)
        # Frame 2: miss
        self.verifier.update([], tracks_assigned=True, timestamp_sec=1.2)
        # Frame 3: hit 2 -> SUSPECT
        r3 = self.verifier.update([box], tracks_assigned=True, timestamp_sec=1.4)
        self.assertEqual(r3["alert_level"], "SUSPECT")
        # Frame 4: miss
        self.verifier.update([], tracks_assigned=True, timestamp_sec=1.6)
        # Frame 5: hit 3 within window [1..5] -> FIRE
        r5 = self.verifier.update([box], tracks_assigned=True, timestamp_sec=1.8)
        self.assertEqual(r5["alert_level"], "FIRE")
        self.assertEqual(r5["active_hits"], 3)

    def test_exclusion_zone_filtering(self):
        """7. Kiểm tra vùng loại trừ ROI loại bỏ đúng các detection nằm trong tọa độ cấm."""
        inside_roi_box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.95,
            xyxy=(750, 750, 850, 850),
            norm_xywh=(0.80, 0.80, 0.1, 0.1),
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

        res = self.verifier.update([inside_roi_box, outside_roi_box], tracks_assigned=True, timestamp_sec=1.0)
        self.assertEqual(len(res["valid_boxes"]), 1)
        self.assertEqual(res["valid_boxes"][0].track_id, 100)

    def test_alert_cooldown_and_retrigger_after_cooldown_expires(self):
        """8. Cooldown chặn cảnh báo lặp trong cooldown_sec nhưng cho phép sự kiện mới sau khi hết cooldown."""
        box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.88,
            xyxy=(100, 100, 150, 150),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
            track_id=5,
        )
        self.verifier.update([box], tracks_assigned=True, timestamp_sec=0.2)
        self.verifier.update([box], tracks_assigned=True, timestamp_sec=0.4)
        r3 = self.verifier.update([box], tracks_assigned=True, timestamp_sec=0.6)
        self.assertTrue(r3["should_trigger_db_alert"], "Lần đầu đạt FIRE ở t=0.6s phải trigger alert.")

        # Trong thời gian cooldown (t=1.5s < 0.6 + 2.0)
        r4 = self.verifier.update([box], tracks_assigned=True, timestamp_sec=1.5)
        self.assertEqual(r4["alert_level"], "FIRE")
        self.assertFalse(r4["should_trigger_db_alert"], "Chưa hết Cooldown không được trigger lại alert.")

        # Sau khi hết thời gian cooldown (t=3.0s >= 0.6 + 2.0)
        r5 = self.verifier.update([box], tracks_assigned=True, timestamp_sec=3.0)
        self.assertEqual(r5["alert_level"], "FIRE")
        self.assertTrue(r5["should_trigger_db_alert"], "Sau khi hết Cooldown phải cho phép trigger alert mới.")

    def test_temporal_features_affect_confirmation_decision(self):
        """9. Temporal features thực sự ảnh hưởng tới điểm xác nhận và chặn cảnh báo tĩnh khi feature_threshold > 0."""
        verifier_feat = TemporalVerifier(
            window_size_n=5,
            confirm_frames_k=3,
            suspect_frames_k=2,
            cooldown_sec=2.0,
            feature_threshold=0.05,
        )
        static_box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.90,
            xyxy=(100, 100, 150, 150),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
            track_id=77,
        )
        static_feat = TemporalFeatureResult(
            area_growth_rate=0.0,
            centroid_speed=0.0,
            roi_intensity_variance=0.0,
            flicker_score=0.0,
            smoke_expansion_score=0.0,
        )
        dynamic_feat = TemporalFeatureResult(
            area_growth_rate=0.12,
            centroid_speed=2.5,
            roi_intensity_variance=14.0,
            flicker_score=0.08,
            smoke_expansion_score=0.10,
        )
        self.assertLess(compute_dynamic_feature_score(static_feat), 0.05)
        self.assertGreaterEqual(compute_dynamic_feature_score(dynamic_feat), 0.05)

        # Static object đạt 3/5 khung nhưng feature_score = 0.0 < 0.05 -> chỉ dừng ở SUSPECT, không báo FIRE
        for t in (1.0, 1.2, 1.4):
            res_static = verifier_feat.update(
                [static_box],
                tracks_assigned=True,
                feature_map={77: static_feat},
                timestamp_sec=t,
            )
        self.assertEqual(res_static["alert_level"], "SUSPECT")
        self.assertFalse(res_static["should_trigger_db_alert"])

        # Dynamic object đạt 3/5 khung và feature_score >= 0.05 -> xác nhận FIRE
        dynamic_box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.90,
            xyxy=(200, 200, 260, 260),
            norm_xywh=(0.4, 0.4, 0.1, 0.1),
            track_id=88,
        )
        for t in (2.0, 2.2, 2.4):
            res_dyn = verifier_feat.update(
                [dynamic_box],
                tracks_assigned=True,
                feature_map={88: dynamic_feat},
                timestamp_sec=t,
            )
        self.assertEqual(res_dyn["alert_level"], "FIRE")
        self.assertTrue(res_dyn["should_trigger_db_alert"])
        self.assertGreater(res_dyn["temporal_score"], 0.5)

    def test_fire_priority_over_smoke(self):
        """10. Kiểm tra ưu tiên nhãn FIRE hơn SMOKE khi cả hai xuất hiện đồng thời."""
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
        res = self.verifier.update([smoke_box, fire_box], tracks_assigned=True, timestamp_sec=1.0)
        self.assertEqual(res["primary_label"], "FIRE")

    def test_backward_compatibility_without_tracks_assigned(self):
        """11. Kiểm tra gọi update không truyền tracks_assigned vẫn hoạt động tự gán track bằng IoU."""
        box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.85,
            xyxy=(100, 100, 150, 150),
            norm_xywh=(0.2, 0.2, 0.1, 0.1),
            track_id=-1,
        )
        res = self.verifier.update([box], timestamp_sec=1.0)
        self.assertGreater(res["valid_boxes"][0].track_id, 0)


if __name__ == "__main__":
    unittest.main()
