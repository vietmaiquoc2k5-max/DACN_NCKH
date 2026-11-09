"""Unit tests for the Week 4 event evaluator and parameter grid search (`src/event_eval.py` & `src/week4_event_eval.py`)."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.event_eval import (
    Event,
    FrameDetection,
    evaluate,
    export_events_csv,
    export_grid_search_csv,
    export_json,
    extract_frame_level_alarms,
    frame_detections_to_events,
    grid_search,
    merge_frame_alarms_to_events,
    select_best_operating_point,
)
import src.week4_event_eval as legacy_eval


class Week4EvaluationTests(unittest.TestCase):
    """Kiểm thử toàn diện công cụ đánh giá theo sự kiện (event_eval.py)."""

    def setUp(self) -> None:
        self.gt = [
            Event("pos-1", "CAM-01", "FIRE", 2.0, 8.0),
            Event("pos-2", "CAM-02", "SMOKE", 5.0, 12.0),
        ]
        self.pos = {"pos-1", "pos-2"}
        self.neg = {"neg-1", "neg-2"}

    def test_frame_level_baseline_triggers_immediately_without_k_n(self) -> None:
        """1. Frame-level baseline tạo cảnh báo ngay khi có detection đủ ngưỡng, không cần K/N."""
        frames = [
            FrameDetection(
                "pos-1",
                "CAM-01",
                1.0,
                ({"label": "FIRE", "confidence": 0.90, "track_id": 1},),
            ),
            FrameDetection("pos-1", "CAM-01", 2.0, ()),
        ]
        raw_alarms = extract_frame_level_alarms(frames, confidence_threshold=0.5)
        self.assertEqual(len(raw_alarms), 1)
        self.assertEqual(raw_alarms[0]["timestamp_sec"], 1.0)

        baseline_events, temporal_events = frame_detections_to_events(
            frames, confidence_threshold=0.5, window_n=5, confirm_k=3
        )
        self.assertEqual(len(baseline_events), 1, "Frame-level baseline phải tạo sự kiện ngay từ 1 frame.")
        self.assertEqual(len(temporal_events), 0, "Temporal K/N (K=3) không được tạo sự kiện khi chỉ có 1 frame.")

    def test_event_matching_recall_delay_precision_and_f1(self) -> None:
        """2. Kiểm tra Event Matching, Event Recall, Detection Delay, Precision và F1."""
        pred = [
            Event("pos-1", "CAM-01", "FIRE", 3.5, 6.0, confidence=0.88),  # Delay = 3.5 - 2.0 = 1.5s
            Event("neg-1", "CAM-03", "FIRE", 10.0, 11.0, confidence=0.70),  # Unmatched false alarm on negative
        ]
        report = evaluate(self.gt, pred, self.pos, self.neg, negative_hours=2.0)
        metrics = report["metrics"]
        self.assertEqual(metrics["video_event_recall"], 0.5)
        self.assertEqual(metrics["event_instance_recall"], 0.5)
        self.assertEqual(metrics["event_precision"], 0.5)
        self.assertAlmostEqual(metrics["event_f1"], 0.5, places=5)
        self.assertAlmostEqual(metrics["mean_detection_delay_sec"], 1.5, places=5)
        self.assertEqual(metrics["false_alarms_per_hour"], 0.5)
        self.assertEqual(len(report["matches"]), 1)
        self.assertEqual(len(report["unmatched_ground_truth"]), 1)
        self.assertEqual(len(report["unmatched_predictions"]), 1)

    def test_negative_video_false_alarm_rate_strictly_from_negative_videos(self) -> None:
        """3. FAR/giờ chỉ tính trên tập video âm tính, không trộn cảnh báo sớm trên video dương tính vào negative FAR."""
        pred = [
            Event("pos-1", "CAM-01", "FIRE", 3.0, 5.0),
            Event("pos-2", "CAM-02", "FIRE", 0.5, 1.0),  # Pre-event / wrong class on positive video
            Event("neg-1", "CAM-09", "FIRE", 10.0, 11.0),
        ]
        report = evaluate(self.gt, pred, self.pos, self.neg, negative_hours=2.0)
        self.assertEqual(report["counts"]["negative_false_alarms"], 1)
        self.assertEqual(report["counts"]["unmatched_positive_predictions"], 1)
        self.assertEqual(report["metrics"]["false_alarms_per_hour"], 0.5)

    def test_missing_negative_videos_handling(self) -> None:
        """4. Xử lý trung thực khi thiếu negative videos: raise ValueError mặc định hoặc báo BLOCKED_BY_NEGATIVE_DATA."""
        pred = [Event("pos-1", "CAM-01", "FIRE", 3.0, 5.0)]
        with self.assertRaises(ValueError):
            evaluate(self.gt, pred, self.pos, set(), negative_hours=0.0, allow_missing_negative=False)

        blocked_report = evaluate(
            self.gt, pred, self.pos, set(), negative_hours=0.0, allow_missing_negative=True
        )
        self.assertEqual(blocked_report["status"], "BLOCKED_BY_NEGATIVE_DATA")
        self.assertIsNone(blocked_report["metrics"]["false_alarms_per_hour"])
        self.assertFalse(blocked_report["targets"]["overall_pass"])

    def test_k_n_confirmation_and_temporal_feature_gating_in_event_conversion(self) -> None:
        """5. Kiểm tra K/N làm giảm false alarm đơn lẻ và feature_threshold lọc đối tượng tĩnh."""
        frames = [
            # Nhiễu 1 frame trên neg-1
            FrameDetection(
                "neg-1",
                "CAM-09",
                1.0,
                ({"label": "FIRE", "confidence": 0.9, "track_id": 99, "feature_score": 0.01},),
            ),
            # Cháy thật 3 frame liên tiếp trên pos-1 với feature_score cao
            FrameDetection(
                "pos-1",
                "CAM-01",
                2.0,
                ({"label": "FIRE", "confidence": 0.9, "track_id": 1, "feature_score": 0.08},),
            ),
            FrameDetection(
                "pos-1",
                "CAM-01",
                3.0,
                ({"label": "FIRE", "confidence": 0.9, "track_id": 1, "feature_score": 0.09},),
            ),
            FrameDetection(
                "pos-1",
                "CAM-01",
                4.0,
                ({"label": "FIRE", "confidence": 0.9, "track_id": 1, "feature_score": 0.10},),
            ),
        ]
        baseline, temporal = frame_detections_to_events(
            frames, confidence_threshold=0.5, window_n=5, confirm_k=3, feature_threshold=0.05
        )
        self.assertEqual(len(baseline), 2, "Baseline bị dính cả báo giả trên neg-1.")
        self.assertEqual(len(temporal), 1, "Temporal lọc sạch báo giả trên neg-1 và giữ lại sự kiện thật.")
        self.assertEqual(temporal[0].video_id, "pos-1")
        self.assertGreaterEqual(temporal[0].start_sec, 4.0)

    def test_grid_search_and_operating_point_selection(self) -> None:
        """6. Grid search trả về điểm vận hành hợp lệ khi đủ dữ liệu hoặc báo NO_FEASIBLE_OPERATING_POINT khi thiếu negative."""
        frames = [
            FrameDetection(
                "pos-1",
                "CAM-01",
                float(t),
                ({"label": "FIRE", "confidence": 0.9, "track_id": 1, "feature_score": 0.05},)
                if t in (2, 3, 4, 5)
                else (),
            )
            for t in range(8)
        ] + [
            FrameDetection(
                "neg-1",
                "CAM-09",
                1.0,
                ({"label": "FIRE", "confidence": 0.9, "track_id": 9, "feature_score": 0.01},),
            )
        ]
        gt_single = [Event("pos-1", "CAM-01", "FIRE", 2.0, 8.0)]
        rows = grid_search(
            frames,
            gt_single,
            {"pos-1"},
            {"neg-1"},
            negative_hours=1.0,
            confidence_thresholds=[0.5],
            k_values=[2, 3],
            n_values=[3, 5],
            feature_thresholds=[0.0, 0.02],
            cooldown_values=[5.0],
        )
        self.assertEqual(len(rows), 8)
        best = select_best_operating_point(rows)
        self.assertEqual(best["status"], "FEASIBLE_OPERATING_POINT_FOUND")
        self.assertIsNotNone(best["selected_operating_point"])
        self.assertTrue(best["selected_operating_point"]["feasible"])

        # Khi không có negative videos -> allow_missing_negative=True -> không thể có điểm feasible
        rows_blocked = grid_search(
            frames,
            gt_single,
            {"pos-1"},
            set(),
            negative_hours=0.0,
            confidence_thresholds=[0.5],
            k_values=[2],
            n_values=[3],
            allow_missing_negative=True,
        )
        best_blocked = select_best_operating_point(rows_blocked)
        self.assertEqual(best_blocked["status"], "NO_FEASIBLE_OPERATING_POINT")
        self.assertIsNone(best_blocked["selected_operating_point"])

    def test_csv_and_json_export(self) -> None:
        """7. Kiểm tra xuất kết quả CSV và JSON hợp lệ."""
        events = [Event("pos-1", "CAM-01", "FIRE", 2.5, 6.0, confidence=0.91)]
        grid_rows = [
            {
                "confidence_threshold": 0.35,
                "k": 3,
                "n": 8,
                "feature_threshold": 0.01,
                "cooldown_sec": 5.0,
                "baseline_far_per_hour": None,
                "temporal_far_per_hour": None,
                "far_reduction_ratio": None,
                "baseline_recall": 0.9,
                "temporal_recall": 0.9,
                "recall_drop_pct": 0.0,
                "event_precision": 0.85,
                "event_f1": 0.87,
                "mean_detection_delay_sec": 1.2,
                "status": "BLOCKED_BY_NEGATIVE_DATA",
                "feasible": False,
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            json_path = tmp / "metrics.json"
            events_csv = tmp / "events.csv"
            grid_csv = tmp / "grid.csv"

            export_json({"test": True}, json_path)
            export_events_csv(events, events_csv)
            export_grid_search_csv(grid_rows, grid_csv)

            self.assertTrue(json_path.exists())
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["test"], True)

            with events_csv.open("r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 1)
            self.assertEqual(reader[0]["video_id"], "pos-1")

            with grid_csv.open("r", encoding="utf-8") as f:
                g_reader = list(csv.DictReader(f))
            self.assertEqual(len(g_reader), 1)
            self.assertEqual(g_reader[0]["status"], "BLOCKED_BY_NEGATIVE_DATA")

    def test_legacy_wrapper_compatibility(self) -> None:
        """8. Kiểm tra wrapper src/week4_event_eval.py tương thích 100% với src/event_eval.py."""
        self.assertIs(legacy_eval.evaluate, evaluate)
        self.assertIs(legacy_eval.grid_search, grid_search)
        self.assertIs(legacy_eval.frame_detections_to_events, frame_detections_to_events)

    def test_invalid_k_n_rejected(self) -> None:
        frames = [FrameDetection("v", "c", 0.0, ())]
        with self.assertRaises(ValueError):
            frame_detections_to_events(frames, 0.5, 2, 3)

    def test_negative_video_pipeline_and_far_calculation(self) -> None:
        """9. Kiểm thử luồng xử lý video âm tính và tính toán FAR/giờ chính xác."""
        neg_frames = [
            FrameDetection("neg-1", "CAM-01", 1.0, ({"label": "FIRE", "confidence": 0.85, "track_id": 1, "temporal_score": 0.08},)),
            FrameDetection("neg-1", "CAM-01", 1.5, ({"label": "FIRE", "confidence": 0.88, "track_id": 1, "temporal_score": 0.09},)),
            FrameDetection("neg-1", "CAM-01", 2.0, ({"label": "FIRE", "confidence": 0.90, "track_id": 1, "temporal_score": 0.10},)),
            FrameDetection("neg-2", "CAM-02", 5.0, ({"label": "SMOKE", "confidence": 0.80, "track_id": 2, "temporal_score": 0.01},)),
        ]
        base_events, temp_events = frame_detections_to_events(
            neg_frames, confidence_threshold=0.5, window_n=5, confirm_k=3, feature_threshold=0.05
        )
        self.assertEqual(len(base_events), 2, "Baseline nhận cả 2 sự kiện giả trên 2 video âm tính")
        self.assertEqual(len(temp_events), 1, "Temporal K/N + Feature gate lọc sạch neg-2, chỉ còn 1 sự kiện lọt qua trên neg-1")

        # Đánh giá FAR trên 2 video âm tính với tổng thời lượng 2.0 giờ
        report_temp = evaluate(
            ground_truth=[],
            predictions=temp_events,
            positive_video_ids={"dummy-pos"},
            negative_video_ids={"neg-1", "neg-2"},
            negative_hours=2.0,
            allow_missing_negative=False,
        )
        self.assertEqual(report_temp["counts"]["negative_false_alarms"], 1)
        self.assertEqual(report_temp["metrics"]["false_alarms_per_hour"], 0.5)

    def test_feature_gate_on_off_behavior(self) -> None:
        """10. Kiểm thử tách biệt rõ ràng khi Feature Gate BẬT vs TẮT."""
        static_frames = [
            FrameDetection("neg-static", "CAM-01", 1.0, ({"label": "FIRE", "confidence": 0.90, "track_id": 10, "temporal_score": 0.01},)),
            FrameDetection("neg-static", "CAM-01", 1.5, ({"label": "FIRE", "confidence": 0.90, "track_id": 10, "temporal_score": 0.01},)),
            FrameDetection("neg-static", "CAM-01", 2.0, ({"label": "FIRE", "confidence": 0.90, "track_id": 10, "temporal_score": 0.01},)),
        ]
        # Gate TẮT (feature_threshold = 0.0) -> Báo động
        _, events_gate_off = frame_detections_to_events(
            static_frames, confidence_threshold=0.5, window_n=5, confirm_k=3, feature_threshold=0.0
        )
        self.assertEqual(len(events_gate_off), 1, "Gate TẮT phải cho phép kích hoạt khi đủ K/N")

        # Gate BẬT (feature_threshold = 0.05 > 0.01) -> Bị chặn
        _, events_gate_on = frame_detections_to_events(
            static_frames, confidence_threshold=0.5, window_n=5, confirm_k=3, feature_threshold=0.05
        )
        self.assertEqual(len(events_gate_on), 0, "Gate BẬT phải lọc bỏ hoàn toàn đối tượng tĩnh có temporal_score thấp")

    def test_grid_search_compares_baseline_at_same_confidence(self) -> None:
        """11. Kiểm thử Grid Search so sánh ứng viên với baseline tại cùng confidence threshold."""
        frames = [
            FrameDetection("pos-1", "CAM-01", 1.0, ({"label": "FIRE", "confidence": 0.40, "track_id": 1, "temporal_score": 0.08},)),
            FrameDetection("pos-1", "CAM-01", 1.5, ({"label": "FIRE", "confidence": 0.40, "track_id": 1, "temporal_score": 0.08},)),
            FrameDetection("pos-1", "CAM-01", 2.0, ({"label": "FIRE", "confidence": 0.40, "track_id": 1, "temporal_score": 0.08},)),
        ]
        gt = [Event("pos-1", "CAM-01", "FIRE", 1.0, 5.0)]
        rows = grid_search(
            frames,
            gt,
            positive_video_ids={"pos-1"},
            negative_video_ids={"neg-1"},
            negative_hours=1.0,
            confidence_thresholds=[0.35, 0.50],
            k_values=[2],
            n_values=[3],
            feature_thresholds=[0.0],
            cooldown_values=[0.0],
        )
        cand_35 = next(r for r in rows if r["confidence_threshold"] == 0.35)
        cand_50 = next(r for r in rows if r["confidence_threshold"] == 0.50)

        # Tại conf=0.35, baseline bắt được (conf 0.40 >= 0.35)
        self.assertEqual(cand_35["baseline_video_event_recall"], 1.0)
        # Tại conf=0.50, baseline không bắt được (conf 0.40 < 0.50)
        self.assertEqual(cand_50["baseline_video_event_recall"], 0.0)

    def test_validation_and_test_split_disjoint(self) -> None:
        """12. Kiểm thử Validation Set và Test Set tách biệt 100% không bị trộn."""
        from scripts.benchmark_week4_bytetrack import load_dataset_catalog
        csv_path = Path("data/video_dataset_list.csv")
        catalog = load_dataset_catalog(csv_path)
        avail_pos = catalog["available_positive"]
        val_videos = [v for idx, v in enumerate(avail_pos) if idx % 2 == 0]
        test_videos = [v for idx, v in enumerate(avail_pos) if idx % 2 == 1]
        val_ids = {v["video_id"] for v in val_videos}
        test_ids = {v["video_id"] for v in test_videos}

        self.assertEqual(len(val_ids), 12)
        self.assertEqual(len(test_ids), 12)
        self.assertTrue(val_ids.isdisjoint(test_ids), "Validation và Test không được có video trùng lặp")


if __name__ == "__main__":
    unittest.main()
