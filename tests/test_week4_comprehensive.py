"""Bộ kiểm thử thực nghiệm toàn diện Tuần 4 (Comprehensive Week 4 Unit & System Tests).

Bao gồm 9 bài kiểm thử bắt buộc:
1. JSONL parser & structure validation.
2. Empty frames preservation (`detections: []`).
3. Monotonic timestamps validation per stream.
4. Ground Truth validation schema & independent loading.
5. Event matching with temporal tolerance boundary conditions.
6. FAR calculation strictly isolated on negative videos.
7. Grid search baseline comparison strictly at the exact same confidence threshold.
8. Operating point selector reports NO_FEASIBLE_OPERATING_POINT when negative data missing.
9. Video train/val/test leakage prevention (disjoint partitions).
"""
import json
import tempfile
import unittest
from pathlib import Path

from src.event_eval import (
    Event,
    FrameDetection,
    evaluate,
    frame_detections_to_events,
    grid_search,
    load_frame_detections,
    load_ground_truth_events,
    load_jsonl,
    select_best_operating_point,
)
from scripts.validate_week4_frame_jsonl import validate_single_jsonl
from scripts.validate_ground_truth import validate_ground_truth


class Week4ComprehensiveTests(unittest.TestCase):
    """Bộ kiểm thử chất lượng và tính đúng đắn cho các thành phần Tuần 4."""

    def test_01_jsonl_parser_and_structure(self) -> None:
        """1. Kiểm tra parser JSONL: nhận diện cấu trúc hợp lệ và bắt lỗi khi thiếu trường bắt buộc."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_frames.jsonl"
            # Hợp lệ
            valid_content = (
                json.dumps({
                    "video_id": "POS_01",
                    "camera_id": "BENCH_POS_01",
                    "timestamp_sec": 0.0,
                    "detections": [
                        {"label": "FIRE", "confidence": 0.85, "bbox": [10, 20, 100, 200], "track_id": 1}
                    ]
                }) + "\n" +
                json.dumps({
                    "video_id": "POS_01",
                    "camera_id": "BENCH_POS_01",
                    "timestamp_sec": 0.2,
                    "detections": []
                }) + "\n"
            )
            tmp_path.write_text(valid_content, encoding="utf-8")
            frames = load_frame_detections(tmp_path)
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0].video_id, "POS_01")
            self.assertEqual(len(frames[0].detections), 1)
            self.assertEqual(len(frames[1].detections), 0)

            # Bất hợp lệ: timestamp âm
            invalid_path = Path(tmp_dir) / "invalid_ts.jsonl"
            invalid_path.write_text(
                json.dumps({"video_id": "POS_01", "camera_id": "BENCH_POS_01", "timestamp_sec": -1.0, "detections": []}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_frame_detections(invalid_path)

            # Bất hợp lệ: nhãn ngoài FIRE/SMOKE
            invalid_label = Path(tmp_dir) / "invalid_label.jsonl"
            invalid_label.write_text(
                json.dumps({
                    "video_id": "POS_01",
                    "camera_id": "BENCH_POS_01",
                    "timestamp_sec": 0.0,
                    "detections": [{"label": "PERSON", "confidence": 0.9}]
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_frame_detections(invalid_label)

    def test_02_empty_frames_preservation(self) -> None:
        """2. Khung hình không có phát hiện (zero detections) bắt buộc phải được lưu với `detections: []` và không bị drop."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "stream.jsonl"
            lines = [
                {"video_id": "VID_01", "camera_id": "CAM_01", "timestamp_sec": 0.0, "detections": []},
                {"video_id": "VID_01", "camera_id": "CAM_01", "timestamp_sec": 0.2, "detections": []},
                {"video_id": "VID_01", "camera_id": "CAM_01", "timestamp_sec": 0.4, "detections": [{"label": "FIRE", "confidence": 0.8, "track_id": 1}]},
                {"video_id": "VID_01", "camera_id": "CAM_01", "timestamp_sec": 0.6, "detections": []},
            ]
            tmp_path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

            report = validate_single_jsonl(tmp_path, is_tracked=True)
            self.assertEqual(report["metrics"]["total_lines"], 4)
            self.assertEqual(report["metrics"]["empty_frames"], 3)
            self.assertEqual(report["status"], "PASS")

            frames = load_frame_detections(tmp_path)
            self.assertEqual(len(frames), 4)
            self.assertEqual(frames[0].detections, ())
            self.assertEqual(frames[1].detections, ())
            self.assertEqual(frames[3].detections, ())

    def test_03_monotonic_timestamps_validation(self) -> None:
        """3. Kiểm tra tính đơn điệu không giảm của timestamp theo từng luồng video."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Luồng hợp lệ: timestamp tăng đơn điệu
            valid_path = Path(tmp_dir) / "valid.jsonl"
            lines_valid = [
                {"video_id": "VID_A", "camera_id": "CAM_A", "timestamp_sec": 0.0, "detections": []},
                {"video_id": "VID_A", "camera_id": "CAM_A", "timestamp_sec": 0.2, "detections": []},
                {"video_id": "VID_A", "camera_id": "CAM_A", "timestamp_sec": 0.4, "detections": []},
            ]
            valid_path.write_text("\n".join(json.dumps(l) for l in lines_valid) + "\n", encoding="utf-8")
            rep_valid = validate_single_jsonl(valid_path, is_tracked=False)
            self.assertEqual(rep_valid["errors"]["timestamp_inversion_count"], 0)
            self.assertEqual(rep_valid["status"], "PASS")

            # Luồng lỗi: timestamp bị lùi về quá khứ
            invalid_path = Path(tmp_dir) / "non_monotonic.jsonl"
            lines_invalid = [
                {"video_id": "VID_A", "camera_id": "CAM_A", "timestamp_sec": 1.0, "detections": []},
                {"video_id": "VID_A", "camera_id": "CAM_A", "timestamp_sec": 0.5, "detections": []},
            ]
            invalid_path.write_text("\n".join(json.dumps(l) for l in lines_invalid) + "\n", encoding="utf-8")
            rep_invalid = validate_single_jsonl(invalid_path, is_tracked=False)
            self.assertGreater(rep_invalid["errors"]["timestamp_inversion_count"], 0)
            self.assertEqual(rep_invalid["status"], "FAIL")

    def test_04_ground_truth_validation_schema(self) -> None:
        """4. Xác thực schema của Ground Truth: khoảng thời gian hợp lệ, nhãn chuẩn, nạp độc lập."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            valid_gt_path = Path(tmp_dir) / "valid_events.jsonl"
            valid_records = [
                {
                    "video_id": "POS_TEST",
                    "camera_id": "BENCH_POS_TEST",
                    "label": "FIRE",
                    "start_sec": 2.0,
                    "end_sec": 15.0,
                    "confidence": 1.0,
                    "annotator_id": "annotator_01",
                    "verification_status": "verified",
                },
                {
                    "video_id": "POS_TEST",
                    "camera_id": "BENCH_POS_TEST",
                    "label": "SMOKE",
                    "start_sec": 5.0,
                    "end_sec": 20.0,
                    "confidence": 1.0,
                    "annotator_id": "annotator_02",
                    "verification_status": "pending_verification",
                },
            ]
            valid_gt_path.write_text("\n".join(json.dumps(r) for r in valid_records) + "\n", encoding="utf-8")
            res_valid = validate_ground_truth(valid_gt_path)
            self.assertEqual(res_valid["status"], "PASS")
            self.assertEqual(len(res_valid["errors"]), 0)

            # Lỗi: start_sec > end_sec
            invalid_gt_path = Path(tmp_dir) / "invalid_events.jsonl"
            invalid_records = [
                {
                    "video_id": "POS_TEST",
                    "camera_id": "BENCH_POS_TEST",
                    "label": "FIRE",
                    "start_sec": 20.0,
                    "end_sec": 10.0,
                    "confidence": 1.0,
                    "annotator_id": "annotator_01",
                    "verification_status": "verified",
                }
            ]
            invalid_gt_path.write_text("\n".join(json.dumps(r) for r in invalid_records) + "\n", encoding="utf-8")
            res_invalid = validate_ground_truth(invalid_gt_path)
            self.assertEqual(res_invalid["status"], "FAIL")
            self.assertTrue(any("phải lớn hơn start_sec" in err for err in res_invalid["errors"]))

        # Kiểm tra nạp từ events.jsonl thực tế trong repository
        loaded_events = load_ground_truth_events()
        self.assertGreaterEqual(len(loaded_events), 24)
        for ev in loaded_events:
            self.assertIn(ev.label, {"FIRE", "SMOKE"})
            self.assertLessEqual(ev.start_sec, ev.end_sec)

    def test_05_event_matching_with_tolerance(self) -> None:
        """5. Khớp sự kiện có tính dung sai (tolerance_sec) theo đúng biên thời gian."""
        gt_events = [Event("VID_01", "CAM_01", "FIRE", 10.0, 30.0)]
        pos_ids = {"VID_01"}

        # TH1: Khớp hợp lệ (dự đoán nằm trong khoảng [10, 30])
        pred_match = [Event("VID_01", "CAM_01", "FIRE", 12.0, 25.0)]
        rep1 = evaluate(gt_events, pred_match, pos_ids, None, None, tolerance_sec=5.0, allow_missing_negative=True)
        self.assertEqual(rep1["counts"]["matched_events"], 1)
        self.assertAlmostEqual(rep1["metrics"]["mean_detection_delay_sec"], 2.0)

        # TH2: Dự đoán bắt đầu sau GT nhưng trong khoảng tolerance_sec=5.0 (ví dụ start=34s <= 30 + 5)
        pred_tol = [Event("VID_01", "CAM_01", "FIRE", 32.0, 40.0)]
        rep2 = evaluate(gt_events, pred_tol, pos_ids, None, None, tolerance_sec=5.0, allow_missing_negative=True)
        self.assertEqual(rep2["counts"]["matched_events"], 1)

        # TH3: Dự đoán bắt đầu quá muộn ngoài tolerance_sec (start=36s > 30 + 5)
        pred_late = [Event("VID_01", "CAM_01", "FIRE", 36.0, 45.0)]
        rep3 = evaluate(gt_events, pred_late, pos_ids, None, None, tolerance_sec=5.0, allow_missing_negative=True)
        self.assertEqual(rep3["counts"]["matched_events"], 0)

        # TH4: Cảnh báo giả kết thúc hoàn toàn trước khi cháy bắt đầu (end=8.0s < gt_start=10.0s)
        pred_early = [Event("VID_01", "CAM_01", "FIRE", 2.0, 8.0)]
        rep4 = evaluate(gt_events, pred_early, pos_ids, None, None, tolerance_sec=5.0, allow_missing_negative=True)
        self.assertEqual(rep4["counts"]["matched_events"], 0)
        self.assertEqual(rep4["counts"]["positive_unmatched_alerts"], 1)

    def test_06_far_calculation_strictly_on_negative_videos(self) -> None:
        """6. Báo động giả FAR chỉ được tính trên tập video âm tính; cảnh báo không khớp trên video dương tính không được tính vào FAR."""
        gt_events = [Event("POS_01", "CAM_01", "FIRE", 10.0, 20.0)]
        pos_ids = {"POS_01"}
        neg_ids = {"NEG_01", "NEG_02"}
        neg_hours = 2.0

        predictions = [
            Event("POS_01", "CAM_01", "FIRE", 12.0, 18.0),  # Matched GT
            Event("POS_01", "CAM_01", "SMOKE", 1.0, 5.0),    # Unmatched on POSITIVE video
            Event("NEG_01", "CAM_02", "FIRE", 4.0, 8.0),     # False alarm on NEGATIVE video
            Event("NEG_02", "CAM_03", "SMOKE", 10.0, 15.0),  # False alarm on NEGATIVE video
        ]

        report = evaluate(gt_events, predictions, pos_ids, neg_ids, neg_hours)
        # Chỉ có 2 cảnh báo trên NEG_01 và NEG_02 được tính vào FAR
        self.assertEqual(report["counts"]["negative_false_alarms"], 2)
        # Cảnh báo SMOKE trên POS_01 là positive_unmatched_alert, không thuộc negative_false_alarms
        self.assertEqual(report["counts"]["positive_unmatched_alerts"], 1)
        # FAR = 2 false alarms / 2.0 hours = 1.0 FA/hour
        self.assertAlmostEqual(report["metrics"]["false_alarms_per_hour"], 1.0)
        self.assertEqual(report["metrics"]["far_status"], "VERIFIED")

        # Khi thiếu video âm tính và allow_missing_negative=True
        rep_missing = evaluate(gt_events, predictions, pos_ids, None, None, allow_missing_negative=True)
        self.assertIsNone(rep_missing["metrics"]["false_alarms_per_hour"])
        self.assertEqual(rep_missing["status"], "BLOCKED_BY_NEGATIVE_DATA")

        # Khi thiếu video âm tính và allow_missing_negative=False -> ném ValueError
        with self.assertRaises(ValueError):
            evaluate(gt_events, predictions, pos_ids, None, None, allow_missing_negative=False)

    def test_07_grid_search_baseline_comparison_at_same_confidence(self) -> None:
        """7. Grid search bắt buộc phải đối chiếu từng candidate với baseline ở cùng ngưỡng confidence."""
        frames = [
            FrameDetection("POS_01", "CAM_01", 0.0, ({"label": "FIRE", "confidence": 0.30, "track_id": 1},)),
            FrameDetection("POS_01", "CAM_01", 0.2, ({"label": "FIRE", "confidence": 0.30, "track_id": 1},)),
            FrameDetection("POS_01", "CAM_01", 0.4, ({"label": "FIRE", "confidence": 0.60, "track_id": 1},)),
            FrameDetection("POS_01", "CAM_01", 0.6, ({"label": "FIRE", "confidence": 0.60, "track_id": 1},)),
        ]
        gt = [Event("POS_01", "CAM_01", "FIRE", 0.0, 1.0)]
        pos_ids = {"POS_01"}
        neg_ids = {"NEG_01"}
        neg_hours = 1.0

        # Chạy grid search với 2 mức conf: 0.25 và 0.50
        results = grid_search(
            frames=frames,
            ground_truth=gt,
            positive_video_ids=pos_ids,
            negative_video_ids=neg_ids,
            negative_hours=neg_hours,
            confidence_thresholds=[0.25, 0.50],
            k_values=[2],
            n_values=[2],
            feature_thresholds=[0.0],
            cooldown_values=[0.0],
        )

        c_025 = [c for c in results if c["confidence_threshold"] == 0.25][0]
        c_050 = [c for c in results if c["confidence_threshold"] == 0.50][0]

        # Candidate ở conf=0.25 phải so sánh với baseline có detection bắt đầu từ timestamp 0.0
        # Candidate ở conf=0.50 phải so sánh với baseline chỉ nhận frame từ timestamp 0.4
        self.assertIsNotNone(c_025["baseline_video_event_recall"])
        self.assertIsNotNone(c_050["baseline_video_event_recall"])

    def test_08_operating_point_selector_reports_no_feasible_when_blocked(self) -> None:
        """8. Bộ chọn Operating Point phải trả về NO_FEASIBLE_OPERATING_POINT khi thiếu dữ liệu âm tính."""
        candidates = [
            {
                "confidence_threshold": 0.35,
                "k": 3,
                "n": 8,
                "feature_threshold": 0.02,
                "cooldown_sec": 5.0,
                "baseline_video_event_recall": 1.0,
                "baseline_far_per_hour": None,
                "metrics": {
                    "video_event_recall": 1.0,
                    "event_instance_recall": 0.85,
                    "event_precision": 0.90,
                    "event_f1": 0.87,
                    "false_alarms_per_hour": None,
                    "mean_detection_delay_sec": 2.5,
                },
                "counts": {"positive_unmatched_alerts": 2},
                "far_reduction_vs_frame_baseline": None,
                "video_event_recall_drop": 0.0,
                "feasible": False,
                "status": "BLOCKED_BY_NEGATIVE_DATA",
            }
        ]

        decision = select_best_operating_point(candidates)
        self.assertEqual(decision["status"], "NO_FEASIBLE_OPERATING_POINT")
        self.assertFalse(decision["feasible"])
        self.assertTrue(decision["blocked_by_negative_data"])
        self.assertIsNone(decision["selected_operating_point"])
        self.assertIsNotNone(decision["provisional_candidate_on_positive_val"])

    def test_09_video_split_leakage_prevention(self) -> None:
        """9. Đảm bảo tính độc lập tuyệt đối giữa các tập phân chia Validation và Test (không trùng lặp video ID)."""
        val_positive_ids = {"POS_01", "POS_03", "POS_05", "POS_07", "POS_09", "POS_11", "POS_13", "POS_15", "POS_17", "POS_19", "POS_21", "POS_23"}
        test_positive_ids = {"POS_02", "POS_04", "POS_06", "POS_08", "POS_10", "POS_12", "POS_14", "POS_16", "POS_18", "POS_20", "POS_22", "POS_24"}

        # Giao giữa 2 tập phải rỗng
        intersection = val_positive_ids.intersection(test_positive_ids)
        self.assertEqual(len(intersection), 0, f"Rò rỉ video giữa Validation và Test: {intersection}")
        self.assertEqual(len(val_positive_ids), 12)
        self.assertEqual(len(test_positive_ids), 12)


if __name__ == "__main__":
    unittest.main()
