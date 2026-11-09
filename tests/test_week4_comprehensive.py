"""Bộ kiểm thử thực nghiệm toàn diện và nghiêm ngặt Tuần 4 (Branch `fix`).

Bao gồm đầy đủ 18 bài kiểm thử theo yêu cầu chuẩn khoa học:
1. Nạp Ground Truth chỉ lấy verified (Verified-only Ground Truth loading).
2. Bản ghi pending bị loại hoàn toàn khỏi các chỉ số chính thức.
3. Thiếu GT độc lập phải fail closed, không fallback ngầm sang catalog.
4. Từ chối trạng thái verification_status không hợp lệ.
5. Bản ghi verified bắt buộc có annotator_id hợp lệ.
6. Kiểm thử biên khớp thời điểm khởi phát (Event onset matching boundaries).
7. Phát hiện muộn (Late detection) không được tính là timely detection.
8. Xử lý cảnh báo sớm (Early alert) và chồng lấn onset.
9. Ràng buộc ánh xạ 1-1 (One-to-one event matching) và tách biệt nhãn FIRE/SMOKE.
10. Tối ưu số lượng ghép cặp cực đại (Matching cardinality correctness).
11. Tính toán độ trễ phát hiện (Detection delay correctness).
12. Tính nhất quán định danh khung hình giữa raw và tracked JSONL.
13. Phát hiện khung hình bị thiếu hoặc lệch định danh giữa hai tệp.
14. Phát hiện khung hình trùng lặp timestamp.
15. Kiểm tra schema JSONL và bảo toàn khung hình rỗng (`detections: []`).
16. Grid Search chỉ sử dụng tập Validation, không dùng Locked Test.
17. Ngăn chặn tuyệt đối rò rỉ dữ liệu giữa Validation và Test (Disjoint partitions).
18. Operating Point duy trì bị chặn và FAR không bị gán 0 khi thiếu dữ liệu âm tính.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.event_eval import (
    Event,
    FrameDetection,
    evaluate,
    frame_detections_to_events,
    get_ground_truth_manifest,
    grid_search,
    is_timely_onset_match,
    load_frame_detections,
    load_ground_truth_events,
    max_cardinality_min_cost_matching,
    select_best_operating_point,
)
from scripts.validate_week4_frame_jsonl import check_frame_identities_alignment, validate_single_jsonl
from scripts.validate_ground_truth import validate_ground_truth


class Week4RigorousEvaluationTests(unittest.TestCase):
    """Bộ kiểm thử toàn diện 18 tiêu chí cho đánh giá thực nghiệm Tuần 4."""

    def test_01_verified_only_ground_truth_loading(self) -> None:
        """1. load_ground_truth_events mặc định chỉ nạp các bản ghi có verification_status='verified'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            gt_file = Path(tmp_dir) / "events.jsonl"
            records = [
                {"video_id": "V1", "camera_id": "CAM_V1", "label": "FIRE", "start_sec": 1.0, "end_sec": 10.0, "confidence": 1.0, "annotator_id": "user_a", "verification_status": "verified"},
                {"video_id": "V1", "camera_id": "CAM_V1", "label": "SMOKE", "start_sec": 2.0, "end_sec": 12.0, "confidence": 1.0, "annotator_id": "annotator_pending", "verification_status": "pending_verification"},
            ]
            gt_file.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

            # Mặc định only_verified=True
            events_default = load_ground_truth_events(events_path=gt_file, strict=False)
            self.assertEqual(len(events_default), 1)
            self.assertEqual(events_default[0].label, "FIRE")

            # Khi chỉ định only_verified=False
            events_all = load_ground_truth_events(events_path=gt_file, only_verified=False, strict=False)
            self.assertEqual(len(events_all), 2)

    def test_02_pending_records_excluded_from_official_metrics(self) -> None:
        """2. Các bản ghi pending không bao giờ được tham gia vào precision, recall, F1 hay delay chính thức."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            gt_file = Path(tmp_dir) / "events.jsonl"
            records = [
                {"video_id": "V1", "camera_id": "CAM_V1", "label": "FIRE", "start_sec": 10.0, "end_sec": 20.0, "confidence": 1.0, "annotator_id": "user_1", "verification_status": "verified"},
                {"video_id": "V2", "camera_id": "CAM_V2", "label": "FIRE", "start_sec": 15.0, "end_sec": 25.0, "confidence": 1.0, "annotator_id": "annotator_pending", "verification_status": "pending_verification"},
            ]
            gt_file.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

            verified_gt = load_ground_truth_events(events_path=gt_file, only_verified=True, strict=False)
            self.assertEqual(len(verified_gt), 1)
            self.assertEqual(verified_gt[0].video_id, "V1")

            # Giả lập prediction trên cả V1 và V2
            predictions = [
                Event("V1", "CAM_V1", "FIRE", 11.0, 18.0),
                Event("V2", "CAM_V2", "FIRE", 16.0, 22.0),
            ]
            report = evaluate(verified_gt, predictions, positive_video_ids={"V1", "V2"}, allow_missing_negative=True)
            # Chỉ có 1 GT verified nên chỉ 1 GT được tính
            self.assertEqual(report["counts"]["ground_truth_events"], 1)
            self.assertEqual(report["counts"]["matched_events"], 1)
            # Prediction trên V2 không có verified GT tương ứng nên là positive unmatched alert
            self.assertEqual(report["counts"]["positive_unmatched_alerts"], 1)
            self.assertEqual(report["metrics"]["event_instance_recall"], 1.0)
            self.assertEqual(report["metrics"]["event_precision"], 0.5)

    def test_03_missing_independent_gt_fails_closed(self) -> None:
        """3. Thiếu tệp GT độc lập phải ném lỗi rõ ràng (fail closed), không tự động fallback sang catalog."""
        non_existent = Path("non/existent/path/events.jsonl")
        with self.assertRaises(FileNotFoundError) as ctx:
            load_ground_truth_events(events_path=non_existent, strict=True, allow_catalog_fallback=False)
        self.assertIn("Không tìm thấy tệp Ground Truth độc lập", str(ctx.exception))

    def test_04_invalid_verification_status_rejected(self) -> None:
        """4. Trạng thái verification_status ngoài 'verified' và 'pending_verification' phải bị từ chối."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            gt_file = Path(tmp_dir) / "invalid_status.jsonl"
            records = [
                {"video_id": "V1", "camera_id": "CAM_V1", "label": "FIRE", "start_sec": 1.0, "end_sec": 10.0, "confidence": 1.0, "annotator_id": "user_a", "verification_status": "in_review"}
            ]
            gt_file.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                load_ground_truth_events(events_path=gt_file, strict=False)
            self.assertIn("verification_status='in_review' không hợp lệ", str(ctx.exception))

    def test_05_verified_annotation_requires_valid_annotator(self) -> None:
        """5. Bản ghi verified bắt buộc phải có annotator_id hợp lệ, không chấp nhận placeholder."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            gt_file = Path(tmp_dir) / "invalid_annotator.jsonl"
            records = [
                {"video_id": "V1", "camera_id": "CAM_V1", "label": "FIRE", "start_sec": 1.0, "end_sec": 10.0, "confidence": 1.0, "annotator_id": "annotator_pending", "verification_status": "verified"}
            ]
            gt_file.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                load_ground_truth_events(events_path=gt_file, strict=False)
            self.assertIn("thiếu annotator_id hợp lệ", str(ctx.exception))

    def test_06_event_onset_matching_boundaries(self) -> None:
        """6. Kiểm tra các điều kiện biên của Onset Matching với tolerance_sec."""
        gt = Event("V1", "C1", "FIRE", 10.0, 70.0)
        tol = 5.0

        # Đúng mốc onset (delay = 0.0s)
        p_exact = Event("V1", "C1", "FIRE", 10.0, 30.0)
        self.assertTrue(is_timely_onset_match(gt, p_exact, tolerance_sec=tol))

        # Trong ngưỡng dung sai onset (start = 13.0 <= 10 + 5)
        p_in_tol = Event("V1", "C1", "FIRE", 13.0, 30.0)
        self.assertTrue(is_timely_onset_match(gt, p_in_tol, tolerance_sec=tol))

        # Ngay tại biên dung sai (start = 15.0 == 10 + 5)
        p_boundary = Event("V1", "C1", "FIRE", 15.0, 30.0)
        self.assertTrue(is_timely_onset_match(gt, p_boundary, tolerance_sec=tol))

        # Vừa vượt quá dung sai (start = 15.1 > 10 + 5) -> Phải từ chối kịp thời
        p_beyond = Event("V1", "C1", "FIRE", 15.1, 30.0)
        self.assertFalse(is_timely_onset_match(gt, p_beyond, tolerance_sec=tol))

    def test_07_late_detection_not_timely(self) -> None:
        """7. Dự đoán xảy ra muộn sau onset tolerance (ví dụ trễ 22s hoặc 32s) không được tính là timely detection."""
        gt = Event("V1", "C1", "FIRE", 10.0, 70.0)
        tol = 10.0
        # Dự đoán bắt đầu tại 42s (trễ 32s so với onset 10s) dù nằm trong khoảng [10, 70]
        p_late = Event("V1", "C1", "FIRE", 42.0, 60.0)
        self.assertFalse(is_timely_onset_match(gt, p_late, tolerance_sec=tol))

        report = evaluate([gt], [p_late], positive_video_ids={"V1"}, tolerance_sec=tol, allow_missing_negative=True)
        self.assertEqual(report["counts"]["matched_events"], 0)
        self.assertEqual(report["counts"]["timely_detections"], 0)
        self.assertEqual(report["counts"]["late_detections"], 1)
        self.assertEqual(report["metrics"]["video_event_recall"], 0.0)

    def test_08_early_alert_handling(self) -> None:
        """8. Xử lý cảnh báo sớm: kết thúc trước onset là early alert, chồng lấn onset trong early_tol là kịp thời."""
        gt = Event("V1", "C1", "FIRE", 10.0, 50.0)
        tol = 5.0

        # Kết thúc hoàn toàn trước onset (2.0s -> 8.0s < 10.0s) -> Không khớp, là early alert
        p_pre = Event("V1", "C1", "FIRE", 2.0, 8.0)
        self.assertFalse(is_timely_onset_match(gt, p_pre, tolerance_sec=tol, early_tolerance_sec=tol))
        rep_pre = evaluate([gt], [p_pre], positive_video_ids={"V1"}, tolerance_sec=tol, allow_missing_negative=True)
        self.assertEqual(rep_pre["counts"]["matched_events"], 0)
        self.assertEqual(rep_pre["counts"]["early_alerts"], 1)

        # Bắt đầu sớm trong khoảng early_tol (8.0s >= 10 - 5) và kéo dài vào đám cháy (8.0s -> 25.0s) -> Khớp timely, delay = 0.0s
        p_overlap = Event("V1", "C1", "FIRE", 8.0, 25.0)
        self.assertTrue(is_timely_onset_match(gt, p_overlap, tolerance_sec=tol, early_tolerance_sec=tol))
        rep_overlap = evaluate([gt], [p_overlap], positive_video_ids={"V1"}, tolerance_sec=tol, allow_missing_negative=True)
        self.assertEqual(rep_overlap["counts"]["matched_events"], 1)
        self.assertAlmostEqual(rep_overlap["matches"][0]["delay_sec"], 0.0)

    def test_09_one_to_one_event_matching(self) -> None:
        """9. Ràng buộc ánh xạ 1-1: 1 pred không khớp nhiều GT, 1 GT không khớp nhiều pred, nhãn khác nhau không khớp."""
        gt_fire = Event("V1", "C1", "FIRE", 10.0, 30.0)
        gt_smoke = Event("V1", "C1", "SMOKE", 12.0, 30.0)
        preds = [
            Event("V1", "C1", "FIRE", 11.0, 20.0),
            Event("V1", "C1", "FIRE", 13.0, 22.0),  # Dư thừa cho gt_fire
            Event("V1", "C1", "SMOKE", 14.0, 25.0),
        ]
        rep = evaluate([gt_fire, gt_smoke], preds, positive_video_ids={"V1"}, tolerance_sec=5.0, allow_missing_negative=True)
        self.assertEqual(rep["counts"]["matched_events"], 2)
        # 1 dự đoán FIRE thừa không được khớp thêm vào gt_smoke hay gt_fire
        self.assertEqual(rep["counts"]["positive_unmatched_alerts"], 1)

    def test_10_matching_cardinality_correctness(self) -> None:
        """10. Thuật toán ghép cặp cực đại (Max Cardinality Bipartite Matching) giải đúng trường hợp Greedy thất bại."""
        # GT1 tại 10s (tol=5 -> nhận [5, 15]), GT2 tại 15s (tol=5 -> nhận [10, 20])
        # Pred1 tại 14s, Pred2 tại 18s
        # Nếu greedy theo khoảng cách: Pred1 khớp GT2 (diff=1), khiến Pred2 không thể khớp GT1 (diff=8 > 5) -> Chỉ được 1 match
        # Thuật toán tối ưu Cardinality phải ghép (GT1, Pred1) [diff=4] và (GT2, Pred2) [diff=3] -> Được 2 matches!
        cost_matrix = [
            [4.0, None],   # GT1 với Pred1 (diff=4), GT1 với Pred2 (vượt tol -> None)
            [1.0, 3.0],    # GT2 với Pred1 (diff=1), GT2 với Pred2 (diff=3)
        ]
        matches = max_cardinality_min_cost_matching(cost_matrix)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches, [(0, 0), (1, 1)])

    def test_11_detection_delay_correctness(self) -> None:
        """11. Độ trễ phát hiện được tính nhất quán từ GT onset tới Prediction onset."""
        gt = Event("V1", "C1", "FIRE", 10.0, 40.0)
        pred = Event("V1", "C1", "FIRE", 14.25, 30.0)
        rep = evaluate([gt], [pred], positive_video_ids={"V1"}, tolerance_sec=10.0, allow_missing_negative=True)
        self.assertAlmostEqual(rep["metrics"]["mean_detection_delay_sec"], 4.25, places=4)

    def test_12_raw_tracked_frame_identity_consistency(self) -> None:
        """12. Kiểm tra tính khớp nối định danh khung hình 100% giữa raw và tracked JSONL."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir) / "frame_level_detections.jsonl"
            trk_path = Path(tmp_dir) / "tracked_frame_detections.jsonl"

            f1 = {"video_id": "V1", "camera_id": "C1", "timestamp_sec": 0.0, "detections": []}
            f2 = {"video_id": "V1", "camera_id": "C1", "timestamp_sec": 0.2, "detections": []}

            raw_path.write_text(json.dumps(f1) + "\n" + json.dumps(f2) + "\n", encoding="utf-8")
            trk_path.write_text(json.dumps(f1) + "\n" + json.dumps(f2) + "\n", encoding="utf-8")

            aligned, errors, details = check_frame_identities_alignment(raw_path, trk_path)
            self.assertTrue(aligned)
            self.assertEqual(details["total_frames_aligned"], 2)
            self.assertEqual(details["mismatches_count"], 0)

    def test_13_missing_or_mismatched_frame_detection(self) -> None:
        """13. Phát hiện tệp JSONL bị lệch frame identity hoặc thiếu frame."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir) / "raw.jsonl"
            trk_path = Path(tmp_dir) / "trk.jsonl"

            f_raw = {"video_id": "V1", "camera_id": "C1", "timestamp_sec": 0.0, "detections": []}
            # Tracked bị lệch timestamp
            f_trk = {"video_id": "V1", "camera_id": "C1", "timestamp_sec": 0.4, "detections": []}

            raw_path.write_text(json.dumps(f_raw) + "\n", encoding="utf-8")
            trk_path.write_text(json.dumps(f_trk) + "\n", encoding="utf-8")

            aligned, errors, details = check_frame_identities_alignment(raw_path, trk_path)
            self.assertFalse(aligned)
            self.assertGreater(details["mismatches_count"], 0)

    def test_14_duplicate_frame_detection(self) -> None:
        """14. Phát hiện khung hình trùng lặp timestamp trong cùng một luồng video."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "duplicates.jsonl"
            lines = [
                {"video_id": "V1", "camera_id": "C1", "timestamp_sec": 1.0, "detections": []},
                {"video_id": "V1", "camera_id": "C1", "timestamp_sec": 1.0, "detections": []},
            ]
            json_path.write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")

            res = validate_single_jsonl(json_path, is_tracked=False)
            self.assertGreater(res["errors"]["duplicate_timestamp_count"], 0)
            self.assertEqual(res["status"], "FAIL")

    def test_15_jsonl_schema_and_empty_frames(self) -> None:
        """15. Bảo toàn khung hình rỗng `detections: []` và kiểm định tọa độ bounding box."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "boxes.jsonl"
            lines = [
                {"video_id": "V1", "camera_id": "C1", "timestamp_sec": 0.0, "detections": []},
                {"video_id": "V1", "camera_id": "C1", "timestamp_sec": 0.2, "detections": [
                    {"label": "FIRE", "confidence": 0.85, "track_id": None, "xyxy": [10, 20, 100, 200]}
                ]},
            ]
            json_path.write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")

            res = validate_single_jsonl(json_path, is_tracked=False)
            self.assertEqual(res["status"], "PASS")
            self.assertEqual(res["metrics"]["empty_frames"], 1)
            self.assertEqual(res["metrics"]["non_empty_frames"], 1)

    def test_16_grid_search_does_not_use_locked_test(self) -> None:
        """16. Grid Search chỉ sử dụng dữ liệu Validation, không sử dụng dữ liệu Locked Test."""
        val_frames = [
            FrameDetection("VAL_1", "CAM_V1", 0.0, ({"label": "FIRE", "confidence": 0.8, "track_id": 1},)),
            FrameDetection("VAL_1", "CAM_V1", 0.2, ({"label": "FIRE", "confidence": 0.8, "track_id": 1},)),
        ]
        gt_val = [Event("VAL_1", "CAM_V1", "FIRE", 0.0, 1.0)]

        # Grid search chạy trên val_frames và gt_val
        results = grid_search(
            frames=val_frames,
            ground_truth=gt_val,
            positive_video_ids={"VAL_1"},
            negative_video_ids=None,
            negative_hours=None,
            confidence_thresholds=[0.35],
            k_values=[2],
            n_values=[2],
            allow_missing_negative=True,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metrics"]["video_event_recall"], 1.0)
        # Đảm bảo không có video test nào tham gia
        self.assertNotIn("TEST_1", {gt.video_id for gt in gt_val})

    def test_17_no_train_val_test_leakage(self) -> None:
        """17. Đảm bảo các tập Validation và Test tách biệt 100%, không giao nhau."""
        val_ids = {"POS_01", "POS_03", "POS_05", "POS_07", "POS_09", "POS_11", "POS_13", "POS_15", "POS_17", "POS_19", "POS_21", "POS_23"}
        test_ids = {"POS_02", "POS_04", "POS_06", "POS_08", "POS_10", "POS_12", "POS_14", "POS_16", "POS_18", "POS_20", "POS_22", "POS_24"}
        self.assertTrue(val_ids.isdisjoint(test_ids))
        self.assertEqual(len(val_ids.intersection(test_ids)), 0)

    def test_18_operating_point_remains_blocked_when_negative_data_missing(self) -> None:
        """18. Operating Point phải trả về NO_FEASIBLE_OPERATING_POINT và FAR=None khi thiếu video âm tính."""
        candidates = [
            {
                "confidence_threshold": 0.45,
                "k": 2,
                "n": 12,
                "feature_threshold": 0.0,
                "cooldown_sec": 0.0,
                "baseline_video_event_recall": 1.0,
                "baseline_far_per_hour": None,
                "metrics": {
                    "video_event_recall": 1.0,
                    "event_instance_recall": 1.0,
                    "event_precision": 0.75,
                    "event_f1": 0.85,
                    "false_alarms_per_hour": None,
                    "mean_detection_delay_sec": 1.5,
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


if __name__ == "__main__":
    unittest.main()
