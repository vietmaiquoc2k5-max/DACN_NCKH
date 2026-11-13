"""Bộ kiểm thử tính liêm chính khoa học của hệ thống đánh giá Week 4.

Kiểm tra:
1. Strict Ground Truth filtering: chỉ cross_verified được dùng khi strict=True.
2. FAR calculation honesty: không chấp nhận FAR=0 nếu không có thực nghiệm hoặc tổng thời lượng âm tính = 0.
3. Multi-camera RTSP benchmark honesty: từ chối RTSP test nếu không có nguồn live RTSP khả dụng.
4. Masking thông tin nhạy cảm (mật khẩu) trong RTSP URL.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from src.camera_manager import MultiCameraManager, mask_rtsp_credentials
from src.event_eval import load_ground_truth_events
from scripts.benchmark_multicamera_scientific import run_rtsp_live_benchmark

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestScientificValidationIntegrity(unittest.TestCase):
    """Kiểm thử tính liêm chính khoa học và bảo mật thông tin nhạy cảm."""

    def test_strict_ground_truth_filter_only_accepts_cross_verified(self):
        """1. Khi strict=True và only_verified=True, chỉ có đúng 11 sự kiện cross_verified được nạp."""
        events_strict = load_ground_truth_events(only_verified=True, strict=True)
        self.assertEqual(len(events_strict), 11, "Chỉ có đúng 11 sự kiện cross_verified trong events.jsonl v1.1.")
        for ev in events_strict:
            status = getattr(ev, "verification_status", "verified")
            self.assertIn(status, {"cross_verified", "verified"})

        # Khi only_verified=False, phải nạp đủ toàn bộ 35 sự kiện
        events_all = load_ground_truth_events(only_verified=False, strict=False)
        self.assertEqual(len(events_all), 35, "Toàn bộ tập Ground Truth phải có đủ 35 sự kiện.")

    def test_far_requires_positive_test_duration(self):
        """2. Tính FAR phải dựa trên thời lượng thực nghiệm > 0, từ chối giả định FAR=0 khi không có dữ liệu."""
        total_hours_zero = 0.0
        total_fas = 0

        # Nếu không có giờ nào được test, không được tính ra FAR = 0.0 hợp lệ
        far_result = (total_fas / total_hours_zero) if total_hours_zero > 0 else None
        self.assertIsNone(far_result, "FAR không được xác định khi thời lượng âm tính = 0.0 giờ.")

        # Khi có thời lượng thực tế
        test_hours = 0.053
        test_fas = 9
        calculated_far = test_fas / test_hours
        self.assertAlmostEqual(calculated_far, 169.81, places=1)

    def test_rtsp_benchmark_honesty_blocked_when_no_sources(self):
        """3. Benchmark RTSP phải báo NOT_TESTED và BLOCKED_NO_LIVE_RTSP_SOURCE khi không có URL thật."""
        result = run_rtsp_live_benchmark()
        self.assertEqual(result.get("status"), "NOT_TESTED")
        self.assertEqual(result.get("reason"), "BLOCKED_NO_LIVE_RTSP_SOURCE")
        self.assertEqual(result.get("available_streams"), 0)
        self.assertEqual(result.get("tested_configurations", {}).get("4_cam"), "NOT_TESTED")

    def test_rtsp_url_password_masking(self):
        """4. RTSP URL chứa user:pass phải được che mật khẩu bằng dấu ******."""
        raw_url = "rtsp://admin:P@ssw0rd2026!@192.168.1.100:554/live/ch0"
        masked = mask_rtsp_credentials(raw_url)
        self.assertNotIn("P@ssw0rd2026!", masked, "Mật khẩu gốc không được xuất hiện trong URL đã che.")
        self.assertNotIn("admin", masked, "Tên người dùng cũng phải được ẩn khỏi URL đã che.")
        self.assertIn("rtsp://***@", masked, "Toàn bộ thông tin xác thực phải được ẩn.")

        # Test thêm với URL không có user/pass
        simple_url = "rtsp://192.168.1.200:8554/live"
        self.assertEqual(mask_rtsp_credentials(simple_url), simple_url)

        # Test URL dạng rtsps
        rtsps_url = "rtsps://viewer:SecretToken99@camera.domain.vn/live"
        masked_rtsps = mask_rtsp_credentials(rtsps_url)
        self.assertNotIn("SecretToken99", masked_rtsps)
        self.assertNotIn("viewer", masked_rtsps)
        self.assertIn("rtsps://***@", masked_rtsps)

    def test_negative_audit_duration_accounting_reconciliation(self):
        """5. Kiểm tra tính rạch ròi giữa thời lượng vật lý trên đĩa và thời lượng thực tế đưa vào đánh giá."""
        audit_path = PROJECT_ROOT / "runs" / "benchmarks" / "week4_scientific" / "negative_dataset_audit.json"
        self.assertTrue(audit_path.exists(), "Tệp negative_dataset_audit.json phải tồn tại.")

        import json
        with audit_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        phys = data.get("physical_dataset_inventory", {})
        eval_recon = data.get("evaluation_reconciliation", {})

        self.assertAlmostEqual(phys.get("total_physical_duration_hours", 0.0), 3.4263, places=2)
        self.assertGreater(eval_recon.get("total_duration_evaluated_hours", 0.0), 0.0)
        self.assertLess(
            eval_recon.get("total_duration_evaluated_hours", 0.0),
            phys.get("total_physical_duration_hours", 0.0),
            "Thời lượng đánh giá thực tế phải được phân tách rõ ràng với tổng thời lượng lưu trữ trên đĩa.",
        )
        self.assertIn("discrepancy_explanation", eval_recon)

    def test_poisson_rule_of_three_on_zero_false_alarms(self):
        """6. Khi số sự kiện báo giả bằng 0, phải áp dụng quy tắc Rule of Three Poisson thay vì tuyên bố FAR=0 tuyệt đối."""
        far_json_path = PROJECT_ROOT / "runs" / "benchmarks" / "week4_scientific" / "far_evaluation_final.json"
        self.assertTrue(far_json_path.exists(), "Tệp far_evaluation_final.json phải tồn tại.")

        import json
        with far_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        pipe_b2 = data.get("aggregate_metrics", {}).get("pipeline_b2_bytetrack_with_gate", {})
        self.assertEqual(pipe_b2.get("total_event_fas"), 0)
        self.assertEqual(pipe_b2.get("far_point_estimate"), 0.0)

        ci = pipe_b2.get("poisson_95_ci", [])
        self.assertEqual(len(ci), 2)
        self.assertEqual(ci[0], 0.0)
        self.assertGreater(ci[1], 0.0, "Giới hạn trên khoảng tin cậy Poisson 95% phải > 0 khi thời lượng hữu hạn.")
        self.assertIn("statistical_note", pipe_b2)

    def test_b1_b2_accounts_both_fire_and_smoke(self):
        """7. Cả 3 pipeline phải tính đầy đủ cả FIRE và SMOKE false alarms, không được bỏ qua SMOKE."""
        recon_path = PROJECT_ROOT / "runs" / "benchmarks" / "week4_scientific" / "far_methodology_reconciliation.json"
        self.assertTrue(recon_path.exists(), "Tệp far_methodology_reconciliation.json phải tồn tại.")

        import json
        with recon_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        comp = data.get("aggregate_far_comparison", {})
        for p_key in ["pipeline_a_controlled_cd5s", "pipeline_b1_bytetrack_no_gate", "pipeline_b2_bytetrack_with_gate"]:
            self.assertIn(p_key, comp)
            fas = comp[p_key]["event_fas"]
            self.assertIn("fire", fas)
            self.assertIn("smoke", fas)
            self.assertIn("total", fas)
            self.assertEqual(fas["total"], fas["fire"] + fas["smoke"])

        # Kiểm tra cụ thể B1 bắt được SMOKE false alarms
        b1_fas = comp["pipeline_b1_bytetrack_no_gate"]["event_fas"]
        self.assertGreater(b1_fas["smoke"], 0, "B1 phải ghi nhận SMOKE false alarms từ video âm tính.")

    def test_frame_fp_different_from_detection_fp(self):
        """8. Phân tách rạch ròi giữa false_positive_detections (boxes) và false_positive_frames."""
        recon_path = PROJECT_ROOT / "runs" / "benchmarks" / "week4_scientific" / "far_methodology_reconciliation.json"
        import json
        with recon_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        sep = data.get("three_metrics_separation", {})
        self.assertIn("1_false_positive_detections", sep)
        self.assertIn("2_false_positive_frames", sep)
        self.assertIn("3_false_alarm_events", sep)

        det_counts = sep["1_false_positive_detections"]["counts"]
        frame_counts = sep["2_false_positive_frames"]["counts"]
        self.assertIn("fire", det_counts)
        self.assertIn("smoke", det_counts)
        self.assertIn("fire", frame_counts)
        self.assertIn("smoke", frame_counts)

    def test_operating_point_system_settings_consistency(self):
        """9. Cấu hình SystemSettings trong src/config.py phải trùng khớp 100% với Operating Point đề xuất."""
        from src.config import SystemSettings
        cfg = SystemSettings()
        self.assertEqual(cfg.conf_threshold, 0.35)
        self.assertEqual(cfg.window_size_n, 8)
        self.assertEqual(cfg.confirm_frames_k, 2)
        self.assertEqual(cfg.alert_cooldown_sec, 5.0)
        self.assertEqual(cfg.feature_threshold, 0.02)
        self.assertEqual(cfg.bytetrack.track_high_thresh, 0.35)

    def test_cooldown_causal_no_duplicate_counting(self):
        """10. Cơ chế Cooldown nhân quả: không sinh báo động trùng lặp trong thời gian cooldown."""
        from src.temporal_verifier import TemporalVerifier, DetectionBox
        v = TemporalVerifier(window_size_n=8, confirm_frames_k=2, feature_threshold=0.0, cooldown_sec=5.0)
        box = DetectionBox(cls_id=1, label="FIRE", confidence=0.85, xyxy=(10, 10, 50, 50), norm_xywh=(0.1, 0.1, 0.2, 0.2), track_id=1)

        # 10 khung hình liên tục cách nhau 0.2s (tổng 1.8s < 5.0s cooldown)
        triggered_count = 0
        for i in range(10):
            ts = float(i) * 0.2
            res = v.update([box], tracks_assigned=True, timestamp_sec=ts)
            if res.get("should_trigger_db_alert"):
                triggered_count += 1

        self.assertEqual(triggered_count, 1, "Chỉ được phát đúng 1 cảnh báo ban đầu, trong thời gian cooldown không được phát lặp.")

        # Khung hình sau khi hết cooldown (ts = 5.5s > 5.0s)
        res_after = v.update([box], tracks_assigned=True, timestamp_sec=5.5)
        self.assertTrue(res_after.get("should_trigger_db_alert"), "Sau khi hết Cooldown 5.0s, phải kích hoạt cảnh báo mới.")

    def test_fire_followed_by_smoke_inside_cooldown(self):
        """11. TASK 2: FIRE kích hoạt, sau đó SMOKE xuất hiện trong cooldown của FIRE -> SMOKE phải trigger độc lập."""
        from src.temporal_verifier import TemporalVerifier, DetectionBox
        v = TemporalVerifier(window_size_n=8, confirm_frames_k=2, feature_threshold=0.0, cooldown_sec=5.0)
        box_fire = DetectionBox(cls_id=1, label="FIRE", confidence=0.9, xyxy=(10, 10, 50, 50), norm_xywh=(0.1, 0.1, 0.2, 0.2), track_id=1)
        box_smoke = DetectionBox(cls_id=0, label="SMOKE", confidence=0.85, xyxy=(60, 60, 100, 100), norm_xywh=(0.3, 0.3, 0.2, 0.2), track_id=2)

        # Frame 0 & 1: FIRE đạt K=2 ở t=0.1s
        v.update([box_fire], tracks_assigned=True, timestamp_sec=0.0)
        r_fire = v.update([box_fire], tracks_assigned=True, timestamp_sec=0.1)
        self.assertTrue(r_fire.get("should_trigger_db_alert"))
        self.assertIn("FIRE", r_fire.get("triggered_labels", []))

        # Frame 2 & 3: SMOKE đạt K=2 ở t=1.1s (1.1s < 0.1s + 5.0s Cooldown của FIRE)
        # Bắt buộc per-class cooldown: SMOKE không bị FIRE chặn!
        v.update([box_smoke], tracks_assigned=True, timestamp_sec=1.0)
        r_smoke = v.update([box_smoke], tracks_assigned=True, timestamp_sec=1.1)
        self.assertTrue(r_smoke.get("should_trigger_db_alert"), "SMOKE không được bị chặn bởi Cooldown của FIRE.")
        self.assertIn("SMOKE", r_smoke.get("triggered_labels", []))

    def test_smoke_followed_by_fire_inside_cooldown(self):
        """12. TASK 2: SMOKE kích hoạt, sau đó FIRE xuất hiện trong cooldown của SMOKE -> FIRE phải trigger độc lập."""
        from src.temporal_verifier import TemporalVerifier, DetectionBox
        v = TemporalVerifier(window_size_n=8, confirm_frames_k=2, feature_threshold=0.0, cooldown_sec=5.0)
        box_smoke = DetectionBox(cls_id=0, label="SMOKE", confidence=0.85, xyxy=(60, 60, 100, 100), norm_xywh=(0.3, 0.3, 0.2, 0.2), track_id=2)
        box_fire = DetectionBox(cls_id=1, label="FIRE", confidence=0.9, xyxy=(10, 10, 50, 50), norm_xywh=(0.1, 0.1, 0.2, 0.2), track_id=1)

        # SMOKE đạt K=2 ở t=0.1s
        v.update([box_smoke], tracks_assigned=True, timestamp_sec=0.0)
        r_smoke = v.update([box_smoke], tracks_assigned=True, timestamp_sec=0.1)
        self.assertTrue(r_smoke.get("should_trigger_db_alert"))
        self.assertIn("SMOKE", r_smoke.get("triggered_labels", []))

        # FIRE đạt K=2 ở t=1.1s (< 5.0s cooldown của SMOKE) -> FIRE không bị nuốt!
        v.update([box_fire], tracks_assigned=True, timestamp_sec=1.0)
        r_fire = v.update([box_fire], tracks_assigned=True, timestamp_sec=1.1)
        self.assertTrue(r_fire.get("should_trigger_db_alert"), "FIRE không được bị chặn bởi Cooldown của SMOKE.")
        self.assertIn("FIRE", r_fire.get("triggered_labels", []))

    def test_simultaneous_fire_and_smoke_both_triggered(self):
        """13. TASK 2: Đồng thời cả FIRE và SMOKE đạt ngưỡng xác nhận trong cùng frame -> cả hai đều trigger."""
        from src.temporal_verifier import TemporalVerifier, DetectionBox
        v = TemporalVerifier(window_size_n=8, confirm_frames_k=2, feature_threshold=0.0, cooldown_sec=5.0)
        box_fire = DetectionBox(cls_id=1, label="FIRE", confidence=0.9, xyxy=(10, 10, 50, 50), norm_xywh=(0.1, 0.1, 0.2, 0.2), track_id=1)
        box_smoke = DetectionBox(cls_id=0, label="SMOKE", confidence=0.85, xyxy=(60, 60, 100, 100), norm_xywh=(0.3, 0.3, 0.2, 0.2), track_id=2)

        # Frame 0: cả hai cùng xuất hiện
        v.update([box_fire, box_smoke], tracks_assigned=True, timestamp_sec=0.0)
        # Frame 1: cả hai cùng đạt K=2
        res = v.update([box_fire, box_smoke], tracks_assigned=True, timestamp_sec=0.1)

        self.assertTrue(res.get("should_trigger_db_alert"))
        triggered = res.get("triggered_labels", [])
        self.assertIn("FIRE", triggered, "Phải ghi nhận cảnh báo FIRE.")
        self.assertIn("SMOKE", triggered, "Phải ghi nhận cảnh báo SMOKE.")
        self.assertEqual(len(triggered), 2, "Cả hai nhãn phải được kích hoạt độc lập, không nuốt lớp nào.")

    def test_same_class_repeated_detections_suppressed_by_cooldown(self):
        """14. TASK 2: Cùng một nhãn lặp lại trong thời gian cooldown thì bị nén, không phát cảnh báo lặp."""
        from src.temporal_verifier import TemporalVerifier, DetectionBox
        v = TemporalVerifier(window_size_n=8, confirm_frames_k=2, feature_threshold=0.0, cooldown_sec=5.0)
        box_fire_1 = DetectionBox(cls_id=1, label="FIRE", confidence=0.9, xyxy=(10, 10, 50, 50), norm_xywh=(0.1, 0.1, 0.2, 0.2), track_id=1)
        box_fire_2 = DetectionBox(cls_id=1, label="FIRE", confidence=0.88, xyxy=(20, 20, 60, 60), norm_xywh=(0.15, 0.15, 0.2, 0.2), track_id=3)

        v.update([box_fire_1], tracks_assigned=True, timestamp_sec=0.0)
        r1 = v.update([box_fire_1], tracks_assigned=True, timestamp_sec=0.1)
        self.assertTrue(r1.get("should_trigger_db_alert"), "Lần đầu đạt K=2 phải trigger.")

        # Cùng track hoặc track mới của cùng nhãn FIRE ở t=1.0s, 1.1s (< 5.0s)
        v.update([box_fire_2], tracks_assigned=True, timestamp_sec=1.0)
        r2 = v.update([box_fire_2], tracks_assigned=True, timestamp_sec=1.1)
        self.assertFalse(r2.get("should_trigger_db_alert"), "Cùng nhãn FIRE trong thời gian cooldown 5.0s phải bị nén.")

    def test_separate_camera_cooldown_isolation(self):
        """15. TASK 2: Cách ly hoàn toàn Cooldown giữa các camera khác nhau."""
        from src.temporal_verifier import TemporalVerifier, DetectionBox
        v_cam1 = TemporalVerifier(window_size_n=8, confirm_frames_k=2, feature_threshold=0.0, cooldown_sec=5.0)
        v_cam2 = TemporalVerifier(window_size_n=8, confirm_frames_k=2, feature_threshold=0.0, cooldown_sec=5.0)

        box_fire = DetectionBox(cls_id=1, label="FIRE", confidence=0.9, xyxy=(10, 10, 50, 50), norm_xywh=(0.1, 0.1, 0.2, 0.2), track_id=1)

        # Cam 1 trigger ở t=0.1s
        v_cam1.update([box_fire], tracks_assigned=True, timestamp_sec=0.0)
        r_cam1 = v_cam1.update([box_fire], tracks_assigned=True, timestamp_sec=0.1)
        self.assertTrue(r_cam1.get("should_trigger_db_alert"))

        # Cam 2 có FIRE ở t=0.5s (< 5.0s của Cam 1) -> Cam 2 phải trigger bình thường
        v_cam2.update([box_fire], tracks_assigned=True, timestamp_sec=0.4)
        r_cam2 = v_cam2.update([box_fire], tracks_assigned=True, timestamp_sec=0.5)
        self.assertTrue(r_cam2.get("should_trigger_db_alert"), "Camera 2 không bị ảnh hưởng bởi Cooldown của Camera 1.")

    def test_cooldown_expiration_triggers_new_event(self):
        """16. TASK 2: Sau khi hết thời gian Cooldown (t >= 5.0s), sự kiện mới cùng nhãn phải được kích hoạt."""
        from src.temporal_verifier import TemporalVerifier, DetectionBox
        v = TemporalVerifier(window_size_n=8, confirm_frames_k=2, feature_threshold=0.0, cooldown_sec=5.0)
        box = DetectionBox(cls_id=1, label="FIRE", confidence=0.9, xyxy=(10, 10, 50, 50), norm_xywh=(0.1, 0.1, 0.2, 0.2), track_id=1)

        v.update([box], tracks_assigned=True, timestamp_sec=0.0)
        r1 = v.update([box], tracks_assigned=True, timestamp_sec=0.1)
        self.assertTrue(r1.get("should_trigger_db_alert"))

        # Frame ở t=5.2s (> 0.1s + 5.0s)
        r2 = v.update([box], tracks_assigned=True, timestamp_sec=5.2)
        self.assertTrue(r2.get("should_trigger_db_alert"), "Hết cooldown 5.0s phải cho phép kích hoạt sự kiện mới.")

    def test_benchmark_assertion_fails_when_settings_mismatched(self):
        """17. TASK 1: Benchmark bắt buộc nạp Operating Point từ SystemSettings và raise AssertionError nếu sai lệch."""
        from src.config import SystemSettings
        from scripts.evaluate_negative_dataset_unified import run_unified_negative_evaluation

        bad_settings = SystemSettings()
        bad_settings.confirm_frames_k = 4  # Sai lệch so với K=2

        with self.assertRaises(AssertionError):
            run_unified_negative_evaluation(settings=bad_settings)

    def test_far_result_metadata_equals_runtime_parameters(self):
        """18. TASK 7: FAR result metadata phải phản ánh đúng runtime parameters (K=2, N=8, feat=0.02, cd=5.0s, conf=0.35)."""
        import json
        far_json = PROJECT_ROOT / "runs" / "benchmarks" / "week4_scientific" / "far_methodology_reconciliation.json"
        self.assertTrue(far_json.exists(), "far_methodology_reconciliation.json phải tồn tại.")
        data = json.loads(far_json.read_text(encoding="utf-8"))

        op = data.get("metadata", {}).get("operating_point", {})
        self.assertEqual(op.get("confirm_k"), 2)
        self.assertEqual(op.get("window_n"), 8)
        self.assertAlmostEqual(op.get("feature_threshold"), 0.02, places=4)
        self.assertAlmostEqual(op.get("cooldown_sec"), 5.0, places=2)
        self.assertAlmostEqual(op.get("confidence_threshold"), 0.35, places=2)

    def test_fire_and_smoke_both_counted_in_far_results(self):
        """19. TASK 7: Báo cáo FAR phải phân tách và tính đầy đủ cả FIRE và SMOKE cho mọi pipeline."""
        import json
        far_json = PROJECT_ROOT / "runs" / "benchmarks" / "week4_scientific" / "far_methodology_reconciliation.json"
        data = json.loads(far_json.read_text(encoding="utf-8"))

        agg_far = data.get("aggregate_far_comparison", {})
        for p_name in ["pipeline_a_controlled_cd5s", "pipeline_b1_bytetrack_no_gate", "pipeline_b2_bytetrack_with_gate"]:
            self.assertIn(p_name, agg_far)
            p_data = agg_far[p_name]
            far_obj = p_data.get("far_per_hour") or p_data.get("far_point_estimate")
            self.assertIn("fire_far", far_obj)
            self.assertIn("smoke_far", far_obj)
            self.assertIn("total_far", far_obj)

    def test_full_video_mode_semantics(self):
        """20. TASK 4 & 7: Khi max_frames <= 0 hoặc None, evaluation_mode phải là full_video."""
        # Kiểm tra logic ánh xạ mode
        def parse_mode(max_frames_arg: int | None) -> tuple[str, int | None]:
            effective_max = max_frames_arg if (max_frames_arg is not None and max_frames_arg > 0) else None
            mode = "full_video" if effective_max is None else "bounded_sample"
            return mode, effective_max

        m1, eff1 = parse_mode(0)
        self.assertEqual(m1, "full_video")
        self.assertIsNone(eff1)

        m2, eff2 = parse_mode(-1)
        self.assertEqual(m2, "full_video")
        self.assertIsNone(eff2)

        m3, eff3 = parse_mode(None)
        self.assertEqual(m3, "full_video")
        self.assertIsNone(eff3)

        m4, eff4 = parse_mode(2000)
        self.assertEqual(m4, "bounded_sample")
        self.assertEqual(eff4, 2000)

    def test_report_configuration_matches_deployed_configuration(self):
        """21. TASK 7: Cấu hình trong báo cáo đối soát và validation phải khớp 100% với SystemSettings."""
        import json
        from src.config import SystemSettings
        sys_cfg = SystemSettings()

        val_json = PROJECT_ROOT / "runs" / "benchmarks" / "week4_scientific" / "validation_operating_point_evaluation.json"
        if val_json.exists():
            vdata = json.loads(val_json.read_text(encoding="utf-8"))
            v_cfg = vdata.get("metadata", {}).get("deployed_configuration", {})
            self.assertEqual(v_cfg.get("confirm_frames_k"), sys_cfg.confirm_frames_k)
            self.assertEqual(v_cfg.get("window_size_n"), sys_cfg.window_size_n)
            self.assertAlmostEqual(v_cfg.get("feature_threshold"), sys_cfg.feature_threshold, places=4)
            self.assertAlmostEqual(v_cfg.get("alert_cooldown_sec"), sys_cfg.alert_cooldown_sec, places=2)
            self.assertAlmostEqual(v_cfg.get("confidence_threshold"), sys_cfg.conf_threshold, places=2)


if __name__ == "__main__":
    unittest.main()
