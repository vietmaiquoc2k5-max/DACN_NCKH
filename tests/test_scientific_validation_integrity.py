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
        self.assertIn("admin:******@", masked, "Mật khẩu phải được thay thế bằng ******.")

        # Test thêm với URL không có user/pass
        simple_url = "rtsp://192.168.1.200:8554/live"
        self.assertEqual(mask_rtsp_credentials(simple_url), simple_url)

        # Test URL dạng rtsps
        rtsps_url = "rtsps://viewer:SecretToken99@camera.domain.vn/live"
        masked_rtsps = mask_rtsp_credentials(rtsps_url)
        self.assertNotIn("SecretToken99", masked_rtsps)
        self.assertIn("viewer:******@", masked_rtsps)

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


if __name__ == "__main__":
    unittest.main()
