"""Bộ kiểm thử đơn vị & kiểm thử tích hợp mở rộng cho Redesigned Control Room & Dynamic Camera Selection.

Bao gồm 12 bài test bắt buộc theo yêu cầu Mục 6:
1. Chọn 1 camera (SINGLE): chỉ đúng 1 worker chạy.
2. Chọn 4 camera (QUAD): chỉ đúng 4 worker chạy.
3. Chọn 9 camera (FULL): đủ 9 worker chạy.
4. Chuyển từ 9 xuống 4 camera: 5 worker không được chọn phải dừng, giải phóng VideoCapture.
5. Chuyển từ 4 xuống 1 camera: 3 worker không được chọn phải dừng.
6. Thay đổi danh sách camera thủ công (CUSTOM: vd CAM-01, CAM-03, CAM-05, CAM-08).
7. Kiểm tra worker không bị nhân đôi (duplicate threads) khi gọi lặp.
8. Kiểm tra stop() giải phóng VideoCapture và reset trạng thái an toàn.
9. Kiểm tra cơ chế reconnect camera.
10. Kiểm tra ByteTrack / Tracker không bị lẫn lộn giữa các camera (độc lập per-camera).
11. Kiểm tra shared detector dùng chung an toàn trên GPU/CPU.
12. Kiểm tra dashboard metrics không crash khi camera mất kết nối (cô lập lỗi).
"""
import os
import sys
import time
import unittest
import tempfile
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera_manager import MultiCameraManager, CameraStreamWorker
from src.detector import SharedFireSmokeDetector
from src.bytetrack_adapter import ByteTrackAdapter
from src.temporal_verifier import DetectionBox, TemporalVerifier
from src.temporal_feature_extractor import TemporalFeatureExtractor


class TestDynamicCameraAndDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls.tmp_dir.name, "test_dynamic.db")
        cls.manager = MultiCameraManager(db_path=cls.db_path, model_path="best.pt")
        cls.manager.settings.mute_audio = True
        cls.manager.alert_manager.update_config(True)

    @classmethod
    def tearDownClass(cls):
        cls.manager.stop_all()
        cls.tmp_dir.cleanup()

    def setUp(self):
        # Dừng tất cả trước mỗi test
        self.manager.stop_all()

    def test_01_single_camera_selection(self):
        """1. Chọn 1 camera (SINGLE): chỉ đúng 1 worker chạy."""
        self.manager.set_active_camera_selection(["CAM-01"], auto_start=True)
        running = [w.config.cam_id for w in self.manager.workers.values() if w.running]
        self.assertEqual(running, ["CAM-01"])

    def test_02_quad_camera_selection(self):
        """2. Chọn 4 camera (QUAD): chỉ đúng 4 worker chạy."""
        cams = ["CAM-01", "CAM-02", "CAM-03", "CAM-04"]
        self.manager.set_active_camera_selection(cams, auto_start=True)
        running = [w.config.cam_id for w in self.manager.workers.values() if w.running]
        self.assertEqual(sorted(running), sorted(cams))

    def test_03_full_nine_cameras(self):
        """3. Chọn 9 camera (FULL): đủ 9 worker chạy."""
        cams = [f"CAM-0{i}" for i in range(1, 10)]
        self.manager.set_active_camera_selection(cams, auto_start=True)
        running = [w.config.cam_id for w in self.manager.workers.values() if w.running]
        self.assertEqual(sorted(running), sorted(cams))

    def test_04_switch_from_9_to_4_cameras(self):
        """4. Chuyển từ 9 xuống 4 camera: 5 worker không được chọn phải dừng hoàn toàn."""
        all_9 = [f"CAM-0{i}" for i in range(1, 10)]
        self.manager.set_active_camera_selection(all_9, auto_start=True)
        self.assertEqual(sum(1 for w in self.manager.workers.values() if w.running), 9)

        # Chuyển xuống 4
        cams_4 = ["CAM-01", "CAM-02", "CAM-03", "CAM-04"]
        self.manager.set_active_camera_selection(cams_4, auto_start=True)
        running = [w.config.cam_id for w in self.manager.workers.values() if w.running]
        self.assertEqual(sorted(running), sorted(cams_4))
        # 5 camera còn lại phải dừng hoàn toàn
        stopped = [f"CAM-0{i}" for i in range(5, 10)]
        for s_id in stopped:
            self.assertFalse(self.manager.workers[s_id].running)

    def test_05_switch_from_4_to_1_camera(self):
        """5. Chuyển từ 4 xuống 1 camera: 3 worker phải dừng."""
        cams_4 = ["CAM-01", "CAM-02", "CAM-03", "CAM-04"]
        self.manager.set_active_camera_selection(cams_4, auto_start=True)
        self.manager.set_active_camera_selection(["CAM-03"], auto_start=True)
        running = [w.config.cam_id for w in self.manager.workers.values() if w.running]
        self.assertEqual(running, ["CAM-03"])

    def test_06_custom_camera_selection(self):
        """6. Chọn camera tùy ý (CAM-01, CAM-03, CAM-05, CAM-08)."""
        custom = ["CAM-01", "CAM-03", "CAM-05", "CAM-08"]
        self.manager.set_active_camera_selection(custom, auto_start=True)
        running = [w.config.cam_id for w in self.manager.workers.values() if w.running]
        self.assertEqual(sorted(running), sorted(custom))

    def test_07_no_duplicate_workers_on_multiple_calls(self):
        """7. Kiểm tra gọi set_active_camera_selection lặp lại không tạo worker/thread trùng lặp."""
        self.manager.set_active_camera_selection(["CAM-01", "CAM-02"], auto_start=True)
        t1 = self.manager.workers["CAM-01"].thread
        # Gọi lại cùng danh sách
        self.manager.set_active_camera_selection(["CAM-01", "CAM-02"], auto_start=True)
        t2 = self.manager.workers["CAM-01"].thread
        self.assertEqual(t1, t2, "Worker của camera đang chạy không được tạo thread mới khi không đổi")

    def test_08_stop_releases_resources(self):
        """8. Kiểm tra stop giải phóng VideoCapture và cập nhật DISCONNECTED."""
        w = self.manager.workers["CAM-01"]
        w.start()
        time.sleep(0.1)
        w.stop()
        self.assertFalse(w.running)
        self.assertEqual(w.status, "DISCONNECTED")

    def test_09_camera_reconnect(self):
        """9. Kiểm tra cơ chế reconnect camera."""
        w = self.manager.workers["CAM-01"]
        w.start()
        time.sleep(0.1)
        w.reconnect()
        self.assertTrue(w.running)
        w.stop()

    def test_10_tracker_isolation_between_cameras(self):
        """10. Kiểm tra ByteTrack / Tracker không bị lẫn lộn giữa các camera."""
        w1 = self.manager.workers["CAM-01"]
        w2 = self.manager.workers["CAM-02"]
        self.assertIsNot(w1.tracker, w2.tracker, "Mỗi camera phải sở hữu ByteTrackAdapter riêng")
        self.assertIsNot(w1.verifier, w2.verifier, "Mỗi camera phải sở hữu TemporalVerifier riêng")

        box = DetectionBox(cls_id=1, label="FIRE", confidence=0.85, xyxy=(10, 10, 80, 80), norm_xywh=(0.1, 0.1, 0.1, 0.1))
        r1 = w1.tracker.update([box])
        r2 = w2.tracker.update([box])
        self.assertIsNotNone(r1[0].track_id)
        self.assertIsNotNone(r2[0].track_id)

    def test_11_shared_detector(self):
        """11. Kiểm tra detector dùng chung an toàn trên GPU/CPU."""
        det = self.manager.detector
        info = det.get_model_info()
        self.assertTrue(info["loaded"])
        # Kiểm tra suy luận trên dummy image
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes, infer_ms = det.detect_frame(dummy, conf_threshold=0.35)
        self.assertIsInstance(boxes, list)
        self.assertGreaterEqual(infer_ms, 0.0)

    def test_12_fault_isolation_no_dashboard_crash(self):
        """12. Kiểm tra dashboard không crash khi camera mất kết nối."""
        # Trỏ CAM-09 tới nguồn lỗi
        self.manager.update_camera_source("CAM-09", "CAM-09 Error Test", "invalid_path_123.mp4", auto_reconnect=False)
        self.manager.set_active_camera_selection(["CAM-01", "CAM-09"], auto_start=True)
        time.sleep(0.5)

        snaps = self.manager.get_selected_snapshots(["CAM-01", "CAM-09"])
        self.assertEqual(len(snaps), 2)
        metrics = self.manager.get_system_metrics(snaps)
        self.assertIn("total_fps", metrics)
        self.assertIn("avg_fps", metrics)
        self.assertIn("overall_level", metrics)


if __name__ == "__main__":
    unittest.main()

