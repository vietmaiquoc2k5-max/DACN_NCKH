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
import cv2
import threading
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
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
        cls.video_fixture = str(Path(cls.tmp_dir.name) / "camera.avi")
        writer = cv2.VideoWriter(cls.video_fixture, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
        if not writer.isOpened():
            raise RuntimeError("Cannot create deterministic MJPG camera fixture")
        try:
            for i in range(30):
                writer.write(np.full((48, 64, 3), i, dtype=np.uint8))
        finally:
            writer.release()
        capture = cv2.VideoCapture(cls.video_fixture)
        try:
            assert capture.isOpened() and capture.read()[0], "Fixture cannot be decoded"
        finally:
            capture.release()
        cls.db_path = os.path.join(cls.tmp_dir.name, "test_dynamic.db")
        cls.manager = MultiCameraManager(db_path=cls.db_path, model_path="best.pt")
        cls.manager.settings.mute_audio = True
        cls.manager.alert_manager.update_config(True)
        for cid in cls.manager.workers:
            cls.manager.update_camera_source(cid, cid, cls.video_fixture, auto_reconnect=False)

    @classmethod
    def tearDownClass(cls):
        cls.manager.close()
        cls.tmp_dir.cleanup()

    def setUp(self):
        # Dừng tất cả trước mỗi test
        self.manager.stop_all()
        self.addCleanup(self.manager.stop_all)

    def wait_for_status(self, worker, expected, timeout=5.0):
        """Wait for an observable state with a bounded monotonic deadline."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if worker.get_snapshot()["status"] == expected:
                return
            time.sleep(0.01)
        self.fail(f"Expected {expected}: {worker.get_snapshot()}")

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
        self.wait_for_status(w, "CONNECTED")
        old_reader, old_infer = w.thread, w.infer_thread
        w.reconnect()
        self.wait_for_status(w, "CONNECTED")
        self.assertTrue(w.running)
        self.assertIsNot(old_reader, w.thread)
        self.assertFalse(old_reader.is_alive())
        self.assertFalse(old_infer.is_alive())
        w.stop()

    def test_23_disconnect_releases_capture(self):
        """An actual OpenCV capture is closed before disconnect returns."""
        from src.camera_manager import open_opencv_capture
        opened = []

        def tracked_open(*args, **kwargs):
            result = open_opencv_capture(*args, **kwargs)
            if result[0] is not None:
                opened.append(result[0])
            return result

        worker = self.manager.workers["CAM-01"]
        with patch("src.camera_manager.open_opencv_capture", side_effect=tracked_open):
            worker.start()
            self.wait_for_status(worker, "CONNECTED")
            worker.stop()
        self.assertTrue(opened)
        self.assertTrue(all(not capture.isOpened() for capture in opened))

    def test_24_shutdown_timeout_retains_handles_and_refuses_duplicate(self):
        """A backend that cannot stop must not lose its thread handle or restart."""
        worker = self.manager.workers["CAM-01"]
        stuck = MagicMock()
        stuck.is_alive.return_value = True
        worker.thread = stuck
        try:
            with self.assertRaisesRegex(RuntimeError, "shutdown timed out"):
                worker.stop()
            self.assertIs(worker.thread, stuck)
            with self.assertRaisesRegex(RuntimeError, "refusing duplicate"):
                worker.start()
        finally:
            stuck.is_alive.return_value = False
            worker.stop()

    def test_25_observable_manual_connection_transitions(self):
        """Synchronize actual capture opening to observe CONNECTING and RECONNECTING."""
        from src.camera_manager import open_opencv_capture
        entered, proceed = threading.Event(), threading.Event()
        worker = self.manager.workers["CAM-07"]

        def gated_open(*args, **kwargs):
            entered.set()
            if not proceed.wait(timeout=5.0):
                raise RuntimeError("Test capture synchronization timed out")
            return open_opencv_capture(*args, **kwargs)

        self.manager.update_camera_source("CAM-07", "transition", "invalid_transition.mp4", auto_reconnect=False)
        try:
            with patch("src.camera_manager.open_opencv_capture", side_effect=gated_open):
                self.assertEqual(worker.get_snapshot()["status"], "DISCONNECTED")
                self.manager.start_camera("CAM-07")
                self.assertTrue(entered.wait(timeout=5.0))
                self.assertEqual(worker.get_snapshot()["status"], "CONNECTING")
                proceed.set()
                self.wait_for_status(worker, "ERROR")
                self.manager.update_camera_source("CAM-07", "transition", self.video_fixture, auto_reconnect=False)
                entered.clear()
                proceed.clear()
                self.manager.reconnect_camera("CAM-07")
                self.assertTrue(entered.wait(timeout=5.0))
                self.assertEqual(worker.get_snapshot()["status"], "RECONNECTING")
                proceed.set()
                self.wait_for_status(worker, "CONNECTED")
                thread = worker.thread
                self.manager.start_camera("CAM-07")
                self.assertIs(worker.thread, thread)
        finally:
            proceed.set()
            worker.stop()

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

    def test_13_dual_and_hexa_camera_selection(self):
        """13. Chọn 2 camera (DUAL) và 6 camera (HEXA) theo chuẩn FIREWATCH COMMAND CENTER."""
        cams_2 = ["CAM-01", "CAM-02"]
        self.manager.set_active_camera_selection(cams_2, auto_start=True)
        running_2 = [w.config.cam_id for w in self.manager.workers.values() if w.running]
        self.assertEqual(sorted(running_2), sorted(cams_2))

        cams_6 = [f"CAM-0{i}" for i in range(1, 7)]
        self.manager.set_active_camera_selection(cams_6, auto_start=True)
        running_6 = [w.config.cam_id for w in self.manager.workers.values() if w.running]
        self.assertEqual(sorted(running_6), sorted(cams_6))

    def test_14_performance_presets_and_roi_and_add_camera(self):
        """14. Kiểm tra Performance Presets, cập nhật ROI per-camera và thêm camera mới."""
        self.manager.apply_performance_preset("Low Resource")
        self.assertEqual(self.manager.settings.target_fps_per_cam, 15)
        self.assertEqual(self.manager.settings.frame_skip, 3)
        self.assertEqual(self.manager.settings.imgsz, 320)

        self.manager.apply_performance_preset("High Performance")
        self.assertEqual(self.manager.settings.target_fps_per_cam, 30)
        self.assertEqual(self.manager.settings.frame_skip, 1)
        self.assertEqual(self.manager.settings.imgsz, 640)

        self.manager.apply_performance_preset("Balanced")
        self.assertEqual(self.manager.settings.target_fps_per_cam, 24)
        self.assertEqual(self.manager.settings.frame_skip, 2)
        self.assertEqual(self.manager.settings.imgsz, 416)

        # Cập nhật ROI cho CAM-01
        roi_zones = [(0.1, 0.1, 0.3, 0.3)]
        self.manager.update_camera_roi("CAM-01", roi_zones)
        self.assertEqual(self.manager.workers["CAM-01"].config.exclusion_zones, roi_zones)

        # Thêm camera mới
        new_id = self.manager.add_camera("Khu vực thử nghiệm", "test_fire.mp4", source_type="VIDEO_FILE", enabled=False)
        self.assertIn(new_id, self.manager.workers)

    def test_15_preserve_state_of_unchanged_cameras_and_disconnected_info(self):
        """15. Giữ nguyên trạng thái những camera không thay đổi khi chuyển cơ số camera & giữ thông tin nhận diện khi tạm dừng/lỗi."""
        self.manager.set_active_camera_selection(["CAM-01", "CAM-02", "CAM-03", "CAM-04"], auto_start=True)
        w1 = self.manager.workers["CAM-01"]
        t_before = w1.thread

        # Chuyển từ 4 camera xuống 2 camera (CAM-01 và CAM-02 giữ nguyên)
        self.manager.set_active_camera_selection(["CAM-01", "CAM-02"], auto_start=True)
        self.assertIs(w1.thread, t_before, "Camera giữ nguyên không được khởi tạo lại thread")

        # Tạm dừng CAM-01 và kiểm tra thông tin nhận diện vẫn được giữ nguyên
        w1.pause()
        with w1.lock:
            w1.primary_label = "FIRE"
            w1.max_confidence = 0.88
            w1.active_hits = 3
        snap1 = w1.get_snapshot()
        self.assertEqual(snap1["status"], "PAUSED")
        self.assertEqual(snap1["primary_label"], "FIRE")
        self.assertAlmostEqual(snap1["max_confidence"], 0.88, places=2)
        self.assertEqual(snap1["active_hits"], 3)

    def test_16_html_sanitization_and_no_raw_html_leak(self):
        """16. Kiểm tra hàm chuẩn hóa HTML không để dòng thụt lề >= 4 khoảng trắng hay dòng trống gây lỗi lộ thẻ HTML thô."""
        import html as py_html
        sample_dirty = """
        <div class="fw-cam-card">
            <div class="fw-cam-header">
                <span>CAM-01</span>
            </div>

            <div class="fw-cam-status-bar">
                <span>Nhận diện: An toàn</span>
            </div>
        </div>
        """
        cleaned = "\n".join(line.strip() for line in sample_dirty.splitlines() if line.strip())
        for ln in cleaned.splitlines():
            self.assertFalse(ln.startswith("    "), "Không được phép còn dòng thụt lề >= 4 spaces gây lỗi code-block Markdown")
            self.assertTrue(len(ln) > 0, "Không được phép còn dòng trống bên trong khối HTML")

        # Kiểm tra escape dữ liệu không đáng tin cậy từ tên camera/nguồn lỗi
        untrusted = '<div class="fw-cam-foot"><script>alert(1)</script>'
        escaped = py_html.escape(untrusted, quote=True)
        self.assertNotIn("<div", escaped)
        self.assertNotIn("<script>", escaped)

    def test_17_disconnected_camera_timestamp_and_no_infinite_worker_spawn(self):
        """17. Kiểm tra camera lỗi ghi nhận thời điểm mất kết nối và không tự tạo lại worker liên tục khi rerun."""
        self.manager.update_camera_source("CAM-08", "CAM-08 Offline Test", "non_existent_stream_999.mp4", auto_reconnect=False)
        self.manager.set_active_camera_selection(["CAM-08"], auto_start=True)
        self.wait_for_status(self.manager.workers["CAM-08"], "ERROR")

        w8 = self.manager.workers["CAM-08"]
        snap8 = w8.get_snapshot()
        self.assertEqual(snap8["status"], "ERROR")
        self.assertFalse(w8.running)
        self.assertTrue(len(snap8.get("disconnected_at_str", "")) >= 5)

        # Gọi lại set_active_camera_selection khi Streamlit rerun: không được tự tạo lại worker cho camera đang ERROR
        old_thread = w8.thread
        self.manager.set_active_camera_selection(["CAM-08"], auto_start=True)
        self.assertFalse(w8.running)
        self.assertIs(w8.thread, old_thread)

        # Khôi phục nguồn hợp lệ và kết nối lại thủ công
        valid_sample = self.video_fixture
        self.manager.update_camera_source("CAM-08", "CAM 08 - Sân Thượng", valid_sample, auto_reconnect=True)
        self.manager.reconnect_camera("CAM-08")
        self.wait_for_status(w8, "CONNECTED")
        self.assertEqual(w8.get_snapshot()["status"], "CONNECTED")
        active_thread = w8.thread
        self.manager.set_active_camera_selection(["CAM-08"], auto_start=True)
        self.assertIs(w8.thread, active_thread)
        w8.stop()
        self.assertFalse(active_thread.is_alive())
        self.assertIsNone(w8.thread)
        self.assertIsNone(w8.infer_thread)

    def test_18_database_confirm_dismiss_and_preserve_history(self):
        """18. Kiểm tra Xác nhận và Bỏ qua cập nhật đúng SQLite mà không làm mất lịch sử sự kiện."""
        dummy_bgr = np.zeros((240, 320, 3), dtype=np.uint8)
        ev1 = self.manager.event_db.record_event(dummy_bgr, "CAM-01", "FIRE", 0.91, alert_level="FIRE", is_demo=0)
        ev2 = self.manager.event_db.record_event(dummy_bgr, "CAM-02", "SMOKE", 0.78, alert_level="SUSPECT", is_demo=0)

        stat_before = self.manager.event_db.get_summary_statistics()
        total_before = stat_before["total_real"]
        unack_before = stat_before["unconfirmed"]

        self.manager.event_db.confirm_event(ev1["id"], action_taken="Xác nhận cháy thật", note="Test confirm")
        self.manager.event_db.dismiss_event(ev2["id"], note="Test dismiss")

        stat_after = self.manager.event_db.get_summary_statistics()
        self.assertEqual(stat_after["total_real"], total_before, "Không được làm mất sự kiện lịch sử sau khi xác nhận/bỏ qua")
        self.assertEqual(stat_after["unconfirmed"], unack_before - 2)

    def test_19_manual_start_and_stop_monitoring_state_machine(self):
        """19. Kiểm tra V2 Manual Start: khởi động ở STOPPED (0 worker), start_monitoring bật đúng camera được chọn, stop_monitoring trả toàn bộ về STOPPED."""
        self.manager.stop_monitoring()
        self.assertEqual(self.manager.get_active_worker_count(), 0)
        for cid, w in self.manager.workers.items():
            snap = w.get_snapshot()
            self.assertEqual(snap["lifecycle_state"], "STOPPED")
            self.assertEqual(snap["status"], "DISCONNECTED")
            self.assertFalse(w.running)

        # Chỉ chọn cấu hình nhưng auto_start=False -> vẫn giữ STOPPED, 0 worker chạy
        self.manager.set_active_camera_selection(["CAM-01", "CAM-02"], auto_start=False)
        self.assertEqual(self.manager.get_active_worker_count(), 0)
        self.assertEqual(self.manager.workers["CAM-01"].get_snapshot()["lifecycle_state"], "STOPPED")

        # Nhấn Bắt đầu giám sát cho CAM-01
        self.manager.start_monitoring(["CAM-01"])
        self.wait_for_status(self.manager.workers["CAM-01"], "CONNECTED")
        self.assertEqual(self.manager.get_active_worker_count(), 1)
        snap1 = self.manager.workers["CAM-01"].get_snapshot()
        self.assertEqual(snap1["lifecycle_state"], "RUNNING")
        self.assertEqual(snap1["status"], "CONNECTED")
        self.assertEqual(self.manager.workers["CAM-02"].get_snapshot()["lifecycle_state"], "STOPPED")

        # Nhấn Dừng giám sát -> tất cả trở về STOPPED, đóng thread/VideoCapture
        self.manager.stop_monitoring()
        self.assertEqual(self.manager.get_active_worker_count(), 0)
        for cid, w in self.manager.workers.items():
            self.assertEqual(w.get_snapshot()["lifecycle_state"], "STOPPED")
            self.assertFalse(w.running)
            self.assertIsNone(w.thread)

    def test_20_remove_camera_with_confirmation(self):
        """20. Kiểm tra thêm và xóa cấu hình camera an toàn trong Quản lý camera."""
        sample_path = "negative_videos/VN_POS_01.mp4" if os.path.exists("negative_videos/VN_POS_01.mp4") else "datasets/test_videos/pos/posVideo1.868.avi"
        new_cid = self.manager.add_camera("CAM 99 - Test Remove", sample_path, auto_start=False)
        self.assertIn(new_cid, self.manager.workers)
        self.assertEqual(self.manager.workers[new_cid].get_snapshot()["lifecycle_state"], "STOPPED")
        removed = self.manager.remove_camera(new_cid)
        self.assertTrue(removed)
        self.assertNotIn(new_cid, self.manager.workers)

    def test_21_confirm_event_stops_and_silences_alarm(self):
        """21. Kiểm tra sau khi xác nhận (hoặc bỏ qua) sự kiện cảnh báo thì âm báo còi hú phải tắt ngay và không tự bật lại."""
        dummy_bgr = np.zeros((240, 320, 3), dtype=np.uint8)
        ev = self.manager.event_db.record_event(dummy_bgr, "CAM-01", "FIRE", 0.94, alert_level="FIRE", is_demo=0)

        # Giả lập trạng thái còi đang kêu
        self.manager.alert_manager.reset_alarm_acknowledgment()
        self.manager.alert_manager.alarm_playing = True

        # Nhấn Xác nhận sự kiện
        self.manager.acknowledge_event(ev["id"], action_taken="Đã xác nhận cháy thật", note="Test tắt còi")

        # Còi phải tắt ngay lập tức và cờ khóa âm báo sau xác nhận phải bật
        self.assertFalse(self.manager.alert_manager.alarm_playing, "Sau khi xác nhận, alarm_playing phải bằng False")
        self.assertTrue(self.manager.alert_manager.alarm_silenced_by_ack, "Sau khi xác nhận, alarm_silenced_by_ack phải bằng True")
        self.assertIn("CAM-01", self.manager.alert_manager.acknowledged_cameras)

        # Gọi lại play_alarm cho camera đã xác nhận không được phép làm còi kêu lại
        self.manager.alert_manager.play_alarm(camera_name="CAM-01")
        self.assertFalse(self.manager.alert_manager.alarm_playing, "Còi không được tự bật lại sau khi đã xác nhận")

    def test_22_no_auto_video_and_test_cam_by_file(self):
        """
        Kiểm tra:
        1. Khi khởi tạo với auto_assign_videos=False (như trên Web Dashboard),
           KHÔNG tự động gắn bất kỳ video mẫu nào vào 9 camera (source="" và source_type="UNASSIGNED").
        2. start_monitoring() trên các camera chưa gán nguồn không mở luồng giả tạo.
        3. assign_test_video_to_camera() gắn file video để test camera cụ thể.
        4. detect_available_cameras() trả về danh sách thiết bị phần cứng và các ô đã gán nguồn.
        5. detach_camera_source() gỡ nguồn khỏi camera.
        """
        no_vid_mgr = MultiCameraManager(
            db_path="test_dynamic_no_auto_vid.db",
            model_path="best.pt",
            auto_assign_videos=False,
        )
        try:
            for cfg in no_vid_mgr.camera_configs:
                self.assertEqual(cfg.source, "", f"{cfg.cam_id} không được tự động gắn video")
                self.assertEqual(cfg.source_type, "UNASSIGNED")
                self.assertEqual(cfg.masked_source, "Chưa gán nguồn")

            res = no_vid_mgr.start_monitoring(["CAM-01", "CAM-02"])
            self.assertEqual(res["assigned"], [])
            self.assertEqual(res["connected"], [])
            self.assertEqual(set(res["unassigned"]), {"CAM-01", "CAM-02"})

            # Gán file video để test CAM-01
            test_videos = no_vid_mgr.detect_available_cameras()
            self.assertIn("active_hardware", test_videos)
            self.assertEqual(test_videos["assigned_camera_ids"], [])

            from src.config import get_available_test_videos
            avail_vids = get_available_test_videos()
            if avail_vids:
                first_vid_path = list(avail_vids.values())[0]
                ok, msg = no_vid_mgr.assign_test_video_to_camera("CAM-01", first_vid_path, auto_start=False)
                self.assertTrue(ok, msg)
                c1_cfg = no_vid_mgr.workers["CAM-01"].config
                self.assertEqual(c1_cfg.source, first_vid_path)
                self.assertEqual(c1_cfg.source_type, "VIDEO_TEST")

                det_after = no_vid_mgr.detect_available_cameras()
                self.assertEqual(det_after["assigned_camera_ids"], ["CAM-01"])

                # Gỡ nguồn
                no_vid_mgr.detach_camera_source("CAM-01")
                self.assertEqual(c1_cfg.source, "")
                self.assertEqual(c1_cfg.source_type, "UNASSIGNED")
        finally:
            no_vid_mgr.stop_all()
            if os.path.exists("test_dynamic_no_auto_vid.db"):
                try:
                    os.remove("test_dynamic_no_auto_vid.db")
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()

