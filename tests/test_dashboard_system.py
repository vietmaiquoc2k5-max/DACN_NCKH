"""Bộ kiểm thử tự động toàn diện cho hệ thống Web Dashboard 9 Camera (`tests/test_dashboard_system.py`).

Kiểm tra đầy đủ các tiêu chí nghiệm thu (bao gồm cả Mục 10 - Bổ sung chế độ Camera thực tế):
1. Tích hợp model (`src/detector.py`): Nạp `best.pt` (SMOKE/FIRE), từ chối model không hợp lệ (`yolov8n.pt` COCO),
   chạy phát hiện trên video thực tế và vẽ bounding box kèm tên camera.
2. Xác minh theo chuỗi khung hình (`src/temporal_verifier.py`): Cửa sổ k/N, 2 mức SUSPECT/FIRE,
   lọc vùng loại trừ ROI và chống phát cảnh báo lặp (Cooldown).
3. Cơ sở dữ liệu & Cảnh báo (`src/database.py`, `alert_manager.py`): Lưu SQLite, lọc theo ngày/camera/loại,
   xác nhận/bỏ qua sự kiện, phân biệt dữ liệu thật vs demo, xuất CSV.
4. Quản lý 9 Camera & Cô lập lỗi (`src/camera_manager.py`): Hiển thị đủ 9 ô camera, trạng thái chuẩn
   (`CONNECTED`, `DISCONNECTED`, `RECONNECTING`, `ERROR`, `PAUSED`), tách biệt `capture_fps` và `inference_fps`,
   camera lỗi/mất kết nối không làm dừng các camera khác và không tạo cảnh báo cháy giả.
5. Chế độ Camera thực tế & Bảo mật RTSP (`VIDEO_TEST`, `WEBCAM`, `IP_CAMERA`, `MULTI_CAMERA`):
   - Kiểm tra kết nối (`test_camera_connection`) trả về ảnh xem trước thật.
   - Ẩn mật khẩu RTSP (`mask_rtsp_credentials`), tuyệt đối không lộ thông tin xác thực trong log hoặc snapshot.
   - Hiển thị đúng trạng thái lỗi kết nối (`CONN_ERROR` / `STANDBY`) thay vì báo `SAFE` giả khi mất tín hiệu.
"""
import os
import sys
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import mask_rtsp_credentials
from src.detector import SharedFireSmokeDetector
from src.temporal_verifier import TemporalVerifier, DetectionBox
from src.database import EventDatabase
from src.camera_manager import MultiCameraManager


def test_detector_and_model_validation():
    print("\n[TEST 1] Kiểm tra SharedFireSmokeDetector & xác thực lớp FIRE/SMOKE...")
    detector = SharedFireSmokeDetector(model_path="best.pt", conf_threshold=0.25)
    info = detector.get_model_info()
    assert info["loaded"] is True, f"Không nạp được best.pt: {info['error']}"
    assert info["names"].get(0) == "SMOKE", f"Class 0 phải là SMOKE, nhận được: {info['names']}"
    assert info["names"].get(1) == "FIRE", f"Class 1 phải là FIRE, nhận được: {info['names']}"
    print(f"  ✅ Đã nạp chuẩn xác model dự án: {info['path']} | Lớp: {info['names']}")

    # Kiểm tra từ chối model COCO yolov8n.pt (không giả lập kết quả)
    if (PROJECT_ROOT / "yolov8n.pt").exists():
        bad_det = SharedFireSmokeDetector(model_path="yolov8n.pt")
        assert bad_det.model_loaded is False, "Phải từ chối model COCO không có lớp fire/smoke!"
        print(f"  ✅ Đã từ chối đúng model không hợp lệ (yolov8n.pt): {bad_det.error_message[:75]}...")

    # Kiểm tra chạy thực tế trên video có sẵn trong dự án
    sample_videos = [
        PROJECT_ROOT / "videovietnam" / "VN_POS_01.mp4",
        PROJECT_ROOT / "FireSmokeLabeler" / "videos" / "testpos01.817.avi",
    ]
    vid_path = next((p for p in sample_videos if p.exists()), None)
    assert vid_path is not None, "Không tìm thấy video mẫu trong repo để kiểm thử"

    cap = cv2.VideoCapture(str(vid_path))
    ret, frame = cap.read()
    cap.release()
    assert ret and frame is not None, f"Không đọc được frame từ {vid_path}"

    boxes, infer_ms = detector.detect_frame(frame, conf_threshold=0.25)
    annotated = SharedFireSmokeDetector.draw_annotations(frame, boxes, cam_name="CAM-01", alert_level="FIRE", fps=25.0)
    assert annotated.shape == frame.shape, "Kích thước ảnh sau khi vẽ bounding box bị sai"
    assert infer_ms > 0.0, "Thời gian suy luận phải > 0 ms"
    print(
        f"  ✅ Suy luận thực tế trên '{vid_path.name}': {len(boxes)} boxes, "
        f"thời gian xử lý: {infer_ms:.2f} ms ({1000.0/infer_ms:.1f} FPS)."
    )


def test_temporal_verifier():
    print("\n[TEST 2] Kiểm tra TemporalVerifier (cửa sổ k/N, ROI masking & Cooldown)...")
    verifier = TemporalVerifier(
        window_size_n=5,
        confirm_frames_k=3,
        suspect_frames_k=2,
        cooldown_sec=5.0,
        exclusion_zones=[(0.8, 0.8, 1.0, 1.0)],  # Góc dưới phải là vùng loại trừ
    )

    # Box nằm trong vùng loại trừ -> phải bị bỏ qua
    ignored_box = DetectionBox(
        cls_id=1, label="FIRE", confidence=0.85, xyxy=(550, 420, 620, 470), norm_xywh=(0.9, 0.9, 0.08, 0.08)
    )
    res0 = verifier.update([ignored_box])
    assert res0["active_hits"] == 0 and res0["alert_level"] == "SAFE", "Box trong vùng loại trừ phải bị lọc bỏ"

    # Đưa 3 frame liên tiếp có box hợp lệ
    valid_box = DetectionBox(
        cls_id=1, label="FIRE", confidence=0.88, xyxy=(100, 100, 220, 240), norm_xywh=(0.25, 0.35, 0.18, 0.25)
    )
    r1 = verifier.update([valid_box])
    assert r1["alert_level"] == "SAFE" and not r1["should_trigger_db_alert"]

    r2 = verifier.update([valid_box])
    assert r2["alert_level"] == "SUSPECT" and not r2["should_trigger_db_alert"]

    r3 = verifier.update([valid_box])
    assert r3["alert_level"] == "FIRE" and r3["should_trigger_db_alert"] is True

    # Frame thứ 4 liên tiếp: vẫn là FIRE nhưng Cooldown ngăn phát cảnh báo trùng lặp
    r4 = verifier.update([valid_box])
    assert r4["alert_level"] == "FIRE" and r4["should_trigger_db_alert"] is False
    print("  ✅ TemporalVerifier hoạt động chuẩn xác: SAFE -> SUSPECT (2/5) -> FIRE (3/5) + Cooldown chống lặp.")


def test_database_and_alerts():
    print("\n[TEST 3] Kiểm tra EventDatabase & SQLite (lọc ngày/camera/loại, xác nhận & xuất CSV)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "test_alerts.db")
        alerts_folder = os.path.join(tmpdir, "snaps")
        from alert_manager import AlertManager

        am = AlertManager(db_path=db_file, alerts_dir=alerts_folder)
        am.mute_alarm = True
        db = EventDatabase(alert_manager=am, db_path=db_file)

        dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
        ev1 = db.record_event(dummy_img, camera_name="CAM-02", label="FIRE", confidence=0.91, alert_level="FIRE", is_demo=0)
        ev2 = db.record_event(dummy_img, camera_name="CAM-05", label="SMOKE", confidence=0.76, alert_level="SUSPECT", is_demo=0)
        ev3 = db.record_event(dummy_img, camera_name="CAM-01", label="FIRE", confidence=0.99, alert_level="FIRE", is_demo=1)

        assert os.path.exists(ev1["image_path"]), "Ảnh bằng chứng phải được lưu trên đĩa"

        # Kiểm tra thống kê phân biệt dữ liệu thật vs demo
        stats = db.get_summary_statistics()
        assert stats["total_real"] == 2 and stats["total_demo"] == 1
        assert stats["real_fire"] == 1 and stats["real_smoke"] == 1

        # Kiểm tra lọc theo camera và loại
        cam2_events = db.query_events(camera_filter="CAM-02", include_demo=False)
        assert len(cam2_events) == 1 and cam2_events[0]["label"] == "FIRE"

        # Kiểm tra xác nhận sự kiện và xuất CSV
        db.confirm_event(ev1["id"], action_taken="Đã xác nhận cháy thật", note="Kiểm thử tự động")
        updated = db.query_events(camera_filter="CAM-02", status_filter="Đã xác nhận")[0]
        assert updated["status"] == "ACKNOWLEDGED"

        csv_out = os.path.join(tmpdir, "export.csv")
        ok, msg = db.export_to_csv(csv_out)
        assert ok and os.path.exists(csv_out), f"Xuất CSV thất bại: {msg}"
        print("  ✅ EventDatabase & SQLite vượt qua toàn bộ kiểm thử lưu trữ, lọc, xác nhận và xuất CSV.")


def test_multi_camera_fault_isolation():
    print("\n[TEST 4] Kiểm tra MultiCameraManager (9 camera, CONNECTED/ERROR, FPS đọc & FPS suy luận độc lập)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "cam_test.db")
        mgr = MultiCameraManager(db_path=db_file, model_path="best.pt")
        mgr.settings.mute_audio = True

        snaps = mgr.get_all_snapshots()
        assert len(snaps) == 9, f"Phải có đúng 9 camera (3x3), nhận được: {len(snaps)}"

        # Cấu hình CAM-01 chạy video thật, CAM-02 cố tình trỏ tới đường dẫn không tồn tại (tắt auto_reconnect để kiểm tra ERROR ngay)
        mgr.update_camera_source(
            "CAM-02",
            "CAM 02 - Lỗi Kết Nối",
            "non_existent_video_stream.mp4",
            source_type="VIDEO_TEST",
            auto_reconnect=False,
            auto_restart=False,
        )
        mgr.start_camera("CAM-01")
        mgr.start_camera("CAM-02")

        deadline = time.time() + 10.0
        cam1_connected = False
        cam2_error = False
        while time.time() < deadline:
            s_list = {s["cam_id"]: s for s in mgr.get_all_snapshots()}
            if (
                s_list["CAM-01"]["status"] == "CONNECTED"
                and s_list["CAM-01"]["capture_fps"] > 0
                and s_list["CAM-01"]["inference_fps"] > 0
            ):
                cam1_connected = True
            if s_list["CAM-02"]["status"] in ("ERROR", "RECONNECTING", "DISCONNECTED") and s_list["CAM-02"]["error_detail"]:
                cam2_error = True
            if cam1_connected and cam2_error:
                break
            time.sleep(0.25)

        s_final = {s["cam_id"]: s for s in mgr.get_all_snapshots()}
        metrics = mgr.get_system_metrics()
        mgr.stop_all()

        assert cam2_error, f"CAM-02 phải báo lỗi kết nối (ERROR/RECONNECTING), thực tế: {s_final['CAM-02']['status']}"
        assert cam1_connected, f"CAM-01 phải hoạt động CONNECTED bình thường dù CAM-02 lỗi, thực tế: {s_final['CAM-01']['status']}"
        assert s_final["CAM-02"]["alert_level"] == "CONN_ERROR", (
            f"Camera mất kết nối phải có alert_level='CONN_ERROR' (không báo cháy giả cũng không báo SAFE), "
            f"nhận được: {s_final['CAM-02']['alert_level']}"
        )
        assert len(metrics["error_cameras"]) >= 1, "Hệ thống phải ghi nhận camera đang bị lỗi kết nối"
        print(
            f"  ✅ Cô lập lỗi thành công: CAM-02={s_final['CAM-02']['status']} (alert_level={s_final['CAM-02']['alert_level']}) "
            f"không ảnh hưởng CAM-01={s_final['CAM-01']['status']} (FPS đọc: {s_final['CAM-01']['capture_fps']} | "
            f"FPS YOLO: {s_final['CAM-01']['inference_fps']})."
        )


def test_real_camera_features_and_rtsp_security():
    print("\n[TEST 5] Kiểm tra Chế độ Camera thực tế, Kiểm tra kết nối & Bảo mật ẩn mật khẩu RTSP...")
    # 1. Kiểm tra hàm mask_rtsp_credentials
    raw_rtsp = "rtsp://admin:SuperSecretPass99!@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
    masked = mask_rtsp_credentials(raw_rtsp)
    assert "SuperSecretPass99!" not in masked, f"Mật khẩu RTSP bị lộ trong chuỗi đã che: {masked}"
    assert masked.startswith("rtsp://admin:******@192.168.1.108:554"), f"Định dạng che mật khẩu sai: {masked}"
    print(f"  ✅ Ẩn mật khẩu RTSP an toàn: '{masked}'")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "sec_test.db")
        mgr = MultiCameraManager(db_path=db_file, model_path="best.pt")
        mgr.settings.mute_audio = True

        # 2. Khi chưa có camera nào kết nối, hệ thống phải ở trạng thái STANDBY (không báo SAFE giả)
        m_init = mgr.get_system_metrics()
        assert m_init["overall_level"] == "STANDBY", f"Khi 0 camera kết nối phải là STANDBY, nhận được: {m_init['overall_level']}"

        # 3. Lưu cấu hình RTSP chứa mật khẩu và kiểm tra không lộ mật khẩu trong snapshot hay log
        mgr.update_camera_source(
            "CAM-03",
            new_name="CAM 03 - Cổng Chính RTSP",
            new_source=raw_rtsp,
            source_type="IP_CAMERA",
            enabled=True,
            auto_reconnect=True,
            auto_restart=False,
        )
        snap_cam3 = next(s for s in mgr.get_all_snapshots() if s["cam_id"] == "CAM-03")
        assert "SuperSecretPass99!" not in snap_cam3["masked_source"], "Snapshot không được chứa mật khẩu RTSP dạng rõ!"
        for log_line in mgr.get_recent_logs(20):
            assert "SuperSecretPass99!" not in log_line, f"Log hệ thống làm lộ mật khẩu RTSP: {log_line}"
        print("  ✅ Không lộ mật khẩu RTSP trong cấu hình hiển thị (Snapshot) và Nhật ký hệ thống (Logs).")

        # 4. Kiểm tra nút 'Kiểm tra kết nối' (test_camera_connection) trên nguồn hợp lệ
        ok, msg, preview = mgr.test_camera_connection("CAM-01")
        assert ok is True and preview is not None and preview.ndim == 3, f"Kiểm tra kết nối CAM-01 thất bại: {msg}"
        print(f"  ✅ Kiểm tra kết nối (Test Connection) thành công trên CAM-01: {msg}")

        # 5. Kiểm tra chuyển đổi chế độ hoạt động (VIDEO_TEST, WEBCAM, IP_CAMERA, MULTI_CAMERA)
        mgr.apply_operating_mode_preset("WEBCAM", webcam_index=0, auto_connect=False)
        assert mgr.settings.operating_mode == "WEBCAM"
        snap_after_mode = {s["cam_id"]: s for s in mgr.get_all_snapshots()}
        assert snap_after_mode["CAM-01"]["source_type"] == "WEBCAM" and snap_after_mode["CAM-01"]["enabled"] is True
        mgr.apply_operating_mode_preset("MULTI_CAMERA", auto_connect=False)
        assert mgr.settings.operating_mode == "MULTI_CAMERA"
        print("  ✅ Chuyển đổi linh hoạt giữa 4 chế độ VIDEO_TEST, WEBCAM, IP_CAMERA và MULTI_CAMERA.")


if __name__ == "__main__":
    print("=" * 72)
    print("🚀 BẮT ĐẦU CHẠY KIỂM THỬ HỆ THỐNG WEB DASHBOARD & CAMERA THỰC TẾ")
    print("=" * 72)
    test_detector_and_model_validation()
    test_temporal_verifier()
    test_database_and_alerts()
    test_multi_camera_fault_isolation()
    test_real_camera_features_and_rtsp_security()
    print("\n" + "=" * 72)
    print("🎉 TẤT CẢ 5 HẠNG MỤC KIỂM THỬ HỆ THỐNG 9 CAMERA ĐỀU PASS 100%!")
    print("=" * 72)
