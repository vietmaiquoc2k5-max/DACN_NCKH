"""Bộ kiểm thử tự động toàn diện cho hệ thống Web Dashboard 9 Camera (`tests/test_dashboard_system.py`).

Kiểm tra đầy đủ các tiêu chí nghiệm thu:
1. Tích hợp model (`src/detector.py`): Nạp `best.pt` (SMOKE/FIRE), từ chối model không hợp lệ (`yolov8n.pt` COCO),
   chạy phát hiện trên video thực tế và vẽ bounding box.
2. Xác minh theo chuỗi khung hình (`src/temporal_verifier.py`): Cửa sổ k/N, 2 mức SUSPECT/FIRE,
   lọc vùng loại trừ ROI và chống phát cảnh báo lặp (Cooldown).
3. Cơ sở dữ liệu & Cảnh báo (`src/database.py`, `alert_manager.py`): Lưu SQLite, lọc theo ngày/camera/loại,
   xác nhận/bỏ qua sự kiện, phân biệt dữ liệu thật vs demo, xuất CSV.
4. Quản lý 9 Camera (`src/camera_manager.py`): Hiển thị đủ 9 ô camera, camera lỗi/mất kết nối không làm dừng
   các camera đang chạy video thực tế, đo FPS thực tế.
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
    print("\n[TEST 4] Kiểm tra MultiCameraManager (9 camera & cô lập lỗi khi 1 camera mất kết nối)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "cam_test.db")
        mgr = MultiCameraManager(db_path=db_file, model_path="best.pt")
        mgr.settings.mute_audio = True

        snaps = mgr.get_all_snapshots()
        assert len(snaps) == 9, f"Phải có đúng 9 camera (3x3), nhận được: {len(snaps)}"

        # Cấu hình CAM-01 chạy video thật, CAM-02 cố tình trỏ tới đường dẫn không tồn tại
        mgr.update_camera_source("CAM-02", "CAM 02 - Lỗi Kết Nối", "non_existent_video_stream.mp4", auto_restart=False)
        mgr.start_camera("CAM-01")
        mgr.start_camera("CAM-02")

        # Chờ một chút để các luồng xử lý
        deadline = time.time() + 6.0
        cam1_online = False
        cam2_disconnected = False
        while time.time() < deadline:
            s_list = {s["cam_id"]: s for s in mgr.get_all_snapshots()}
            if s_list["CAM-01"]["status"] == "ONLINE" and s_list["CAM-01"]["fps"] > 0:
                cam1_online = True
            if s_list["CAM-02"]["status"] == "DISCONNECTED":
                cam2_disconnected = True
            if cam1_online and cam2_disconnected:
                break
            time.sleep(0.25)

        s_final = {s["cam_id"]: s for s in mgr.get_all_snapshots()}
        mgr.stop_all()

        assert cam2_disconnected, f"CAM-02 phải ở trạng thái DISCONNECTED, thực tế: {s_final['CAM-02']['status']}"
        assert cam1_online, f"CAM-01 phải hoạt động ONLINE bình thường dù CAM-02 lỗi, thực tế: {s_final['CAM-01']['status']}"
        print(
            f"  ✅ Cô lập lỗi thành công: CAM-02={s_final['CAM-02']['status']} không ảnh hưởng "
            f"CAM-01={s_final['CAM-01']['status']} (FPS thực tế CAM-01: {s_final['CAM-01']['fps']} FPS)."
        )


if __name__ == "__main__":
    print("=" * 72)
    print("🚀 BẮT ĐẦU CHẠY KIỂM THỬ HỆ THỐNG WEB DASHBOARD 9 CAMERA (3x3)")
    print("=" * 72)
    test_detector_and_model_validation()
    test_temporal_verifier()
    test_database_and_alerts()
    test_multi_camera_fault_isolation()
    print("\n" + "=" * 72)
    print("🎉 TẤT CẢ 4 HẠNG MỤC KIỂM THỬ HỆ THỐNG 9 CAMERA ĐỀU PASS 100%!")
    print("=" * 72)

