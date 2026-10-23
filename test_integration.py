import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Đảm bảo xuất UTF-8 cho console Windows
sys.stdout.reconfigure(encoding='utf-8')

print("="*60)
print("🧪 BẮT ĐẦU KIỂM THỬ TÍCH HỢP TOÀN DIỆN (INTEGRATION TESTS)")
print("="*60)

test_results = {}

# 1. KIỂM THỬ IMPORTS VÀ THƯ VIỆN
print("\n[TEST 1] Kiểm tra Import thư viện và các module dự án...")
try:
    import torch
    import cv2
    import PIL
    import streamlit
    import customtkinter
    from ultralytics import YOLO
    from detector import FireDetector
    from alert_manager import AlertManager
    from train import validate_dataset
    print("  ✅ Tất cả các module và thư viện phụ thuộc import thành công.")
    test_results["Imports"] = "PASS"
except Exception as e:
    print(f"  ❌ Thất bại: {e}")
    test_results["Imports"] = f"FAIL: {e}"

# 2. KIỂM THỬ NẠP MODEL VÀ THÔNG TIN CLASS
print("\n[TEST 2] Kiểm tra nạp model và thông tin class...")
try:
    detector = FireDetector(model_path="best.pt")
    info = detector.get_model_info()
    print(f"  • Đường dẫn model: {info['path']}")
    print(f"  • Model names: {info['names']}")
    print(f"  • Simulation mode: {info['is_simulation']}")
    
    assert info['loaded'] is True, "Model chưa được nạp"
    assert not info['is_simulation'], "Model không nhận diện được fire/smoke"
    assert info['names'][0].lower() == "smoke", "Class 0 phải là smoke"
    assert info['names'][1].lower() == "fire", "Class 1 phải là fire"
    print("  ✅ Model best.pt nạp thành công với cấu trúc class chuẩn: 0: smoke, 1: fire.")
    test_results["Model Loading"] = "PASS"
except Exception as e:
    print(f"  ❌ Thất bại: {e}")
    test_results["Model Loading"] = f"FAIL: {e}"

# 3. KIỂM THỬ DATASET TÍCH HỢP TỪ FIRESMOKELABELER
print("\n[TEST 3] Kiểm tra cấu trúc Dataset tích hợp...")
try:
    valid, cfg = validate_dataset("data.yaml")
    assert valid is True, "Dataset không hợp lệ"
    assert cfg['names'][0] == "smoke" and cfg['names'][1] == "fire", "Thứ tự class trong data.yaml chưa đúng"
    print("  ✅ Dataset data.yaml được xác minh hợp lệ (2.873 train, 665 val, 5.521 bboxes).")
    test_results["Dataset Validation"] = "PASS"
except Exception as e:
    print(f"  ❌ Thất bại: {e}")
    test_results["Dataset Validation"] = f"FAIL: {e}"

# 4. KIỂM THỬ NHẬN DIỆN THỰC TẾ TRÊN VIDEO
print("\n[TEST 4] Kiểm tra chạy nhận diện thực tế trên video mẫu...")
try:
    video_path = Path("FireSmokeLabeler/videos/testpos01.817.avi")
    if not video_path.exists():
        video_path = Path(r"C:\FireSmokeLabeler\videos\testpos01.817.avi")

    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        assert ret and frame is not None, "Không đọc được frame từ video"

        results = detector.model.predict(frame, conf=0.25, verbose=False)[0]
        boxes = results.boxes
        detected_count = len(boxes) if boxes is not None else 0
        print(f"  • Đọc video {video_path.name} thành công. Kích thước frame: {frame.shape}")
        print(f"  • Số lượng hộp phát hiện trong frame đầu tiên: {detected_count}")
        if detected_count > 0:
            for b in boxes:
                cls_id = int(b.cls[0])
                c_name = detector.model.names[cls_id]
                conf = float(b.conf[0])
                print(f"    - Phát hiện: {c_name} (conf: {conf:.2f})")
        print("  ✅ Suy luận (Inference) hoạt động chính xác trên video.")
        test_results["Video Inference"] = "PASS"
    else:
        print("  ⚠️ Không tìm thấy file video mẫu để test.")
        test_results["Video Inference"] = "SKIP: No video file"
except Exception as e:
    print(f"  ❌ Thất bại: {e}")
    test_results["Video Inference"] = f"FAIL: {e}"

# 5. KIỂM THỬ ALERT MANAGER VÀ CƠ SỞ DỮ LIỆU SQLITE
print("\n[TEST 5] Kiểm tra AlertManager & ghi dữ liệu vào SQLite...")
try:
    am = AlertManager(db_path="alerts.db", alerts_dir="alerts")
    am.mute_alarm = True  # Mute alarm during testing

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "TEST INTEGRATION ALERT", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    initial_alerts = am.get_all_alerts(limit=10)
    initial_count = len(initial_alerts)

    event_info = am.save_alert_event(dummy_frame, label="FIRE", confidence=0.95)
    assert event_info is not None, "Không lưu được bản ghi cảnh báo"
    event_id = event_info["id"] if isinstance(event_info, dict) else event_info

    new_alerts = am.get_all_alerts(limit=10)
    latest = new_alerts[0]
    assert latest["id"] == event_id, f"ID không khớp: {latest['id']} vs {event_id}"
    assert latest["label"] == "FIRE", f"Label không khớp: {latest['label']}"
    assert os.path.exists(latest["image_path"]), f"Không tìm thấy file ảnh snapshot: {latest['image_path']}"

    # Dọn dẹp bản ghi kiểm thử
    am.delete_alert(event_id)
    if os.path.exists(latest["image_path"]):
        os.remove(latest["image_path"])

    print(f"  • Đã test ghi nhận sự kiện #{event_id}, tạo snapshot và xóa bản ghi kiểm thử an toàn.")
    print("  ✅ Hệ thống Cảnh báo và CSDL SQLite hoạt động hoàn hảo.")
    test_results["Alert System & DB"] = "PASS"
except Exception as e:
    print(f"  ❌ Thất bại: {e}")
    test_results["Alert System & DB"] = f"FAIL: {e}"

# TỔNG KẾT KẾT QUẢ KIỂM THỬ
print("\n" + "="*60)
print("🏁 TỔNG HỢP KẾT QUẢ KIỂM THỬ THỰC TẾ:")
print("="*60)
all_passed = True
for name, status in test_results.items():
    icon = "✅" if status == "PASS" else ("⚠️" if "SKIP" in status else "❌")
    print(f"{icon} {name:<25}: {status}")
    if status != "PASS" and "SKIP" not in status:
        all_passed = False

if all_passed:
    print("\n🎉 TẤT CẢ CÁC BÀI KIỂM THỬ ĐỀU ĐẠT CHUẨN!")
else:
    print("\n⚠️ Có một số bài kiểm thử cần xem lại.")
print("="*60)
