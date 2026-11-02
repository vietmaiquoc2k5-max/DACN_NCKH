import os
import sys
import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# Đảm bảo in UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import torch

CSV_PATH = Path("data/video_dataset_list.csv")
MODEL_PATH = "best.pt"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

def auto_detect_timestamps(prefix=None, only_unassigned=True, force=False):
    if not CSV_PATH.exists():
        print(f"❌ Không tìm thấy {CSV_PATH}")
        return

    print("\n" + "="*70)
    print("🤖 TỰ ĐỘNG DÒ TÌM MỐC THỜI GIAN CHÁY/KHÓI BẰNG AI (CUDA GPU)")
    print("="*70)
    print(f"• Tải mô hình: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    df = pd.read_csv(CSV_PATH)
    pos_mask = df["category"].str.contains("pos")
    if prefix:
        pos_mask &= df["video_id"].astype(str).str.startswith(prefix)
    if only_unassigned and not force:
        pos_mask &= (
            df["notes"].astype(str).str.contains("CHUA_GAN_MOC")
            | ((df["fire_start_sec"] < 0) & (df["smoke_start_sec"] < 0))
        )

    pos_df = df[pos_mask]
    if len(pos_df) == 0:
        print("ℹ️ Không có video nào cần dò mốc (đã gán hết hoặc không khớp điều kiện lọc).")
        return

    print(f"• Đang quét {len(pos_df)} video dương tính...")

    for idx, row in pos_df.iterrows():
        v_path = Path(row["relative_path"])
        if not v_path.exists():
            continue

        cap = cv2.VideoCapture(str(v_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        fire_detected_sec = -1
        smoke_detected_sec = -1

        # Bộ đếm khung liên tiếp để tránh nhận nhầm đơn khung
        consec_fire = 0
        consec_smoke = 0
        frame_idx = 0

        # Bước nhảy ~5 khung hình/giây để vừa chính xác tới 0.2s vừa quét nhanh video dài
        stride = max(1, int(round(fps / 5.0)))
        first_any_hit_sec = -1.0

        while cap.isOpened():
            ret = cap.grab()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % stride != 0:
                continue
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                continue

            sec = round(frame_idx / fps, 1)

            # Dự đoán với YOLO trên GPU (hoặc CPU nếu không có CUDA)
            results = model.predict(frame, conf=0.3, imgsz=416, verbose=False, device=DEVICE)[0]
            boxes = results.boxes

            has_fire_in_frame = False
            has_smoke_in_frame = False

            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    cls_id = int(b.cls[0])
                    if cls_id == 1: # Fire
                        has_fire_in_frame = True
                    elif cls_id == 0: # Smoke
                        has_smoke_in_frame = True

            # Xác nhận khói
            if has_smoke_in_frame:
                consec_smoke += 1
                if consec_smoke >= 2 and smoke_detected_sec < 0:
                    smoke_detected_sec = sec
                    if first_any_hit_sec < 0:
                        first_any_hit_sec = sec
            else:
                consec_smoke = max(0, consec_smoke - 1)

            # Xác nhận lửa
            if has_fire_in_frame:
                consec_fire += 1
                if consec_fire >= 2 and fire_detected_sec < 0:
                    fire_detected_sec = sec
                    if first_any_hit_sec < 0:
                        first_any_hit_sec = sec
            else:
                consec_fire = max(0, consec_fire - 1)

            # Nếu cả 2 đều đã phát hiện thì dừng sớm
            if fire_detected_sec >= 0 and smoke_detected_sec >= 0:
                break
            # Nếu đã phát hiện 1 loại và đã quét thêm 45s sau đó, hoặc đã quét hết 120s đầu của video dài thì dừng
            if first_any_hit_sec >= 0 and (sec - first_any_hit_sec) >= 45.0:
                break
            if sec >= 120.0:
                break

        cap.release()

        # Nếu video là fire_pos mà chưa thấy smoke thì giữ smoke = -1
        if "fire" in row["category"] and fire_detected_sec < 0:
            fire_detected_sec = 0.0 # Nếu AI không bắt được mốc sau thì mặc định bắt đầu từ 0
        if "smoke" in row["category"] and smoke_detected_sec < 0:
            smoke_detected_sec = 0.0

        df.loc[idx, "fire_start_sec"] = fire_detected_sec
        df.loc[idx, "smoke_start_sec"] = smoke_detected_sec

        # Nếu đã tìm thấy ít nhất một mốc, bỏ ghi chú CHUA_GAN_MOC
        if fire_detected_sec >= 0 or smoke_detected_sec >= 0:
            cur_note = str(df.loc[idx, "notes"]) if pd.notna(df.loc[idx, "notes"]) else ""
            if "CHUA_GAN_MOC" in cur_note:
                new_note = cur_note.replace("CHUA_GAN_MOC", "AI_GOI_Y_MOC").strip("; ")
                df.loc[idx, "notes"] = new_note

        print(f"  [{row['video_id']}] {row['file_name']:<22} | Smoke Start: {smoke_detected_sec:>5}s | Fire Start: {fire_detected_sec:>5}s")

    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ ĐÃ HOÀN TẤT VÀ CẬP NHẬT MỐC THỜI GIAN VÀO: {CSV_PATH.resolve()}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tự động dò mốc thời gian cháy/khói bằng YOLO")
    parser.add_argument("--prefix", default=None, help="Chỉ quét video_id bắt đầu bằng tiền tố (vd: VN_)")
    parser.add_argument("--all", action="store_true", help="Quét lại toàn bộ kể cả video đã có mốc")
    args = parser.parse_args()
    auto_detect_timestamps(prefix=args.prefix, only_unassigned=not args.all)
