"""Trích xuất và làm mờ khuôn mặt/biển số từ bộ 23 video Việt Nam (videovietnam/) phục vụ MT1 & MT4.

Yêu cầu theo Đề cương NCKH:
- Đạt >= 1.000 khung hình Việt Nam (hiện nâng lên 1.450 khung hình từ 23 video thực tế).
- Chia 60/20/20 NGHIÊM NGẶT THEO VIDEO cho cả video dương tính (20 video) và âm tính (3 video):
  tuyệt đối không để khung hình của cùng một video xuất hiện ở hai tập.
- Làm mờ khuôn mặt và biển số xe trước khi gán nhãn / công bố dữ liệu (dùng YOLOv8n).
- Bảo toàn 100% các khung hình và nhãn đã kiểm duyệt trước đó (incremental update).
- Lưu cấu trúc chuẩn YOLO: datasets/vn_dataset/{images,labels}/{train,val,test}/
- Xuất manifest theo dõi: datasets/vn_dataset/split_manifest.csv
"""
import os
import sys
import argparse
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

VN_DIR = Path("videovietnam")
OUT_DIR = Path("datasets/vn_dataset")
MANIFEST_PATH = OUT_DIR / "split_manifest.csv"
VIDEO_CSV_PATH = Path("data/video_dataset_list.csv")
DATA_YAML_PATH = Path("data_vn.yaml")
YOLO_ANONYMIZER_PATH = "yolov8n.pt"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Phân chia 60/20/20 NGHIÊM NGẶT THEO VIDEO cho toàn bộ 20 video POS và 3 video NEG:
# - Train (60%): 12 video POS (01..06, 11..16) + 1 video NEG (VN_NEG_01) -> 330 + 180 + 360 = 870 ảnh (60.0%)
# - Val   (20%):  4 video POS (07..08, 17..18) + 1 video NEG (VN_NEG_03) -> 110 +  60 + 120 = 290 ảnh (20.0%)
# - Test  (20%):  4 video POS (09..10, 19..20) + 1 video NEG (VN_NEG_04) -> 110 +  60 + 120 = 290 ảnh (20.0%)
POS_SPLIT_CONFIG = {
    "train": [
        "VN_POS_01", "VN_POS_02", "VN_POS_03", "VN_POS_04", "VN_POS_05", "VN_POS_06",
        "VN_POS_11", "VN_POS_12", "VN_POS_13", "VN_POS_14", "VN_POS_15", "VN_POS_16",
    ],
    "val": ["VN_POS_07", "VN_POS_08", "VN_POS_17", "VN_POS_18"],
    "test": ["VN_POS_09", "VN_POS_10", "VN_POS_19", "VN_POS_20"],
}

NEG_SPLIT_CONFIG = {
    "train": ("VN_NEG_01", 360),
    "val":   ("VN_NEG_03", 120),
    "test":  ("VN_NEG_04", 120),
}

# Số khung hình mục tiêu trên mỗi video POS
POS_FRAMES_PER_VIDEO = {
    **{f"VN_POS_{i:02d}": 55 for i in range(1, 11)},   # Giữ nguyên 55 khung/video đã kiểm duyệt của đợt 1 (550 ảnh)
    **{f"VN_POS_{i:02d}": 30 for i in range(11, 21)},  # Cắt thêm 30 khung/video cho 10 video mới đợt 2 (300 ảnh)
}


class Anonymizer:
    """Sử dụng YOLOv8n để phát hiện người (khuôn mặt) và phương tiện (biển số) và làm mờ theo chuẩn bảo mật."""

    def __init__(self, model_path: str = YOLO_ANONYMIZER_PATH, device=DEVICE):
        self.enabled = False
        try:
            self.model = YOLO(model_path)
            self.device = device
            self.enabled = True
        except Exception as e:
            print(f"⚠️ Không thể khởi tạo bộ làm mờ AI: {e}")

    def anonymize(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame

        h, w = frame.shape[:2]
        # Classes: 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
        results = self.model.predict(
            frame, conf=0.35, imgsz=416, classes=[0, 2, 3, 5, 7], verbose=False, device=self.device
        )[0]
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            return frame

        for box in boxes:
            cls_id = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            bx1, by1, bx2, by2 = xyxy
            bw = bx2 - bx1
            bh = by2 - by1
            if bw <= 10 or bh <= 10:
                continue

            if cls_id == 0:
                face_y1 = max(0, by1)
                face_y2 = min(h, by1 + int(bh * 0.28))
                face_x1 = max(0, bx1)
                face_x2 = min(w, bx2)
                roi = frame[face_y1:face_y2, face_x1:face_x2]
                if roi.shape[0] > 5 and roi.shape[1] > 5:
                    kw = max(15, (roi.shape[1] // 3) * 2 + 1)
                    kh = max(15, (roi.shape[0] // 3) * 2 + 1)
                    frame[face_y1:face_y2, face_x1:face_x2] = cv2.GaussianBlur(roi, (kw, kh), 0)

            elif cls_id in [2, 3, 5, 7]:
                plate_y1 = max(0, by2 - int(bh * 0.28))
                plate_y2 = min(h, by2)
                plate_x1 = max(0, bx1 + int(bw * 0.15))
                plate_x2 = min(w, bx2 - int(bw * 0.15))
                roi = frame[plate_y1:plate_y2, plate_x1:plate_x2]
                if roi.shape[0] > 5 and roi.shape[1] > 5:
                    kw = max(15, (roi.shape[1] // 3) * 2 + 1)
                    kh = max(15, (roi.shape[0] // 3) * 2 + 1)
                    frame[plate_y1:plate_y2, plate_x1:plate_x2] = cv2.GaussianBlur(roi, (kw, kh), 0)

        return frame


def load_start_seconds() -> dict:
    """Đọc mốc giây bắt đầu cháy/khói từ data/video_dataset_list.csv để lấy mẫu đúng vùng có cháy/khói."""
    starts = {}
    if not VIDEO_CSV_PATH.exists():
        return starts
    df = pd.read_csv(VIDEO_CSV_PATH)
    for _, r in df.iterrows():
        vid = str(r["video_id"])
        fs = float(r.get("fire_start_sec", -1.0))
        ss = float(r.get("smoke_start_sec", -1.0))
        valid = [x for x in (fs, ss) if x >= 0]
        starts[vid] = min(valid) if valid else 0.0
    return starts


def extract_frames(apply_blur: bool = True):
    print("\n" + "=" * 78)
    print("📸 CẮT KHUNG HÌNH 23 VIDEO VIỆT NAM & CHIA 60/20/20 NGHIÊM NGẶT THEO VIDEO")
    print("=" * 78)
    print(f"• Thư mục nguồn: {VN_DIR.resolve()}")
    print(f"• Thư mục đích:   {OUT_DIR.resolve()}")
    print(f"• Làm mờ mặt/biển số (YOLOv8n): {'BẬT (Bảo mật)' if apply_blur else 'TẮT'}")

    anonymizer = Anonymizer() if apply_blur else None
    start_sec_map = load_start_seconds()

    for split in ["train", "val", "test"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Dọn các khung VN_NEG_01 cũ từng nằm ở val/test để đảm bảo VN_NEG_01 chỉ thuộc tập train
    for old_split in ["val", "test"]:
        for old_img in (OUT_DIR / "images" / old_split).glob("VN_NEG_01_*.jpg"):
            old_img.unlink(missing_ok=True)
        for old_lbl in (OUT_DIR / "labels" / old_split).glob("VN_NEG_01_*.txt"):
            old_lbl.unlink(missing_ok=True)

    manifest_records = []

    # 1. Xử lý 20 Video Dương Tính (VN_POS_01 -> VN_POS_20)
    print("\n[1/2] Đang xử lý 20 video dương tính (VN_POS_01 .. VN_POS_20)...")
    for split, vid_list in POS_SPLIT_CONFIG.items():
        out_img_dir = OUT_DIR / "images" / split
        for vid in vid_list:
            v_path = VN_DIR / f"{vid}.mp4"
            if not v_path.exists():
                print(f"  ⚠️ Không tìm thấy {v_path.name}")
                continue

            # Kiểm tra nếu video đã được cắt đủ khung hình trước đó thì giữ nguyên (không ghi đè)
            existing_imgs = sorted(out_img_dir.glob(f"{vid}_f*.jpg"))
            target_n = POS_FRAMES_PER_VIDEO.get(vid, 30)
            if len(existing_imgs) >= target_n:
                cap = cv2.VideoCapture(str(v_path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                cap.release()
                for img_p in existing_imgs:
                    f_str = img_p.stem.split("_f")[-1]
                    f_idx = int(f_str) if f_str.isdigit() else 0
                    manifest_records.append({
                        "image_name": img_p.name,
                        "video_id": vid,
                        "split": split,
                        "category": "pos",
                        "frame_idx": f_idx,
                        "timestamp_sec": round(f_idx / fps, 2),
                        "relative_path": str(img_p.relative_to(OUT_DIR.parent)),
                    })
                print(f"  ⏩ [{vid}] ({split.upper():<5}) Giữ nguyên {len(existing_imgs):>3} khung đã kiểm duyệt.")
                continue

            cap = cv2.VideoCapture(str(v_path))
            if not cap.isOpened():
                print(f"  ❌ Không mở được {v_path.name}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            dur = total_frames / fps if fps > 0 else 0

            # Lấy mẫu từ thời điểm bắt đầu có cháy/khói để đảm bảo khung hình dương tính chất lượng cao
            start_s = max(0.0, start_sec_map.get(vid, 0.0))
            start_frame = min(max(0, int(start_s * fps)), max(0, total_frames - target_n - 1))
            # Với các video rất dài (> 5 phút), lấy mẫu trong cửa sổ 180 giây kể từ lúc bùng cháy/khói
            max_window_frames = int(180.0 * fps)
            end_frame = min(total_frames - 1, start_frame + max_window_frames) if total_frames > max_window_frames else (total_frames - 1)
            if end_frame <= start_frame:
                start_frame, end_frame = 0, max(0, total_frames - 1)

            n_samples = min(end_frame - start_frame + 1, target_n)
            sample_indices = np.linspace(start_frame, end_frame, num=n_samples, dtype=int)

            # Nếu ảnh cần resize nhẹ khi độ phân giải 4K (như VN_POS_14 3840x2160) để đồng nhất tốc độ
            saved_count = 0
            for f_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                fh, fw = frame.shape[:2]
                if max(fh, fw) > 1920:
                    scale = 1920.0 / max(fh, fw)
                    frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)

                sec = round(f_idx / fps, 2)
                if anonymizer is not None:
                    frame = anonymizer.anonymize(frame)

                out_name = f"{vid}_f{int(f_idx):06d}.jpg"
                out_path = out_img_dir / out_name
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                manifest_records.append({
                    "image_name": out_name,
                    "video_id": vid,
                    "split": split,
                    "category": "pos",
                    "frame_idx": int(f_idx),
                    "timestamp_sec": sec,
                    "relative_path": str(out_path.relative_to(OUT_DIR.parent)),
                })
                saved_count += 1

            cap.release()
            print(f"  ✅ [{vid}] ({split.upper():<5}) Đã cắt mới {saved_count:>3} khung (từ {start_s:.1f}s -> {end_frame/fps:.1f}s / {dur:.1f}s)")

    # 2. Xử lý 3 Video Âm Tính độc lập cho 3 tập (VN_NEG_01 -> train, VN_NEG_03 -> val, VN_NEG_04 -> test)
    print("\n[2/2] Đang xử lý 3 video âm tính độc lập theo tập (Train: VN_NEG_01, Val: VN_NEG_03, Test: VN_NEG_04)...")
    for split, (vid, target_neg_n) in NEG_SPLIT_CONFIG.items():
        v_path = VN_DIR / f"{vid}.mp4"
        out_img_dir = OUT_DIR / "images" / split
        out_lbl_dir = OUT_DIR / "labels" / split
        if not v_path.exists():
            print(f"  ⚠️ Không tìm thấy {v_path.name}")
            continue

        existing_imgs = sorted(out_img_dir.glob(f"{vid}_f*.jpg"))
        cap = cv2.VideoCapture(str(v_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = total_frames / fps if fps > 0 else 0

        if len(existing_imgs) >= target_neg_n:
            cap.release()
            for img_p in existing_imgs[:target_neg_n]:
                f_str = img_p.stem.split("_f")[-1]
                f_idx = int(f_str) if f_str.isdigit() else 0
                manifest_records.append({
                    "image_name": img_p.name,
                    "video_id": vid,
                    "split": split,
                    "category": "neg",
                    "frame_idx": f_idx,
                    "timestamp_sec": round(f_idx / fps, 2),
                    "relative_path": str(img_p.relative_to(OUT_DIR.parent)),
                })
            print(f"  ⏩ [{vid}] ({split.upper():<5}) Đã có đủ {len(existing_imgs):>3} khung âm tính.")
            continue

        sample_indices = np.linspace(0, max(0, total_frames - 2), num=target_neg_n, dtype=int)
        saved_count = 0
        for f_idx in sample_indices:
            out_name = f"{vid}_f{int(f_idx):06d}.jpg"
            out_path = out_img_dir / out_name
            label_path = out_lbl_dir / f"{vid}_f{int(f_idx):06d}.txt"

            if not out_path.exists():
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                if anonymizer is not None:
                    frame = anonymizer.anonymize(frame)
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            label_path.touch(exist_ok=True)
            manifest_records.append({
                "image_name": out_name,
                "video_id": vid,
                "split": split,
                "category": "neg",
                "frame_idx": int(f_idx),
                "timestamp_sec": round(f_idx / fps, 2),
                "relative_path": str(out_path.relative_to(OUT_DIR.parent)),
            })
            saved_count += 1

        cap.release()
        print(f"  ✅ [{vid}] ({split.upper():<5}) Lưu {saved_count:>3} khung âm tính (Thời lượng: {dur/60:.1f} phút)")

    # Lưu manifest CSV
    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")

    # Kiểm tra tính độc lập theo video giữa các tập
    train_vids = set(manifest_df[manifest_df["split"] == "train"]["video_id"])
    val_vids = set(manifest_df[manifest_df["split"] == "val"]["video_id"])
    test_vids = set(manifest_df[manifest_df["split"] == "test"]["video_id"])
    overlap = (train_vids & val_vids) | (train_vids & test_vids) | (val_vids & test_vids)
    assert len(overlap) == 0, f"Lỗi rò rỉ video giữa các tập: {overlap}"

    total_imgs = len(manifest_df)
    train_n = len(manifest_df[manifest_df["split"] == "train"])
    val_n = len(manifest_df[manifest_df["split"] == "val"])
    test_n = len(manifest_df[manifest_df["split"] == "test"])

    print("\n" + "=" * 78)
    print("🎉 HOÀN THÀNH CẬP NHẬT BỘ KHUNG HÌNH VIỆT NAM (23 VIDEO ĐỘC LẬP):")
    print(f"• Tổng số khung hình: {total_imgs} khung (Giao nhau video giữa các tập = {len(overlap)})")
    print(f"  - Train: {train_n:>4} ảnh ({train_n/total_imgs*100:.1f}%) | {len(train_vids)} video: {sorted(train_vids)}")
    print(f"  - Val:   {val_n:>4} ảnh ({val_n/total_imgs*100:.1f}%) | {len(val_vids)} video: {sorted(val_vids)}")
    print(f"  - Test:  {test_n:>4} ảnh ({test_n/total_imgs*100:.1f}%) | {len(test_vids)} video: {sorted(test_vids)}")
    print(f"• File Manifest: {MANIFEST_PATH.resolve()}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cắt khung hình 23 video Việt Nam và chia 60/20/20 theo video")
    parser.add_argument("--no-blur", action="store_true", help="Không áp dụng làm mờ mặt và biển số")
    args = parser.parse_args()
    extract_frames(apply_blur=not args.no_blur)
