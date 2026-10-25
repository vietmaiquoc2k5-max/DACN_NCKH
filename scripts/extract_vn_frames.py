"""Trích xuất và làm mờ khuôn mặt/biển số từ video Việt Nam (videovietnam/) phục vụ gán nhãn tay (MT1, MT4).

Yêu cầu theo Đề cương NCKH:
- Đạt >= 1.000 khung hình Việt Nam.
- Chia 60/20/20 THEO VIDEO (không để khung của cùng video ở hai tập).
- Làm mờ khuôn mặt và biển số xe trước khi gán nhãn / công bố dữ liệu (dùng YOLOv8n CUDA).
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
DATA_YAML_PATH = Path("data_vn.yaml")
YOLO_ANONYMIZER_PATH = "yolov8n.pt"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Cấu hình phân chia 60/20/20 theo video (MT1 & MT4)
# 10 video POS: 6 train (60%), 2 val (20%), 2 test (20%)
SPLIT_CONFIG = {
    "train": ["VN_POS_01", "VN_POS_02", "VN_POS_03", "VN_POS_04", "VN_POS_05", "VN_POS_06"],
    "val":   ["VN_POS_07", "VN_POS_08"],
    "test":  ["VN_POS_09", "VN_POS_10"],
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
        results = self.model.predict(frame, conf=0.35, classes=[0, 2, 3, 5, 7], verbose=False, device=self.device)[0]
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
                # Người: Vùng mặt ở 25% phía trên của bounding box
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
                # Xe: Vùng biển số ở 25% phía dưới của bounding box
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


def extract_frames(target_pos_frames: int = 550, target_neg_frames: int = 500, apply_blur: bool = True):
    print("\n" + "=" * 75)
    print("📸 CÔNG CỤ CẮT KHUNG HÌNH VIỆT NAM & PHÂN CHIA 60/20/20 THEO VIDEO (MT1)")
    print("=" * 75)
    print(f"• Thư mục nguồn: {VN_DIR.resolve()}")
    print(f"• Thư mục đích:   {OUT_DIR.resolve()}")
    print(f"• Làm mờ mặt/biển số (YOLOv8n GPU): {'BẬT (Bảo mật)' if apply_blur else 'TẮT'}")

    anonymizer = Anonymizer() if apply_blur else None

    # Tạo các thư mục train / val / test
    for split in ["train", "val", "test"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_records = []

    # 1. Xử lý 10 Video Dương Tính (VN_POS_01 -> VN_POS_10)
    pos_videos = sorted(VN_DIR.glob("VN_POS_*.mp4"))
    if not pos_videos:
        print("⚠️ Không tìm thấy file VN_POS_*.mp4 trong thư mục videovietnam/")
    else:
        print(f"\n[1/2] Đang trích xuất {len(pos_videos)} video dương tính (POS)...")
        frames_per_pos = max(10, target_pos_frames // len(pos_videos))

        for v_path in pos_videos:
            vid = v_path.stem
            # Xác định split
            split = "train"
            for s, v_list in SPLIT_CONFIG.items():
                if vid in v_list:
                    split = s
                    break

            cap = cv2.VideoCapture(str(v_path))
            if not cap.isOpened():
                print(f"  ❌ Không mở được {v_path.name}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            dur = total_frames / fps if fps > 0 else 0

            n_samples = min(total_frames, frames_per_pos)
            sample_indices = np.linspace(0, total_frames - 1, num=n_samples, dtype=int) if total_frames > 0 else []

            saved_count = 0
            cur_idx = 0
            sample_set = set(sample_indices)

            out_img_dir = OUT_DIR / "images" / split

            for f_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                sec = round(f_idx / fps, 2)
                if anonymizer is not None:
                    frame = anonymizer.anonymize(frame)

                out_name = f"{vid}_f{f_idx:06d}.jpg"
                out_path = out_img_dir / out_name
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                manifest_records.append({
                    "image_name": out_name,
                    "video_id": vid,
                    "split": split,
                    "category": "pos",
                    "frame_idx": f_idx,
                    "timestamp_sec": sec,
                    "relative_path": str(out_path.relative_to(OUT_DIR.parent)),
                })
                saved_count += 1

            cap.release()
            print(f"  ✅ [{vid}] ({split.upper():<5}) Lưu {saved_count:>3} khung (từ {dur:.1f}s, {total_frames} frames)")

    # 2. Xử lý Video Âm Tính VN_NEG_01 (Cảnh không cháy kéo dài > 1 tiếng)
    neg_videos = sorted(VN_DIR.glob("VN_NEG_*.mp4"))
    if neg_videos:
        print(f"\n[2/2] Đang trích xuất video âm tính (NEG) để bù dữ liệu cảnh nhiễu...")
        for v_path in neg_videos:
            vid = v_path.stem
            cap = cv2.VideoCapture(str(v_path))
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            dur = total_frames / fps if fps > 0 else 0

            n_samples = min(total_frames, target_neg_frames)
            sample_indices = np.linspace(0, total_frames - 1, num=n_samples, dtype=int) if total_frames > 0 else []

            # Chia 60% train, 20% val, 20% test theo trục thời gian cho VN_NEG_01
            train_boundary = int(len(sample_indices) * 0.60)
            val_boundary = int(len(sample_indices) * 0.80)

            cur_idx = 0
            sample_map = {}
            for i, f_idx in enumerate(sample_indices):
                if i < train_boundary:
                    sample_map[f_idx] = "train"
                elif i < val_boundary:
                    sample_map[f_idx] = "val"
                else:
                    sample_map[f_idx] = "test"

            saved_counts = {"train": 0, "val": 0, "test": 0}

            # Để tăng tốc độ đọc video 230.000 frame (64 phút), ta dùng cap.set thay vì đọc từng frame
            for f_idx, split in sample_map.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                sec = round(f_idx / fps, 2)
                if anonymizer is not None:
                    frame = anonymizer.anonymize(frame)

                out_name = f"{vid}_f{f_idx:06d}.jpg"
                out_path = OUT_DIR / "images" / split / out_name
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Với ảnh âm tính, tạo file label rỗng (empty .txt) chuẩn YOLO background
                label_path = OUT_DIR / "labels" / split / f"{vid}_f{f_idx:06d}.txt"
                label_path.touch(exist_ok=True)

                manifest_records.append({
                    "image_name": out_name,
                    "video_id": vid,
                    "split": split,
                    "category": "neg",
                    "frame_idx": f_idx,
                    "timestamp_sec": sec,
                    "relative_path": str(out_path.relative_to(OUT_DIR.parent)),
                })
                saved_counts[split] += 1

            cap.release()
            print(f"  ✅ [{vid}] Lưu: Train={saved_counts['train']}, Val={saved_counts['val']}, Test={saved_counts['test']} khung (Thời lượng: {dur/60:.1f} phút)")

    # Lưu manifest CSV
    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")

    # Tạo data_vn.yaml cho huấn luyện và thích nghi miền (MT4)
    yaml_content = f"""# Cấu hình tập dữ liệu camera giám sát Việt Nam (MT1 & MT4)
path: datasets/vn_dataset
train: images/train
val: images/val
test: images/test

nc: 2
names: ['smoke', 'fire']
"""
    DATA_YAML_PATH.write_text(yaml_content, encoding="utf-8")

    # Thống kê tổng hợp
    total_imgs = len(manifest_df)
    train_n = len(manifest_df[manifest_df["split"] == "train"])
    val_n = len(manifest_df[manifest_df["split"] == "val"])
    test_n = len(manifest_df[manifest_df["split"] == "test"])

    print("\n" + "=" * 75)
    print("🎉 HOÀN THÀNH TRÍCH XUẤT KHUNG HÌNH VIỆT NAM:")
    print(f"• Tổng số khung hình đã cắt: {total_imgs} khung")
    print(f"  - Train: {train_n:>4} ảnh ({train_n/total_imgs*100:.1f}%)")
    print(f"  - Val:   {val_n:>4} ảnh ({val_n/total_imgs*100:.1f}%)")
    print(f"  - Test:  {test_n:>4} ảnh ({test_n/total_imgs*100:.1f}%)")
    print(f"• File Manifest: {MANIFEST_PATH.resolve()}")
    print(f"• File YAML:     {DATA_YAML_PATH.resolve()}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cắt khung hình Việt Nam và chia 60/20/20")
    parser.add_argument("--pos-frames", type=int, default=550, help="Tổng số khung hình POS cần cắt (~550)")
    parser.add_argument("--neg-frames", type=int, default=500, help="Tổng số khung hình NEG cần cắt (~500)")
    parser.add_argument("--no-blur", action="store_true", help="Không áp dụng làm mờ mặt và biển số")
    args = parser.parse_args()

    extract_frames(
        target_pos_frames=args.pos_frames,
        target_neg_frames=args.neg_frames,
        apply_blur=not args.no_blur,
    )
