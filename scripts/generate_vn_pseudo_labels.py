"""Tự động sinh nhãn sơ bộ (Pseudo-labels) bằng mô hình best.pt cho tập ảnh Việt Nam (MT1, MT4).

Mục đích:
- Hỗ trợ gán nhãn nhanh (Phương án B): AI dự đoán trước khung bao cho các ảnh có cháy/khói (VN_POS_*).
- Xuất nhãn chuẩn định dạng YOLO: datasets/vn_dataset/labels/{train,val,test}/*.txt
- Tạo bảng theo dõi kiểm duyệt review_status.csv để người dùng nghiệm thu thủ công từng ảnh theo đúng đề cương NCKH.
"""
import sys
from pathlib import Path
import cv2
import pandas as pd
import torch
from ultralytics import YOLO

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

DATASET_DIR = Path("datasets/vn_dataset")
MODEL_PATH = "best.pt"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
REVIEW_LOG_PATH = DATASET_DIR / "review_status.csv"


def generate_pseudo_labels(conf_thresh: float = 0.25):
    print("\n" + "=" * 75)
    print("🤖 TỰ ĐỘNG SINH NHÃN SƠ BỘ BẰNG MÔ HÌNH BEST.PT (CUDA GPU)")
    print("=" * 75)
    print(f"• Dataset:   {DATASET_DIR.resolve()}")
    print(f"• Model:     {MODEL_PATH} (Thiết bị: {'CUDA:0' if DEVICE == 0 else 'CPU'})")
    print(f"• Ngưỡng tin cậy (conf): {conf_thresh}")

    if not DATASET_DIR.exists():
        print(f"❌ Không tìm thấy thư mục {DATASET_DIR}. Hãy chạy scripts/extract_vn_frames.py trước.")
        return

    model = YOLO(MODEL_PATH)

    review_records = []
    total_pos_images = 0
    total_boxes = 0

    for split in ["train", "val", "test"]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # Lấy tất cả ảnh dương tính VN_POS_*
        pos_images = sorted(img_dir.glob("VN_POS_*.jpg"))
        print(f"\n📂 Đang xử lý tập [{split.upper()}]: {len(pos_images)} ảnh dương tính...")

        for img_path in pos_images:
            total_pos_images += 1
            label_file = lbl_dir / f"{img_path.stem}.txt"

            # Đọc ảnh để lấy kích thước chính xác
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Dự đoán với YOLO
            results = model.predict(img, conf=conf_thresh, verbose=False, device=DEVICE)[0]
            boxes = results.boxes

            lines = []
            smoke_count = 0
            fire_count = 0

            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    cls_id = int(b.cls[0])
                    # Bounding box chuẩn hóa YOLO: x_center, y_center, width, height (0-1)
                    xywhn = b.xywhn[0].cpu().numpy()
                    xc, yc, bw, bh = xywhn
                    lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    if cls_id == 0:
                        smoke_count += 1
                    elif cls_id == 1:
                        fire_count += 1

            # Ghi file nhãn .txt
            label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n_boxes = len(lines)
            total_boxes += n_boxes

            review_records.append({
                "image_name": img_path.name,
                "split": split,
                "label_file": str(label_file.relative_to(DATASET_DIR)),
                "num_boxes": n_boxes,
                "num_smoke": smoke_count,
                "num_fire": fire_count,
                "is_reviewed": False,  # Đánh dấu chờ người dùng kiểm duyệt
                "modified": False,
            })

    # Cập nhật hoặc lưu file review_status.csv
    review_df = pd.DataFrame(review_records)
    review_df.to_csv(REVIEW_LOG_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 75)
    print("🎉 HOÀN THÀNH SINH NHÃN SƠ BỘ:")
    print(f"• Tổng số ảnh dương tính đã gán nhãn: {total_pos_images} ảnh")
    print(f"• Tổng số khung bao (bounding box) phát hiện: {total_boxes} boxes")
    print(f"• Thư mục nhãn đã cập nhật: {DATASET_DIR / 'labels'}")
    print(f"• Bảng theo dõi kiểm duyệt lưu tại: {REVIEW_LOG_PATH.resolve()}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sinh nhãn sơ bộ cho tập ảnh Việt Nam")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng tự tin (confidence threshold), mặc định 0.25")
    args = parser.parse_args()

    generate_pseudo_labels(conf_thresh=args.conf)
