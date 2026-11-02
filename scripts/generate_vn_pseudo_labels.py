"""Tự động sinh nhãn sơ bộ (Pseudo-labels) bằng mô hình best.pt cho các khung hình Việt Nam mới (MT1, MT4).

Đặc điểm:
- Bảo toàn 100% các file nhãn và trạng thái `is_reviewed=True` của các ảnh đã được người dùng kiểm duyệt.
- Chỉ sinh nhãn sơ bộ cho các ảnh dương tính mới chưa có nhãn hoặc chưa kiểm duyệt.
- Xuất nhãn chuẩn định dạng YOLO: datasets/vn_dataset/labels/{train,val,test}/*.txt
- Cập nhật bảng theo dõi kiểm duyệt: datasets/vn_dataset/review_status.csv
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
    print("🤖 TỰ ĐỘNG SINH NHÃN SƠ BỘ CHO CÁC KHUNG HÌNH VIỆT NAM MỚI (BẢO TOÀN NHÃN ĐÃ DUYỆT)")
    print("=" * 75)
    print(f"• Dataset:   {DATASET_DIR.resolve()}")
    print(f"• Model:     {MODEL_PATH} (Thiết bị: {'CUDA:0' if DEVICE == 0 else 'CPU'})")
    print(f"• Ngưỡng tin cậy (conf): {conf_thresh}")

    if not DATASET_DIR.exists():
        print(f"❌ Không tìm thấy thư mục {DATASET_DIR}. Hãy chạy scripts/extract_vn_frames.py trước.")
        return

    # Đọc trạng thái kiểm duyệt hiện có để không ghi đè các ảnh đã kiểm duyệt
    existing_review = {}
    if REVIEW_LOG_PATH.exists():
        old_df = pd.read_csv(REVIEW_LOG_PATH)
        for _, r in old_df.iterrows():
            existing_review[str(r["image_name"])] = {
                "is_reviewed": bool(r.get("is_reviewed", False)),
                "modified": bool(r.get("modified", False)),
                "num_boxes": int(r.get("num_boxes", 0)),
                "num_smoke": int(r.get("num_smoke", 0)),
                "num_fire": int(r.get("num_fire", 0)),
            }

    model = YOLO(MODEL_PATH)

    review_records = []
    total_pos_images = 0
    newly_labeled = 0
    preserved_reviewed = 0
    total_boxes = 0

    for split in ["train", "val", "test"]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        lbl_dir.mkdir(parents=True, exist_ok=True)

        pos_images = sorted(img_dir.glob("VN_POS_*.jpg"))
        print(f"\n📂 Đang kiểm tra tập [{split.upper()}]: {len(pos_images)} ảnh dương tính...")

        for img_path in pos_images:
            total_pos_images += 1
            label_file = lbl_dir / f"{img_path.stem}.txt"
            old_info = existing_review.get(img_path.name)

            # Nếu ảnh đã được duyệt và có sẵn file nhãn -> bảo toàn tuyệt đối
            if old_info and old_info["is_reviewed"] and label_file.exists():
                preserved_reviewed += 1
                total_boxes += old_info["num_boxes"]
                review_records.append({
                    "image_name": img_path.name,
                    "split": split,
                    "label_file": str(label_file.relative_to(DATASET_DIR)),
                    "num_boxes": old_info["num_boxes"],
                    "num_smoke": old_info["num_smoke"],
                    "num_fire": old_info["num_fire"],
                    "is_reviewed": True,
                    "modified": old_info["modified"],
                })
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            results = model.predict(img, conf=conf_thresh, imgsz=416, verbose=False, device=DEVICE)[0]
            boxes = results.boxes

            lines = []
            smoke_count = 0
            fire_count = 0

            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    cls_id = int(b.cls[0])
                    xywhn = b.xywhn[0].cpu().numpy()
                    xc, yc, bw, bh = xywhn
                    lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    if cls_id == 0:
                        smoke_count += 1
                    elif cls_id == 1:
                        fire_count += 1

            label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n_boxes = len(lines)
            total_boxes += n_boxes
            newly_labeled += 1

            review_records.append({
                "image_name": img_path.name,
                "split": split,
                "label_file": str(label_file.relative_to(DATASET_DIR)),
                "num_boxes": n_boxes,
                "num_smoke": smoke_count,
                "num_fire": fire_count,
                "is_reviewed": False,
                "modified": False,
            })

    review_df = pd.DataFrame(review_records)
    review_df.to_csv(REVIEW_LOG_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 75)
    print("🎉 HOÀN THÀNH CẬP NHẬT NHÃN CHO TẬP ẢNH VIỆT NAM:")
    print(f"• Tổng số ảnh dương tính (VN_POS_01..20): {total_pos_images} ảnh")
    print(f"  - Đã bảo toàn (Reviewed trước đó):     {preserved_reviewed} ảnh")
    print(f"  - Đã sinh nhãn mới (VN_POS_11..20):    {newly_labeled} ảnh")
    print(f"• Tổng số khung bao (bounding box):       {total_boxes} boxes")
    print(f"• Bảng theo dõi kiểm duyệt:               {REVIEW_LOG_PATH.resolve()}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sinh nhãn sơ bộ cho tập ảnh Việt Nam")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng tự tin (confidence threshold), mặc định 0.25")
    args = parser.parse_args()

    generate_pseudo_labels(conf_thresh=args.conf)
