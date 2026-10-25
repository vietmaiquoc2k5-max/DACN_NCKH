import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

# Đảm bảo in tiếng Việt không lỗi bảng mã trên Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def validate_dataset(data_yaml_path):
    """
    Xác minh cấu trúc dataset, tập train/val, thứ tự class và tính hợp lệ trước khi huấn luyện.
    Đảm bảo:
    - Không chạy huấn luyện nếu dataset rỗng hoặc val rỗng.
    - Cảnh báo về nhãn tự động gắn bởi AI.
    """
    yaml_file = Path(data_yaml_path)
    if not yaml_file.exists():
        print(f"❌ LỖI: Không tìm thấy file cấu hình dataset: {data_yaml_path}")
        return False, None

    with open(yaml_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root_path = Path(cfg.get("path", "."))
    if not root_path.is_absolute():
        # Đường dẫn tương đối tính từ thư mục chứa file yaml hoặc thư mục hiện tại
        candidate1 = Path.cwd() / root_path
        candidate2 = yaml_file.parent / root_path
        root_path = candidate1 if candidate1.exists() else candidate2

    train_rel = cfg.get("train", "images/train")
    val_rel = cfg.get("val", "images/val")

    train_img_dir = root_path / train_rel
    val_img_dir = root_path / val_rel

    print("\n" + "="*60)
    print("🔍 KIỂM TRA TẬP DỮ LIỆU HUẤN LUYỆN (DATASET INTEGRITY CHECK)")
    print("="*60)
    print(f"• File cấu hình: {yaml_file.resolve()}")
    print(f"• Thư mục gốc dataset: {root_path}")
    print(f"• Thư mục ảnh Train: {train_img_dir}")
    print(f"• Thư mục ảnh Val: {val_img_dir}")
    print(f"• Danh sách Class: {cfg.get('names')}")

    if not train_img_dir.exists():
        print(f"❌ LỖI: Thư mục ảnh Train không tồn tại: {train_img_dir}")
        return False, None

    if not val_img_dir.exists():
        print(f"❌ LỖI: Thư mục ảnh Validation không tồn tại: {val_img_dir}")
        return False, None

    train_images = list(train_img_dir.glob("*.*"))
    val_images = list(val_img_dir.glob("*.*"))

    print(f"  -> Số lượng ảnh Train: {len(train_images):,} ảnh")
    print(f"  -> Số lượng ảnh Val:   {len(val_images):,} ảnh")

    if len(train_images) == 0:
        print("❌ LỖI NGHIÊM TRỌNG: Thư mục Train không có ảnh nào để huấn luyện!")
        return False, None

    if len(val_images) == 0:
        print("❌ LỖI NGHIÊM TRỌNG: Thư mục Validation rỗng!")
        print("   Quy tắc an toàn: Không tự động huấn luyện khi tập validation rỗng.")
        return False, None

    # Kiểm tra thư mục labels tương ứng (tương thích chuẩn Ultralytics: thay 'images' bằng 'labels')
    labels_train = Path(str(train_img_dir).replace("images", "labels"))
    if not labels_train.exists():
        labels_train = root_path / "labels" / "train"

    labels_val = Path(str(val_img_dir).replace("images", "labels"))
    if not labels_val.exists():
        labels_val = root_path / "labels" / "val"

    train_labels = list(labels_train.glob("*.txt")) if labels_train.exists() else []
    val_labels = list(labels_val.glob("*.txt")) if labels_val.exists() else []

    print(f"  -> Số lượng file nhãn Train: {len(train_labels):,} file")
    print(f"  -> Số lượng file nhãn Val:   {len(val_labels):,} file")

    # Thống kê phân bố lớp
    class_names = cfg.get("names", {})
    if isinstance(class_names, list):
        class_names = dict(enumerate(class_names))
    
    counts = {str(k): 0 for k in class_names.keys()}
    total_bboxes = 0
    for lf in train_labels + val_labels:
        try:
            with open(lf, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = str(parts[0])
                        counts[cls_id] = counts.get(cls_id, 0) + 1
                        total_bboxes += 1
        except Exception:
            pass

    print(f"  -> Tổng số bounding box dự đoán: {total_bboxes:,}")
    for cid, cname in class_names.items():
        print(f"     Class {cid} ({cname}): {counts.get(str(cid), 0):,} bboxes")

    print("\n" + "!"*60)
    print("⚠️  LƯU Ý QUAN TRỌNG VỀ NHÃN TỰ ĐỘNG:")
    print("• Dữ liệu trên được trích xuất và gắn nhãn tự động từ video bởi model AI.")
    print("• Hãy chắc chắn bạn đã chạy trực quan hóa để rà soát nhãn sai:")
    print("  Lệnh: python FireSmokeLabeler/visualize_labels.py")
    print("!"*60 + "\n")

    return True, cfg

def train_model(data_yaml="data.yaml", epochs=50, imgsz=640, batch=16, resume=False):
    print("="*60)
    print("🔥 TIẾN TRÌNH HUẤN LUYỆN YOLOv8 NHẬN DIỆN LỬA & KHÓI 🔥")
    print("="*60)

    # 1. Kiểm tra toàn vẹn Dataset
    valid, cfg = validate_dataset(data_yaml)
    if not valid:
        print("⛔ Dừng tiến trình: Dataset chưa đạt tiêu chuẩn kiểm tra an toàn.")
        return False

    # 2. Kiểm tra phần cứng
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"🍀 Thiết bị sử dụng: {device.upper()}")
    if device == '0':
        print(f"  -> GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  -> Chạy trên CPU. (Lưu ý: Có thể mất nhiều thời gian hơn)")

    # 3. Base model
    base_model_path = "yolov8n.pt"
    if os.path.exists("best.pt"):
        print(f"💡 Tìm thấy model hiện hành: best.pt. Sử dụng làm nền tảng fine-tune.")
        base_model_path = "best.pt"
    else:
        print(f"⏳ Tải mô hình nền: {base_model_path}...")

    model = YOLO(base_model_path)

    # 4. Huấn luyện an toàn, lưu ra thư mục riêng biệt KHÔNG ghi đè best.pt
    project_dir = "runs/detect"
    run_name = "train_fire_smoke_v2"
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Bắt đầu huấn luyện {epochs} epochs...")
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=2,
            project=project_dir,
            name=run_name,
            resume=resume,
            exist_ok=True
        )

        trained_best = Path(project_dir) / run_name / "weights" / "best.pt"
        output_model_v2 = models_dir / "fire_smoke_v2.pt"

        if trained_best.exists():
            import shutil
            shutil.copy2(str(trained_best), str(output_model_v2))
            print("\n" + "="*60)
            print("🎉 HUẤN LUYỆN THÀNH CÔNG! 🎉")
            print("="*60)
            print(f"• Trọng số mới được lưu an toàn tại: {output_model_v2}")
            print(f"• File gốc của lần train: {trained_best}")
            print("• Model đang chạy 'best.pt' được giữ nguyên an toàn, KHÔNG bị ghi đè.")
            print("\n👉 CÁC BƯỚC TIẾP THEO:")
            print("1. Chạy đánh giá so sánh model:")
            print(f"   python evaluate_model.py --model {output_model_v2} --compare best.pt")
            print("2. Khi hài lòng với độ chính xác, bạn có thể nạp model mới qua giao diện Admin")
            print(f"   hoặc cấu hình đường dẫn model mới vào detector: '{output_model_v2}'.")
            return True
        else:
            print("⚠️ Huấn luyện xong nhưng không tìm thấy file trọng số đầu ra.")
            return False

    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình huấn luyện: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình YOLOv8 nhận diện lửa và khói")
    parser.add_argument("--data", default="data.yaml", help="Đường dẫn file cấu hình data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Số epochs huấn luyện")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh")
    parser.add_argument("--batch", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--validate-only", action="store_true", help="Chỉ kiểm tra dataset, không huấn luyện")
    args = parser.parse_args()

    if args.validate_only:
        valid, _ = validate_dataset(args.data)
        sys.exit(0 if valid else 1)
    else:
        train_model(data_yaml=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)

if __name__ == "__main__":
    main()
