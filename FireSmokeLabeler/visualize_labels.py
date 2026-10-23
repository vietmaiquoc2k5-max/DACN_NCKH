from pathlib import Path
import cv2

# ===== CẤU HÌNH ĐƯỜNG DẪN LINH HOẠT =====
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "dataset").is_dir() or any(SCRIPT_DIR.glob("dataset*")):
    ROOT = SCRIPT_DIR
elif Path(r"C:\FireSmokeLabeler").is_dir():
    ROOT = Path(r"C:\FireSmokeLabeler")
else:
    ROOT = SCRIPT_DIR

# Tự tìm dataset_auto_* mới nhất có đủ images và labels.
# Nếu muốn chọn chính xác, điền đường dẫn vào DATASET_OVERRIDE.
DATASET_OVERRIDE = None  # Ví dụ: r"C:\FireSmokeLabeler\dataset_auto_20261022_120000"

OUTPUT_ROOT = ROOT / "visualized"
CLASS_NAMES = {0: "smoke", 1: "fire"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def find_dataset():
    if DATASET_OVERRIDE:
        candidate = Path(DATASET_OVERRIDE)
        if not candidate.exists():
            raise FileNotFoundError(f"Không tìm thấy dataset: {candidate}")
        return candidate

    candidates = []
    for p in ROOT.glob("dataset*"):
        if (p / "images").is_dir() and (p / "labels").is_dir():
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy dataset có thư mục images và labels trong {ROOT}.\n"
            "Hãy sửa DATASET_OVERRIDE thành đúng đường dẫn dataset."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)

def get_label_path(image_path: Path, images_dir: Path, labels_dir: Path):
    # Hỗ trợ cấu trúc YOLO images/train -> labels/train.
    relative = image_path.relative_to(images_dir)
    return (labels_dir / relative).with_suffix(".txt")

def draw_labels(image, label_path: Path):
    h, w = image.shape[:2]
    counts = {"fire": 0, "smoke": 0, "unknown": 0}

    if not label_path.exists():
        cv2.putText(image, "NO LABEL FILE", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        return image, counts, False

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()

    for line_no, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            print(f"[BỎ QUA] Nhãn sai định dạng: {label_path}, dòng {line_no}")
            continue

        try:
            class_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])
        except ValueError:
            print(f"[BỎ QUA] Không đọc được nhãn: {label_path}, dòng {line_no}")
            continue

        # YOLO dùng tọa độ tâm và kích thước chuẩn hóa từ 0 đến 1.
        x1 = round((xc - bw / 2) * w)
        y1 = round((yc - bh / 2) * h)
        x2 = round((xc + bw / 2) * w)
        y2 = round((yc + bh / 2) * h)

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        # OpenCV dùng BGR: fire màu đỏ, smoke màu xanh dương.
        color = (0, 0, 255) if name == "fire" else (255, 100, 0)
        if name not in counts:
            counts["unknown"] += 1
        else:
            counts[name] += 1

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = name
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, min(0.8, w / 1200))
        (tw, th), baseline = cv2.getTextSize(label, font, scale, 2)
        label_y = max(th + 8, y1)
        cv2.rectangle(image, (x1, label_y - th - 8),
                      (min(w - 1, x1 + tw + 8), label_y + baseline - 2),
                      color, -1)
        cv2.putText(image, label, (x1 + 4, label_y - 4),
                    font, scale, (255, 255, 255), 2, cv2.LINE_AA)

    return image, counts, True

def main():
    dataset = find_dataset()
    images_root = dataset / "images"
    labels_root = dataset / "labels"

    if not images_root.is_dir() or not labels_root.is_dir():
        raise FileNotFoundError("Dataset phải có cả thư mục images và labels.")

    # Lưu ngoài dataset, trong thư mục riêng; không đụng vào ảnh gốc.
    output_root = OUTPUT_ROOT / dataset.name
    output_root.mkdir(parents=True, exist_ok=True)

    image_files = [p for p in images_root.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not image_files:
        print(f"Không tìm thấy ảnh trong: {images_root}")
        return

    total = 0
    missing_labels = 0
    total_fire = 0
    total_smoke = 0

    print(f"Dataset: {dataset}")
    print(f"Ảnh kết quả: {output_root}")
    print(f"Tìm thấy {len(image_files)} ảnh. Bắt đầu vẽ...")

    for index, image_path in enumerate(image_files, start=1):
        relative = image_path.relative_to(images_root)
        label_path = get_label_path(image_path, images_root, labels_root)

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[BỎ QUA] Không đọc được ảnh: {image_path}")
            continue

        annotated, counts, label_exists = draw_labels(image, label_path)
        if not label_exists:
            missing_labels += 1

        out_path = output_root / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), annotated):
            print(f"[LỖI] Không lưu được: {out_path}")
            continue

        total += 1
        total_fire += counts["fire"]
        total_smoke += counts["smoke"]

        if index % 100 == 0 or index == len(image_files):
            print(f"Tiến độ: {index}/{len(image_files)}")

    print("\n===== HOÀN TẤT =====")
    print(f"Đã lưu: {total} ảnh")
    print(f"Khung fire: {total_fire}")
    print(f"Khung smoke: {total_smoke}")
    print(f"Ảnh thiếu file nhãn: {missing_labels}")
    print(f"Thư mục kết quả: {output_root}")
    print("Ảnh gốc và file .txt không bị thay đổi.")

if __name__ == "__main__":
    main()
