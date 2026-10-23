from pathlib import Path
import cv2
import random
import shutil
import yaml
from ultralytics import YOLO

# ====== CẤU HÌNH ĐƯỜNG DẪN LINH HOẠT ======
SCRIPT_DIR = Path(__file__).resolve().parent
# Ưu tiên thư mục hiện tại trong dự án, dự phòng sang C:\FireSmokeLabeler nếu không có
if (SCRIPT_DIR / "videos").is_dir() or (SCRIPT_DIR / "model").is_dir():
    ROOT = SCRIPT_DIR
elif Path(r"C:\FireSmokeLabeler").is_dir():
    ROOT = Path(r"C:\FireSmokeLabeler")
else:
    ROOT = SCRIPT_DIR

# Tìm model: ưu tiên thư mục model/, sau đó đến root dự án, cuối cùng là C:\
if (ROOT / "model" / "best.pt").is_file():
    MODEL_PATH = ROOT / "model" / "best.pt"
elif (ROOT.parent / "best.pt").is_file():
    MODEL_PATH = ROOT.parent / "best.pt"
elif Path(r"C:\FireSmokeLabeler\model\best.pt").is_file():
    MODEL_PATH = Path(r"C:\FireSmokeLabeler\model\best.pt")
else:
    MODEL_PATH = ROOT / "model" / "best.pt"

VIDEO_DIR = ROOT / "videos"
DATASET_DIR = ROOT / "dataset"

FRAME_STRIDE = 5       # Lấy 1 khung hình sau mỗi 5 khung hình video
CONFIDENCE = 0.25      # Ngưỡng dự đoán sơ bộ
IMAGE_SIZE = 640
VAL_RATIO = 0.20
SEED = 42

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg"}


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")
    if not VIDEO_DIR.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục video: {VIDEO_DIR}\n"
            "Hãy tạo thư mục videos và chép các video AVI/MP4 vào đó."
        )

    videos = sorted(p for p in VIDEO_DIR.iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise FileNotFoundError(f"Không tìm thấy video AVI/MP4 trong {VIDEO_DIR}")

    # Chỉ chấp nhận model có đúng hai lớp mong muốn; thứ tự chuẩn hóa là smoke=0, fire=1.
    model = YOLO(str(MODEL_PATH))
    names = model.names
    if isinstance(names, list):
        names = dict(enumerate(names))
    normalized = {int(k): str(v).strip().lower() for k, v in names.items()}
    smoke_ids = [k for k, v in normalized.items() if v == "smoke"]
    fire_ids = [k for k, v in normalized.items() if v == "fire"]
    if not smoke_ids or not fire_ids:
        raise ValueError(
            f"Model phải có lớp fire và smoke. Các lớp model hiện có: {normalized}"
        )

    # Không tự động xóa dataset cũ để tránh mất dữ liệu.
    # Script tạo thư mục mới có tên riêng cho mỗi lần chạy.
    run_dir = DATASET_DIR
    if run_dir.exists() and any(run_dir.iterdir()):
        from datetime import datetime
        run_dir = ROOT / f"dataset_auto_{datetime.now():%Y%m%d_%H%M%S}"
        print(f"Dataset cũ đã tồn tại; tạo bộ mới tại: {run_dir}")

    for split in ("train", "val"):
        (run_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (run_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    shuffled = videos[:]
    rng.shuffle(shuffled)
    if len(shuffled) >= 2:
        val_count = max(1, round(len(shuffled) * VAL_RATIO))
        val_videos = set(shuffled[:val_count])
    else:
        val_videos = set()
        print("CẢNH BÁO: Chỉ có 1 video; chưa thể tạo validation độc lập theo video.")

    total_frames = 0
    total_detections = 0

    for video_index, video_path in enumerate(videos, start=1):
        split = "val" if video_path in val_videos else "train"
        print(f"\n[{video_index}/{len(videos)}] Đang xử lý: {video_path.name} -> {split}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  BỎ QUA: Không mở được video {video_path.name}")
            continue

        frame_index = 0
        saved_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % FRAME_STRIDE != 0:
                    frame_index += 1
                    continue

                # Tên ảnh gắn với video nguồn để tránh trùng tên.
                image_name = f"{video_path.stem}_frame{frame_index:07d}.jpg"
                image_path = run_dir / "images" / split / image_name
                label_path = run_dir / "labels" / split / f"{video_path.stem}_frame{frame_index:07d}.txt"

                result = model.predict(
                    source=frame,
                    conf=CONFIDENCE,
                    imgsz=IMAGE_SIZE,
                    verbose=False
                )[0]

                h, w = frame.shape[:2]
                lines = []
                if result.boxes is not None:
                    for box in result.boxes:
                        original_cls = int(box.cls.item())
                        if original_cls in smoke_ids:
                            yolo_cls = 0
                        elif original_cls in fire_ids:
                            yolo_cls = 1
                        else:
                            continue

                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        box_w = (x2 - x1) / w
                        box_h = (y2 - y1) / h
                        x_center = min(max(x_center, 0.0), 1.0)
                        y_center = min(max(y_center, 0.0), 1.0)
                        box_w = min(max(box_w, 0.0), 1.0)
                        box_h = min(max(box_h, 0.0), 1.0)
                        lines.append(
                            f"{yolo_cls} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
                        )

                # Lưu cả ảnh không phát hiện gì với file nhãn rỗng.
                # Nhãn vẫn là dự đoán tự động, cần kiểm tra trước khi huấn luyện.
                if cv2.imwrite(str(image_path), frame):
                    label_path.write_text("\n".join(lines), encoding="utf-8")
                    total_frames += 1
                    total_detections += len(lines)
                    saved_index += 1

                frame_index += 1
        finally:
            cap.release()

        print(f"  Đã lưu {saved_index} ảnh")

    data = {
        "path": str(run_dir).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "smoke", 1: "fire"},
    }
    (run_dir / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8"
    )

    print("\nHOÀN TẤT")
    print(f"Dataset: {run_dir}")
    print(f"Tổng ảnh đã lưu: {total_frames}")
    print(f"Tổng hộp dự đoán đã lưu: {total_detections}")
    print(f"File cấu hình: {run_dir / 'data.yaml'}")
    print("\nLƯU Ý QUAN TRỌNG:")
    print("- Đây là nhãn AI tự dự đoán, không phải nhãn đã được xác minh.")
    print("- Hãy xem ảnh/nhãn và sửa dự đoán sai trước khi huấn luyện.")
    print("- Nếu thư mục val trống (chỉ có 1 video hoặc video không đọc được), "
          "không chạy train cho đến khi có tập validation hợp lệ.")


if __name__ == "__main__":
    main()
