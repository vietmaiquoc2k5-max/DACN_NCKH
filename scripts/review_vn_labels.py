"""Công cụ kiểm duyệt & chỉnh sửa nhãn ảnh Việt Nam trực quan (Interactive Label Reviewer).

Mục đích:
- Thực hiện nghiệm thu và tinh chỉnh Ground Truth thủ công từ nhãn do AI gợi ý (Phương án B).
- Đáp ứng điều kiện tiên quyết của Đề cương NCKH: Mọi nhãn dùng làm dữ liệu kiểm định/thích nghi
  phải được con người kiểm duyệt và ghi nhận trong review_status.csv.

Phím tắt:
  • [Space] / [Enter] / [N] : Chấp nhận nhãn hợp lệ & chuyển sang ảnh kế tiếp
  • [B] / [P]               : Quay lại ảnh trước
  • [0]                     : Chọn lớp Smoke (Khói) để vẽ box mới bằng chuột
  • [1]                     : Chọn lớp Fire (Lửa) để vẽ box mới bằng chuột
  • [D]                     : Xóa box vừa vẽ gần nhất
  • [C]                     : Xóa sạch toàn bộ box trên ảnh (chuyển thành background rỗng)
  • [Q] / [ESC]             : Lưu tiến độ kiểm duyệt và thoát
"""
import sys
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

DATASET_DIR = Path("datasets/vn_dataset")
REVIEW_LOG_PATH = DATASET_DIR / "review_status.csv"

COLOR_SMOKE = (255, 191, 0)   # Cyan / Light Blue (BGR)
COLOR_FIRE  = (0, 69, 255)    # Orange / Red (BGR)
CLASS_NAMES = {0: "smoke", 1: "fire"}

# Biến toàn cục hỗ trợ vẽ chuột
drawing = False
ix, iy = -1, -1
cur_draw_box = None
active_class = 1  # Mặc định chọn vẽ Lửa (1)


def parse_labels(label_file: Path, w: int, h: int):
    """Đọc file nhãn YOLO và chuyển sang pixel (cls_id, x1, y1, x2, y2)."""
    boxes = []
    if not label_file.exists():
        return boxes

    lines = label_file.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            boxes.append([cls_id, x1, y1, x2, y2])
    return boxes


def save_labels(label_file: Path, boxes, w: int, h: int):
    """Lưu lại bounding box pixel sang format chuẩn hóa YOLO."""
    lines = []
    for cls_id, x1, y1, x2, y2 in boxes:
        # Giới hạn trong kích thước ảnh
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        if bw <= 2 or bh <= 2:
            continue
        xc = (x1 + bw / 2) / w
        yc = (y1 + bh / 2) / h
        bwn = bw / w
        bhn = bh / h
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bwn:.6f} {bhn:.6f}")
    label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, cur_draw_box
    boxes = param["boxes"]

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        cur_draw_box = (ix, iy, x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cur_draw_box = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            cur_draw_box = None
            x1, y1 = min(ix, x), min(iy, y)
            x2, y2 = max(ix, x), max(iy, y)
            if (x2 - x1) > 8 and (y2 - y1) > 8:
                boxes.append([active_class, x1, y1, x2, y2])
                param["modified"] = True


def review_labels(split_filter=None):
    global active_class, cur_draw_box

    if not REVIEW_LOG_PATH.exists():
        print(f"❌ Không tìm thấy {REVIEW_LOG_PATH}. Hãy chạy scripts/generate_vn_pseudo_labels.py trước.")
        return

    df = pd.read_csv(REVIEW_LOG_PATH)
    if split_filter:
        df_indices = df[df["split"] == split_filter].index.tolist()
    else:
        df_indices = df.index.tolist()

    if not df_indices:
        print("ℹ️ Không tìm thấy ảnh nào cần duyệt theo điều kiện lọc.")
        return

    win_name = "Kiem Duyet Nhan Anh Viet Nam (Ground Truth Editor)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1024, 680)

    curr_idx = 0
    total = len(df_indices)

    print("\n" + "=" * 75)
    print("🎯 BẮT ĐẦU KIỂM DUYỆT & CHỈNH SỬA NHÃN THỦ CÔNG")
    print("=" * 75)
    print("• Kéo chuột trái để VẼ BOX MỚI.")
    print("• Nhấn [0] để đổi lớp thành Khói, [1] để đổi thành Lửa.")
    print("• Nhấn [D] để xóa box vừa vẽ, [C] để xóa sạch box (ảnh background).")
    print("• Nhấn [Space] / [Enter] để CHẤP NHẬN và sang ảnh kế tiếp.")
    print("• Nhấn [Q] / [ESC] để lưu và thoát.")
    print("=" * 75 + "\n")

    while 0 <= curr_idx < total:
        row_id = df_indices[curr_idx]
        row = df.loc[row_id]

        img_path = DATASET_DIR / "images" / row["split"] / row["image_name"]
        lbl_path = DATASET_DIR / row["label_file"]

        img = cv2.imread(str(img_path))
        if img is None:
            curr_idx += 1
            continue

        h, w = img.shape[:2]
        boxes = parse_labels(lbl_path, w, h)
        modified = False

        param = {"boxes": boxes, "modified": modified}
        cv2.setMouseCallback(win_name, mouse_callback, param)

        while True:
            display = img.copy()

            # Vẽ các box hiện có
            for i, (cls_id, x1, y1, x2, y2) in enumerate(param["boxes"], start=1):
                color = COLOR_FIRE if cls_id == 1 else COLOR_SMOKE
                cls_str = "FIRE" if cls_id == 1 else "SMOKE"
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                tag = f"#{i} {cls_str}"
                cv2.putText(display, tag, (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # Vẽ box đang kéo chuột
            if cur_draw_box is not None:
                cx1, cy1, cx2, cy2 = cur_draw_box
                cur_color = COLOR_FIRE if active_class == 1 else COLOR_SMOKE
                cv2.rectangle(display, (cx1, cy1), (cx2, cy2), cur_color, 1)

            # Thanh trạng thái bên dưới
            overlay = display.copy()
            cv2.rectangle(overlay, (0, h - 60), (w, h), (20, 20, 20), -1)
            display = cv2.addWeighted(overlay, 0.8, display, 0.2, 0)

            active_str = "FIRE [1]" if active_class == 1 else "SMOKE [0]"
            info_1 = f"[{curr_idx + 1}/{total}] {row['image_name']} | Split: {row['split'].upper()} | Boxes: {len(param['boxes'])} | Ve lop: {active_str}"
            info_2 = "[Space/Enter]: Chap nhan | [0/1]: Doi lop | [D]: Xoa box | [C]: Xoa het | [B]: Quay lai | [Q]: Thoat"

            cv2.putText(display, info_1, (12, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)
            cv2.putText(display, info_2, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

            cv2.imshow(win_name, display)
            key = cv2.waitKey(20) & 0xFF

            if key in [13, 32, ord('n'), ord('N')]:  # Enter, Space, N
                if param["modified"]:
                    save_labels(lbl_path, param["boxes"], w, h)
                    df.loc[row_id, "modified"] = True
                df.loc[row_id, "is_reviewed"] = True
                df.loc[row_id, "num_boxes"] = len(param["boxes"])
                curr_idx += 1
                break

            elif key in [ord('b'), ord('B'), ord('p'), ord('P')]:  # Back
                if param["modified"]:
                    save_labels(lbl_path, param["boxes"], w, h)
                    df.loc[row_id, "modified"] = True
                curr_idx = max(0, curr_idx - 1)
                break

            elif key == ord('0'):
                active_class = 0
            elif key == ord('1'):
                active_class = 1
            elif key in [ord('d'), ord('D')]:  # Xóa box gần nhất
                if param["boxes"]:
                    param["boxes"].pop()
                    param["modified"] = True
            elif key in [ord('c'), ord('C')]:  # Xóa sạch
                param["boxes"].clear()
                param["modified"] = True
            elif key in [27, ord('q'), ord('Q')]:  # ESC or Q
                if param["modified"]:
                    save_labels(lbl_path, param["boxes"], w, h)
                    df.loc[row_id, "modified"] = True
                df.loc[row_id, "is_reviewed"] = True
                df.to_csv(REVIEW_LOG_PATH, index=False, encoding="utf-8-sig")
                cv2.destroyAllWindows()
                print(f"\n💾 Đã lưu tiến độ kiểm duyệt vào: {REVIEW_LOG_PATH.resolve()}")
                return

    df.to_csv(REVIEW_LOG_PATH, index=False, encoding="utf-8-sig")
    cv2.destroyAllWindows()
    print(f"\n🎉 HOÀN TẤT KIỂM DUYỆT TOÀN BỘ ẢNH! Báo cáo lưu tại: {REVIEW_LOG_PATH.resolve()}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kiểm duyệt và chỉnh sửa nhãn ảnh Việt Nam")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None, help="Chỉ duyệt ảnh của tập cụ thể")
    args = parser.parse_args()

    review_labels(split_filter=args.split)
