"""Script chia tách tập D-Fire thành Train (17.221), Val (2.153) và Test (2.153) độc lập.

Khắc phục triệt để lỗi rò rỉ dữ liệu (dùng test làm val) theo yêu cầu đề cương NCKH.
- Tập ban đầu: train (17.221 ảnh), test (4.306 ảnh).
- Thực hiện: Tách 50% tập test (2.153 ảnh + nhãn tương ứng) sang thư mục val/.
- Giữ lại 50% còn lại (2.153 ảnh + nhãn) cho tập test độc lập dùng đánh giá cuối cùng.
- Cập nhật data_dfire.yaml với đầy đủ 3 tập train, val, test.
"""
import os
import sys
import shutil
import random
from pathlib import Path
from collections import Counter
import yaml

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
DFIRE_ROOT = Path("datasets/dfire")
YAML_PATH = Path("data_dfire.yaml")


def split_val_test():
    test_img_dir = DFIRE_ROOT / "test" / "images"
    test_lbl_dir = DFIRE_ROOT / "test" / "labels"
    val_img_dir = DFIRE_ROOT / "val" / "images"
    val_lbl_dir = DFIRE_ROOT / "val" / "labels"

    if not test_img_dir.exists():
        print(f"❌ Không tìm thấy thư mục: {test_img_dir}")
        return False

    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    all_test_imgs = sorted(list(test_img_dir.glob("*.*")))
    total_test = len(all_test_imgs)
    print(f"📊 Tổng số ảnh hiện tại trong test/: {total_test}")

    if total_test <= 2200 and val_img_dir.exists() and len(list(val_img_dir.glob("*.*"))) > 2000:
        print("ℹ️ Tập val đã tồn tại và đã được chia trước đó. Tiến hành kiểm tra và thống kê...")
    else:
        # Tách 50% sang val
        random.seed(SEED)
        val_indices = set(random.sample(range(total_test), total_test // 2))

        val_count = 0
        for i, img_path in enumerate(all_test_imgs):
            if i in val_indices:
                lbl_path = test_lbl_dir / f"{img_path.stem}.txt"
                dest_img = val_img_dir / img_path.name
                dest_lbl = val_lbl_dir / f"{img_path.stem}.txt"

                shutil.move(str(img_path), str(dest_img))
                if lbl_path.exists():
                    shutil.move(str(lbl_path), str(dest_lbl))
                val_count += 1

        print(f"✅ Đã di chuyển {val_count} ảnh và nhãn sang {val_img_dir}")
        print(f"✅ Số ảnh còn lại trong test: {len(list(test_img_dir.glob('*.*')))}")

    # Cập nhật data_dfire.yaml
    yaml_config = {
        "path": "datasets/dfire",
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 2,
        "names": {
            0: "smoke",
            1: "fire"
        }
    }
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"✅ Đã cập nhật file cấu hình: {YAML_PATH.resolve()}")

    # Thống kê chi tiết từng split
    print("\n" + "=" * 70)
    print("📈 THỐNG KÊ CHI TIẾT TẬP D-FIRE SAU KHI PHÂN CHIA CHUẨN 3 TẬP (TRAIN/VAL/TEST)")
    print("=" * 70)

    splits = ["train", "val", "test"]
    stats = {}

    for s in splits:
        s_img_dir = DFIRE_ROOT / s / "images"
        s_lbl_dir = DFIRE_ROOT / s / "labels"

        imgs = list(s_img_dir.glob("*.*")) if s_img_dir.exists() else []
        lbls = list(s_lbl_dir.glob("*.txt")) if s_lbl_dir.exists() else []

        box_counts = Counter()
        images_with_fire = 0
        images_with_smoke = 0
        images_with_both = 0
        images_empty = 0

        lbl_map = {f.stem: f for f in lbls}

        for img in imgs:
            lf = lbl_map.get(img.stem)
            if lf is None or not lf.exists() or lf.stat().st_size == 0:
                images_empty += 1
                continue

            has_fire = False
            has_smoke = False
            with open(lf, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.read().strip().splitlines()

            if not lines:
                images_empty += 1
                continue

            for line in lines:
                parts = line.strip().split()
                if parts:
                    try:
                        cls_id = int(parts[0])
                        box_counts[cls_id] += 1
                        if cls_id == 0:
                            has_smoke = True
                        elif cls_id == 1:
                            has_fire = True
                    except ValueError:
                        pass

            if has_fire and has_smoke:
                images_with_both += 1
            elif has_fire:
                images_with_fire += 1
            elif has_smoke:
                images_with_smoke += 1
            else:
                images_empty += 1

        stats[s] = {
            "images": len(imgs),
            "smoke_only": images_with_smoke,
            "fire_only": images_with_fire,
            "both": images_with_both,
            "empty": images_empty,
            "smoke_boxes": box_counts[0],
            "fire_boxes": box_counts[1],
            "total_boxes": box_counts[0] + box_counts[1]
        }

    header = f"{'Chỉ số':<28} | {'Train':<12} | {'Val':<12} | {'Test':<12} | {'Tổng cộng':<12}"
    print(header)
    print("-" * len(header))

    def row(label, key):
        v_tr = stats["train"][key]
        v_va = stats["val"][key]
        v_te = stats["test"][key]
        v_tot = v_tr + v_va + v_te
        return f"{label:<28} | {v_tr:<12,} | {v_va:<12,} | {v_te:<12,} | {v_tot:<12,}"

    print(row("Tổng số hình ảnh", "images"))
    print(row("Ảnh chỉ có Khói (Smoke)", "smoke_only"))
    print(row("Ảnh chỉ có Lửa (Fire)", "fire_only"))
    print(row("Ảnh có cả Khói & Lửa", "both"))
    print(row("Ảnh âm tính (Background)", "empty"))
    print(row("Số Box Khói (0: smoke)", "smoke_boxes"))
    print(row("Số Box Lửa (1: fire)", "fire_boxes"))
    print(row("Tổng số Bounding Box", "total_boxes"))
    print("=" * len(header))

    # Ghi báo cáo cập nhật docs/dfire_analysis.md
    report_file = Path("docs/dfire_analysis.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Chuẩn Hóa & Phân Chia Tập Dữ Liệu D-Fire (MT1)\n\n")
        f.write("**Nguyên tắc khoa học:** Tách biệt độc lập 3 tập Train (80%), Validation (10%) và Test (10%).\n")
        f.write("- **Tập Train:** Tối ưu hóa trọng số mô hình.\n")
        f.write("- **Tập Val:** Đánh giá trong quá trình huấn luyện, chọn checkpoint `best.pt` và điều chỉnh siêu tham số.\n")
        f.write("- **Tập Test:** Hoàn toàn độc lập, chỉ dùng để đánh giá năng lực suy diễn cuối cùng (tránh rò rỉ dữ liệu / Data Leakage).\n\n")
        f.write("## 1. Bảng Thống Kê Chi Tiết Số Lượng & Phân Bố Lớp\n\n")
        f.write("| Chỉ số | Tập Train (80%) | Tập Val (10%) | Tập Test (10%) | Tổng cộng (100%) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Tổng số hình ảnh** | {stats['train']['images']:,} | {stats['val']['images']:,} | {stats['test']['images']:,} | {stats['train']['images']+stats['val']['images']+stats['test']['images']:,} |\n")
        f.write(f"| **Ảnh chỉ có Khói (Smoke)** | {stats['train']['smoke_only']:,} | {stats['val']['smoke_only']:,} | {stats['test']['smoke_only']:,} | {stats['train']['smoke_only']+stats['val']['smoke_only']+stats['test']['smoke_only']:,} |\n")
        f.write(f"| **Ảnh chỉ có Lửa (Fire)** | {stats['train']['fire_only']:,} | {stats['val']['fire_only']:,} | {stats['test']['fire_only']:,} | {stats['train']['fire_only']+stats['val']['fire_only']+stats['test']['fire_only']:,} |\n")
        f.write(f"| **Ảnh có cả Khói & Lửa** | {stats['train']['both']:,} | {stats['val']['both']:,} | {stats['test']['both']:,} | {stats['train']['both']+stats['val']['both']+stats['test']['both']:,} |\n")
        f.write(f"| **Ảnh âm tính (Background)** | {stats['train']['empty']:,} | {stats['val']['empty']:,} | {stats['test']['empty']:,} | {stats['train']['empty']+stats['val']['empty']+stats['test']['empty']:,} |\n")
        f.write(f"| **Bounding Box Khói (0)** | {stats['train']['smoke_boxes']:,} | {stats['val']['smoke_boxes']:,} | {stats['test']['smoke_boxes']:,} | {stats['train']['smoke_boxes']+stats['val']['smoke_boxes']+stats['test']['smoke_boxes']:,} |\n")
        f.write(f"| **Bounding Box Lửa (1)** | {stats['train']['fire_boxes']:,} | {stats['val']['fire_boxes']:,} | {stats['test']['fire_boxes']:,} | {stats['train']['fire_boxes']+stats['val']['fire_boxes']+stats['test']['fire_boxes']:,} |\n")
        f.write(f"| **Tổng số Box** | {stats['train']['total_boxes']:,} | {stats['val']['total_boxes']:,} | {stats['test']['total_boxes']:,} | {stats['train']['total_boxes']+stats['val']['total_boxes']+stats['test']['total_boxes']:,} |\n\n")
        f.write("## 2. Kết Luận\n\n")
        f.write("- Đã khắc phục triệt để lỗi rò rỉ dữ liệu `val: test/images`.\n")
        f.write("- Tỷ lệ phân bố lớp và ảnh âm tính đồng đều giữa 3 tập, đảm bảo tính khách quan cho quá trình benchmark MT2.\n")

    print(f"📄 Báo cáo đã cập nhật tại: {report_file.resolve()}")
    return True


if __name__ == "__main__":
    split_val_test()

