"""Tạo tập dữ liệu Replay kết hợp D-Fire và dữ liệu camera Việt Nam (MT4).

Đề tài NCKH: Phát hiện & cảnh báo cháy từ camera giám sát.
Mục tiêu MT4:
- Trích xuất ngẫu nhiên 20% dữ liệu từ tập Train của D-Fire.
- Trộn 20% D-Fire với 100% tập Train của Việt Nam (`datasets/vn_dataset`).
- Tạo file cấu hình YAML mới (`vn_replay_20.yaml`) phục vụ Chiến lược 3 (Đóng băng backbone + Replay).

Phương pháp lưu trữ:
- Để tối ưu dung lượng đĩa và tránh sao chép tốn kém hàng nghìn file ảnh, script tạo danh sách
  đường dẫn ảnh chuẩn (image path list txt) mà Ultralytics YOLO hỗ trợ tự nhiên.
- Đảm bảo tính tất định (deterministic) 100% thông qua Random Seed.
"""
import os
import sys
import json
import random
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import yaml

# Tự động xác định thư mục gốc dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_image_label_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """Lấy danh sách các cặp (ảnh, nhãn) hợp lệ với tốc độ cao bằng os.scandir."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    pairs = []
    if not images_dir.exists():
        raise FileNotFoundError(f"Thư mục ảnh không tồn tại: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Thư mục nhãn không tồn tại: {labels_dir}")

    # Quét trước nhãn thành tập hợp stem O(1)
    label_stems = set()
    with os.scandir(labels_dir) as it:
        for entry in it:
            if entry.name.endswith(".txt"):
                label_stems.add(entry.name[:-4])

    # Quét ảnh và đối chiếu
    with os.scandir(images_dir) as it:
        for entry in it:
            dot_idx = entry.name.rfind(".")
            if dot_idx != -1:
                ext = entry.name[dot_idx:].lower()
                stem = entry.name[:dot_idx]
                if ext in valid_exts and stem in label_stems:
                    pairs.append((images_dir / entry.name, labels_dir / f"{stem}.txt"))

    # Sắp xếp để đảm bảo tính tất định tuyệt đối
    pairs.sort(key=lambda p: p[0].name)
    return pairs


def sample_dfire_train(
    dfire_root: Path,
    ratio: float = 0.20,
    seed: int = 42
) -> List[Path]:
    """Trích xuất ngẫu nhiên tỷ lệ ratio ảnh từ tập Train của D-Fire."""
    train_img_dir = dfire_root / "train" / "images"
    train_lbl_dir = dfire_root / "train" / "labels"
    
    pairs = get_image_label_pairs(train_img_dir, train_lbl_dir)
    total_dfire = len(pairs)
    sample_count = int(round(total_dfire * ratio))

    rng = random.Random(seed)
    sampled_pairs = rng.sample(pairs, sample_count)
    # Sắp xếp lại để danh sách nhất quán
    sampled_images = sorted([p[0] for p in sampled_pairs])
    return sampled_images


def get_all_vn_train(vn_root: Path) -> List[Path]:
    """Lấy 100% ảnh tập Train của bộ dữ liệu Việt Nam."""
    train_img_dir = vn_root / "images" / "train"
    train_lbl_dir = vn_root / "labels" / "train"

    pairs = get_image_label_pairs(train_img_dir, train_lbl_dir)
    return sorted([p[0] for p in pairs])


def create_replay_dataset(
    dfire_root: Path = PROJECT_ROOT / "datasets" / "dfire",
    vn_root: Path = PROJECT_ROOT / "datasets" / "vn_dataset",
    output_txt: Path = PROJECT_ROOT / "data" / "vn_replay_20_train.txt",
    output_yaml: Path = PROJECT_ROOT / "vn_replay_20.yaml",
    manifest_path: Path = PROJECT_ROOT / "data" / "vn_replay_20_manifest.json",
    dfire_ratio: float = 0.20,
    seed: int = 42
) -> Dict[str, Any]:
    """Thực hiện tạo tập dữ liệu Replay 20% D-Fire + 100% VN Train."""
    print("=" * 65)
    print(f"📦 BẮT ĐẦU TẠO TẬP DỮ LIỆU REPLAY (MT4)")
    print(f"  • Thư mục D-Fire: {dfire_root}")
    print(f"  • Thư mục VN Dataset: {vn_root}")
    print(f"  • Tỷ lệ Replay D-Fire: {dfire_ratio * 100:.1f}% | Seed: {seed}")
    print("=" * 65)

    sampled_dfire_imgs = sample_dfire_train(dfire_root, ratio=dfire_ratio, seed=seed)
    all_vn_imgs = get_all_vn_train(vn_root)

    print(f"✅ Đã chọn {len(sampled_dfire_imgs)} ảnh từ D-Fire Train ({dfire_ratio*100:.1f}%)")
    print(f"✅ Đã lấy {len(all_vn_imgs)} ảnh từ VN Train (100%)")

    # Kết hợp toàn bộ
    combined_images = all_vn_imgs + sampled_dfire_imgs
    # Đảo ngẫu nhiên danh sách để khi batching không bị dồn một domain
    rng = random.Random(seed)
    rng.shuffle(combined_images)

    total_combined = len(combined_images)
    print(f"🎯 Tổng số ảnh tập huấn luyện kết hợp: {total_combined} ảnh")

    # Ghi file text danh sách đường dẫn
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        for img in combined_images:
            # Ghi đường dẫn tương đối so với PROJECT_ROOT (dùng dấu / chuẩn POSIX)
            try:
                rel_path = img.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                rel_path = img.as_posix()
            f.write(f"{rel_path}\n")

    # Tính SHA-256 file text
    sha256 = hashlib.sha256(output_txt.read_bytes()).hexdigest()
    print(f"💾 Đã lưu danh sách ảnh: {output_txt.resolve()}")
    print(f"   SHA-256: {sha256}")

    # Tạo file YAML cấu hình dùng đường dẫn tương đối để đảm bảo tính khả chuyển (portable)
    try:
        rel_txt = output_txt.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel_txt = output_txt.as_posix()
    yaml_content = {
        "path": ".",
        "train": rel_txt,
        "val": "datasets/vn_dataset/images/val",
        "test": "datasets/vn_dataset/images/test",
        "nc": 2,
        "names": {
            0: "smoke",
            1: "fire"
        }
    }

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False, allow_unicode=True)

    print(f"📄 Đã tạo file cấu hình YAML: {output_yaml.resolve()}")

    # Tạo manifest kiểm chứng
    manifest = {
        "dataset_name": "vn_replay_20",
        "seed": seed,
        "dfire_ratio": dfire_ratio,
        "num_vn_train": len(all_vn_imgs),
        "num_dfire_sampled": len(sampled_dfire_imgs),
        "total_train_images": total_combined,
        "train_list_txt": rel_txt,
        "train_list_sha256": sha256,
        "yaml_config": output_yaml.name,
        "val_set": "datasets/vn_dataset/images/val",
        "test_set": "datasets/vn_dataset/images/test",
        "classes": {0: "smoke", 1: "fire"}
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"📑 Đã lưu manifest xác thực: {manifest_path.resolve()}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Tạo tập dữ liệu Replay 20% D-Fire + 100% VN Train (MT4)")
    parser.add_argument("--dfire-dir", type=str, default="datasets/dfire", help="Đường dẫn thư mục D-Fire")
    parser.add_argument("--vn-dir", type=str, default="datasets/vn_dataset", help="Đường dẫn thư mục VN Dataset")
    parser.add_argument("--ratio", type=float, default=0.20, help="Tỷ lệ lấy mẫu từ D-Fire Train (mặc định 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed cố định (mặc định 42)")
    parser.add_argument("--output-txt", type=str, default="data/vn_replay_20_train.txt", help="Đường dẫn file txt danh sách ảnh")
    parser.add_argument("--output-yaml", type=str, default="vn_replay_20.yaml", help="Đường dẫn file YAML đầu ra")
    parser.add_argument("--manifest", type=str, default="data/vn_replay_20_manifest.json", help="Đường dẫn file manifest JSON")

    args = parser.parse_args()

    dfire_path = Path(args.dfire_dir)
    if not dfire_path.is_absolute():
        dfire_path = PROJECT_ROOT / dfire_path

    vn_path = Path(args.vn_dir)
    if not vn_path.is_absolute():
        vn_path = PROJECT_ROOT / vn_path

    output_txt = Path(args.output_txt)
    if not output_txt.is_absolute():
        output_txt = PROJECT_ROOT / output_txt

    output_yaml = Path(args.output_yaml)
    if not output_yaml.is_absolute():
        output_yaml = PROJECT_ROOT / output_yaml

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    create_replay_dataset(
        dfire_root=dfire_path,
        vn_root=vn_path,
        output_txt=output_txt,
        output_yaml=output_yaml,
        manifest_path=manifest_path,
        dfire_ratio=args.ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

