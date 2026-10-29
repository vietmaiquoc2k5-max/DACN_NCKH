"""Script huấn luyện Baseline B2: MobileNetV3-Small trên tập dữ liệu D-Fire (MT2).

Huấn luyện mạng phân loại nhẹ cấp độ khung hình (Frame-level multi-label classification)
để phát hiện sự hiện diện của Khói (smoke) và Lửa (fire).
"""
import sys
import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from baselines.baseline_b2_mobilenet import (
    MobileNetV3FireClassifier,
    DFireClassificationDataset
)


def train_baseline_b2(
    dfire_dir: str = "datasets/dfire",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    device_str: str = "cpu",
    output_model: str = "models/mobilenetv3_b2.pt",
    seed: int = 42
):
    print("=" * 65)
    print("🔥 HUẤN LUYỆN BASELINE B2: MOBILENETV3-SMALL TRÊN D-FIRE (MT2) 🔥")
    print("=" * 65)

    torch.manual_seed(seed)
    if torch.cuda.is_available() and device_str.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        device = torch.device(device_str)
        print(f"🍀 Sử dụng GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("🍀 Sử dụng CPU cho huấn luyện.")

    dfire_path = Path(dfire_dir)
    train_img_dir = dfire_path / "train" / "images"
    train_lbl_dir = dfire_path / "train" / "labels"
    val_img_dir = dfire_path / "test" / "images"
    val_lbl_dir = dfire_path / "test" / "labels"

    if not train_img_dir.exists() or not val_img_dir.exists():
        print(f"❌ Không tìm thấy dữ liệu D-Fire tại {dfire_dir}")
        return False

    # Pipeline tăng cường dữ liệu cho tập huấn luyện
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("📦 Đang nạp dataset...")
    train_dataset = DFireClassificationDataset(train_img_dir, train_lbl_dir, transform=train_transform)
    val_dataset = DFireClassificationDataset(val_img_dir, val_lbl_dir, transform=val_transform)

    print(f"  • Số lượng ảnh Train: {len(train_dataset):,} ảnh")
    print(f"  • Số lượng ảnh Validation/Test: {len(val_dataset):,} ảnh")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if sys.platform != "win32" else 0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if sys.platform != "win32" else 0,
        pin_memory=torch.cuda.is_available()
    )

    # Khởi tạo mô hình
    print("⏳ Nạp mô hình MobileNetV3-Small (pretrained ImageNet)...")
    model = MobileNetV3FireClassifier(num_classes=2, pretrained=True).to(device)

    # Tiêu chuẩn mất mát: BCEWithLogitsLoss cho multi-label
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    out_path = Path(output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Bắt đầu quá trình huấn luyện {epochs} epochs...")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = running_loss / max(1, n_batches)

        # Đánh giá trên tập val
        model.eval()
        val_loss = 0.0
        val_batches = 0
        tp_smoke, fp_smoke, fn_smoke = 0, 0, 0
        tp_fire, fp_fire, fn_fire = 0, 0, 0

        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                v_loss = criterion(outputs, targets)
                val_loss += v_loss.item()
                val_batches += 1

                preds = (torch.sigmoid(outputs) >= 0.5).float()

                # Thống kê Smoke (class 0)
                tp_smoke += int(((preds[:, 0] == 1) & (targets[:, 0] == 1)).sum().item())
                fp_smoke += int(((preds[:, 0] == 1) & (targets[:, 0] == 0)).sum().item())
                fn_smoke += int(((preds[:, 0] == 0) & (targets[:, 0] == 1)).sum().item())

                # Thống kê Fire (class 1)
                tp_fire += int(((preds[:, 1] == 1) & (targets[:, 1] == 1)).sum().item())
                fp_fire += int(((preds[:, 1] == 1) & (targets[:, 1] == 0)).sum().item())
                fn_fire += int(((preds[:, 1] == 0) & (targets[:, 1] == 1)).sum().item())

        val_loss = val_loss / max(1, val_batches)

        prec_s = tp_smoke / max(1, tp_smoke + fp_smoke)
        rec_s = tp_smoke / max(1, tp_smoke + fn_smoke)
        f1_s = 2 * prec_s * rec_s / max(1e-6, prec_s + rec_s)

        prec_f = tp_fire / max(1, tp_fire + fp_fire)
        rec_f = tp_fire / max(1, tp_fire + fn_fire)
        f1_f = 2 * prec_f * rec_f / max(1e-6, prec_f + rec_f)

        macro_f1 = (f1_s + f1_f) / 2.0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Loss: Train={train_loss:.4f}, Val={val_loss:.4f} | "
            f"Smoke F1={f1_s:.3f} | Fire F1={f1_f:.3f} | Mean F1={macro_f1:.3f}"
        )

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_macro_f1": best_val_f1,
                "smoke_metrics": {"prec": prec_s, "rec": rec_s, "f1": f1_s},
                "fire_metrics": {"prec": prec_f, "rec": rec_f, "f1": f1_f}
            }, str(out_path))
            print(f"  ⭐ Đã lưu checkpoint mới tốt nhất: {out_path} (F1: {best_val_f1:.3f})")

    elapsed_min = (time.time() - t_start) / 60.0
    print("\n" + "=" * 65)
    print(f"🎉 HOÀN THÀNH HUẤN LUYỆN BASELINE B2 ({elapsed_min:.1f} phút)!")
    print(f"• Trọng số tối ưu được lưu tại: {out_path}")
    print(f"• Best Macro F1: {best_val_f1:.4f}")
    print("=" * 65)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Baseline B2 MobileNetV3 trên D-Fire")
    parser.add_argument("--dfire", default="datasets/dfire", help="Thư mục dataset D-Fire")
    parser.add_argument("--epochs", type=int, default=10, help="Số epochs huấn luyện")
    parser.add_argument("--batch", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Thiết bị ('cpu' hoặc 'cuda:0')")
    parser.add_argument("--output", default="models/mobilenetv3_b2.pt", help="File lưu checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_baseline_b2(
        dfire_dir=args.dfire,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device_str=args.device,
        output_model=args.output,
        seed=args.seed
    )
