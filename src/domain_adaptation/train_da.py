"""Huấn luyện Thích nghi miền (Domain Adaptation) cho YOLOv8 (MT4).

Đề tài NCKH: Phát hiện & cảnh báo cháy từ camera giám sát.
Mục tiêu MT4:
- Thực hiện 3 chiến lược Domain Adaptation từ trọng số ban đầu `best.pt`:
  1. Chiến lược 1 (Toàn bộ / Full): Fine-tune toàn bộ trọng số mạng trên tập `vn_dataset`.
  2. Chiến lược 2 (Đóng băng backbone / Freeze): Đóng băng 10 layer backbone (freeze=10),
     chỉ fine-tune neck và head trên tập `vn_dataset`.
  3. Chiến lược 3 (Đóng băng + Replay 20% / Freeze + Replay): Đóng băng 10 layer backbone (freeze=10),
     huấn luyện trên tập dữ liệu kết hợp `vn_replay_20.yaml`.

Ràng buộc nghiên cứu:
- Kích thước ảnh chuẩn hóa: imgsz = 640.
- Cố định Random Seed để đảm bảo tính tái lập (reproducibility).
- Augmentation vừa phải (hsv, scale, translate, fliplr, mosaic).
- Tuyệt đối không cài đặt kiến trúc Hope/CMS (chỉ dùng lý thuyết trong báo cáo).
"""
import os
import sys
import json
import time
import shutil
import argparse
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

# Tối ưu matplotlib trên Windows
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import yaml
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Định nghĩa 3 chiến lược theo đề cương
STRATEGIES_CONFIG = {
    "full": {
        "name": "Strategy 1: Full Fine-tuning",
        "desc": "Fine-tune toàn bộ trọng số mạng trên dữ liệu camera Việt Nam",
        "data_yaml": "data_vn.yaml",
        "freeze": None,
        "output_model": "da_full.pt"
    },
    "freeze": {
        "name": "Strategy 2: Freeze Backbone",
        "desc": "Đóng băng 10 layer backbone, chỉ fine-tune neck và head trên dữ liệu Việt Nam",
        "data_yaml": "data_vn.yaml",
        "freeze": 10,
        "output_model": "da_freeze.pt"
    },
    "replay": {
        "name": "Strategy 3: Freeze Backbone + Replay 20% D-Fire",
        "desc": "Đóng băng backbone + trộn 20% D-Fire chống quên thảm khốc (Catastrophic Forgetting)",
        "data_yaml": "vn_replay_20.yaml",
        "freeze": 10,
        "output_model": "da_replay.pt"
    }
}


def verify_freeze_layers(model: YOLO, freeze_count: Optional[int]) -> Dict[str, Any]:
    """Kiểm tra và báo cáo trạng thái đóng băng layer của mô hình."""
    total_params = sum(p.numel() for p in model.model.parameters())
    if freeze_count is None or freeze_count <= 0:
        return {
            "freeze_count": 0,
            "frozen_param_count": 0,
            "trainable_param_count": total_params,
            "total_param_count": total_params,
            "frozen_ratio": 0.0
        }

    # Trong kiến trúc YOLOv8/11, model.model là nn.Sequential chứa các layers
    # Ultralytics freeze=N đóng băng model.0 đến model.(N-1)
    frozen_params = 0
    trainable_params = 0
    running_total = 0

    freeze_prefixes = [f"model.{x}." for x in range(freeze_count)]
    for name, param in model.model.named_parameters():
        running_total += param.numel()
        if any(name.startswith(pfx) for pfx in freeze_prefixes):
            frozen_params += param.numel()
        else:
            trainable_params += param.numel()

    return {
        "freeze_count": freeze_count,
        "frozen_param_count": frozen_params,
        "trainable_param_count": trainable_params,
        "total_param_count": total_params,
        "frozen_ratio": frozen_params / total_params if total_params > 0 else 0.0
    }


def train_single_strategy(
    strategy_key: str,
    base_weights: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 8,
    imgsz: int = 640,
    seed: int = 42,
    device: str = "auto",
    dry_run: bool = False
) -> Dict[str, Any]:
    """Huấn luyện một chiến lược Domain Adaptation cụ thể."""
    if strategy_key not in STRATEGIES_CONFIG:
        raise ValueError(f"Chiến lược không hợp lệ: {strategy_key}. Chọn một trong: {list(STRATEGIES_CONFIG.keys())}")

    cfg = STRATEGIES_CONFIG[strategy_key]
    data_yaml_path = PROJECT_ROOT / cfg["data_yaml"]
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình dữ liệu: {data_yaml_path}")

    # Tự động chọn device
    if device == "auto":
        resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    print("\n" + "=" * 70)
    print(f"🔥 CHIẾN LƯỢC: {cfg['name']}")
    print(f"   Mô tả: {cfg['desc']}")
    print(f"   Dữ liệu: {cfg['data_yaml']}")
    print(f"   Freeze backbone: {cfg['freeze']} layers")
    print(f"   Trọng số gốc: {base_weights}")
    print(f"   Epochs: {epochs} | Batch: {batch_size} | Imgsz: {imgsz} | Seed: {seed}")
    print(f"   Thiết bị: {resolved_device}")
    print("=" * 70)

    # Nạp mô hình gốc
    model = YOLO(str(base_weights))

    # Kiểm tra phân bố layer đóng băng
    freeze_info = verify_freeze_layers(model, cfg["freeze"])
    print(f"📊 Thông số Freeze Backbone:")
    print(f"   • Tổng số tham số: {freeze_info['total_param_count']:,}")
    print(f"   • Tham số đóng băng: {freeze_info['frozen_param_count']:,} ({freeze_info.get('frozen_ratio', 0.0)*100:.1f}%)")
    print(f"   • Tham số huấn luyện: {freeze_info['trainable_param_count']:,}")

    run_name = f"da_{strategy_key}_seed{seed}"
    run_save_dir = output_dir / run_name

    if dry_run:
        print("💡 [DRY-RUN]: Kiểm tra cấu hình và mô hình hoàn tất thành công. Không thực thi epochs.")
        mock_output = output_dir / cfg["output_model"]
        return {
            "strategy": strategy_key,
            "status": "dry_run_success",
            "freeze_info": freeze_info,
            "data_yaml": str(data_yaml_path),
            "epochs": epochs,
            "seed": seed,
            "target_model_name": cfg["output_model"]
        }

    # Bắt đầu huấn luyện thực tế
    t0 = time.time()
    train_args = {
        "data": str(data_yaml_path),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "seed": seed,
        "device": resolved_device,
        "workers": 0 if sys.platform == "win32" else 2,
        "project": str(output_dir),
        "name": run_name,
        "exist_ok": True,
        "save": True,
        "save_period": 10,
        "plots": True,
        "amp": True if "cuda" in resolved_device else False,
        # Siêu tham số Augmentation chuẩn hóa vừa phải cho fine-tuning
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.3,
        "fliplr": 0.5,
        "mosaic": 0.5,
    }

    if cfg["freeze"] is not None:
        train_args["freeze"] = cfg["freeze"]

    results = model.train(**train_args)
    elapsed_minutes = (time.time() - t0) / 60.0

    # Xác định file best weight sau khi train
    trained_best_pt = run_save_dir / "weights" / "best.pt"
    final_output_pt = output_dir / cfg["output_model"]

    if trained_best_pt.exists():
        # Sao chép và lưu riêng thành tên file chuẩn (da_full.pt, da_freeze.pt, da_replay.pt)
        shutil.copy2(trained_best_pt, final_output_pt)
        print(f"🎯 Đã lưu trọng số tối ưu chiến lược: {final_output_pt.resolve()}")

    res_summary = {
        "strategy": strategy_key,
        "status": "success",
        "training_time_min": elapsed_minutes,
        "epochs_completed": epochs,
        "seed": seed,
        "trained_best_pt": str(trained_best_pt) if trained_best_pt.exists() else None,
        "final_output_pt": str(final_output_pt) if final_output_pt.exists() else None,
        "freeze_info": freeze_info,
        "results_dict": getattr(results, "results_dict", {})
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res_summary


def run_domain_adaptation_pipeline(
    strategies: List[str],
    base_weights: Path = PROJECT_ROOT / "best.pt",
    output_dir: Path = PROJECT_ROOT / "runs" / "domain_adaptation",
    epochs: int = 30,
    batch_size: int = 8,
    imgsz: int = 640,
    seed: int = 42,
    device: str = "auto",
    dry_run: bool = False
) -> Dict[str, Any]:
    """Thực thi pipeline huấn luyện các chiến lược Thích nghi miền."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "training_summary.json"

    print("=" * 70)
    print("🚀 BẮT ĐẦU PIPELINE HUẤN LUYỆN THÍCH NGHI MIỀN (TUẦN 5 - MT4)")
    print(f"  • Các chiến lược thực thi: {strategies}")
    print(f"  • Trọng số gốc: {base_weights}")
    print(f"  • Thư mục lưu kết quả: {output_dir}")
    print(f"  • Epochs: {epochs} | Batch: {batch_size} | Seed: {seed}")
    print(f"  • Chế độ Dry-Run: {dry_run}")
    print("=" * 70)

    pipeline_results = {}
    for strat in strategies:
        res = train_single_strategy(
            strategy_key=strat,
            base_weights=base_weights,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=imgsz,
            seed=seed,
            device=device,
            dry_run=dry_run
        )
        pipeline_results[strat] = res

    # Ghi nhận summary
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_results, f, indent=2, ensure_ascii=False)
    print(f"\n📑 Đã lưu tổng kết pipeline huấn luyện: {summary_path.resolve()}")

    return pipeline_results


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Thích nghi miền (MT4 Domain Adaptation)")
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["all", "full", "freeze", "replay"],
        help="Chiến lược cần huấn luyện (full, freeze, replay, hoặc all)"
    )
    parser.add_argument("--weights", type=str, default="best.pt", help="Đường dẫn file trọng số ban đầu")
    parser.add_argument("--output-dir", type=str, default="runs/domain_adaptation", help="Thư mục lưu mô hình đầu ra")
    parser.add_argument("--epochs", type=int, default=30, help="Số epochs huấn luyện (mặc định 30)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (mặc định 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào (mặc định 640)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed cố định (mặc định 42)")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị tính toán: auto, cuda:0, cpu")
    parser.add_argument("--dry-run", action="store_true", help="Chạy kiểm tra cấu hình không tốn chi phí huấn luyện")

    args = parser.parse_args()

    base_weights = Path(args.weights)
    if not base_weights.is_absolute():
        base_weights = PROJECT_ROOT / base_weights

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    if args.strategy == "all":
        strategies = ["full", "freeze", "replay"]
    else:
        strategies = [args.strategy]

    run_domain_adaptation_pipeline(
        strategies=strategies,
        base_weights=base_weights,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        seed=args.seed,
        device=args.device,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

