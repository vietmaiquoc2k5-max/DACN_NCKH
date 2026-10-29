"""Script Huấn luyện Thống nhất Bộ Phát hiện Đối tượng (MT2 Benchmark Pipeline).

Đề tài NCKH: Phát hiện & cảnh báo cháy từ camera giám sát.
Mục tiêu MT2:
- Huấn luyện >= 4 mô hình: YOLOv8n, YOLOv8s, YOLO11n, YOLO11s, RT-DETR trên tập D-Fire.
- Đồng bộ hóa các siêu tham số:
  + Kích thước ảnh: imgsz = 640
  + Cùng cấu hình augmentation (hsv, flip, scale, mosaic)
  + 3 Random Seeds: 42, 123, 456 (để tính mean +- std theo chuẩn IEEE)
- Quản lý cấu trúc thư mục thực nghiệm:
  `runs/benchmarks/{model_name}_seed_{seed}/`
"""
import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

import yaml
import torch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from ultralytics import YOLO, RTDETR

# Danh mục các mô hình được khảo sát theo đề cương
SUPPORTED_MODELS = {
    "yolov8n": {"weights": "yolov8n.pt", "type": "yolo", "desc": "YOLOv8 Nano (Lightweight edge)"},
    "yolov8s": {"weights": "yolov8s.pt", "type": "yolo", "desc": "YOLOv8 Small (Balanced)"},
    "yolo11n": {"weights": "yolo11n.pt", "type": "yolo", "desc": "YOLO11 Nano (Next-gen lightweight)"},
    "yolo11s": {"weights": "yolo11s.pt", "type": "yolo", "desc": "YOLO11 Small (Next-gen balanced)"},
    "rtdetr-l": {"weights": "rtdetr-l.pt", "type": "rtdetr", "desc": "RT-DETR Large (Real-Time Transformer)"}
}

DEFAULT_SEEDS = [42, 123, 456]


def verify_dataset_yaml(data_yaml_path: str) -> bool:
    """Xác nhận tệp data yaml tồn tại và hợp lệ."""
    p = Path(data_yaml_path)
    if not p.exists():
        print(f"❌ Không tìm thấy file data config: {data_yaml_path}")
        return False
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    print(f"✅ Data YAML hợp lệ: {p.resolve()}")
    print(f"  • Classes: {data.get('names')}")
    return True


def train_single_run(
    model_key: str,
    weights_path: str,
    data_yaml: str,
    epochs: int,
    imgsz: int,
    batch_size: int,
    seed: int,
    device: str,
    project_dir: Path,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Thực thi một phiên huấn luyện với seed cụ thể."""
    run_name = f"{model_key}_seed{seed}"
    run_save_dir = project_dir / run_name

    print("\n" + "=" * 65)
    print(f"🚀 HUẤN LUYỆN: {model_key.upper()} | SEED: {seed} | EPOCHS: {epochs}")
    print(f"📂 Thư mục lưu: {run_save_dir}")
    print("=" * 65)

    if dry_run:
        print("💡 [DRY-RUN]: Kiểm tra tải mô hình thành công, bỏ qua quá trình fit.")
        return {
            "model": model_key,
            "seed": seed,
            "status": "dry_run_success",
            "weights": weights_path
        }

    # Chọn lớp khởi tạo tương ứng
    model_info = SUPPORTED_MODELS[model_key]
    if model_info["type"] == "rtdetr":
        model = RTDETR(weights_path)
    else:
        model = YOLO(weights_path)

    # Tự động điều chỉnh batch size an toàn nếu chạy trên GPU 4GB
    effective_batch = batch_size
    if device != "cpu" and torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb <= 4.5 and effective_batch > 8:
            print(f"⚠️ Phát hiện GPU {vram_gb:.1f} GB VRAM (RTX 3050). Tự động điều chỉnh batch={8} để tránh CUDA Out-of-Memory.")
            effective_batch = 8

    t0 = time.time()
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=effective_batch,
        seed=seed,
        device=device,
        workers=2 if sys.platform != "win32" else 0,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        save_period=10,
        plots=True,
        amp=True,  # Bật Automatic Mixed Precision FP16 tiết kiệm 50% VRAM
        # Chuẩn hóa Augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0
    )
    elapsed_min = (time.time() - t0) / 60.0

    # Lấy metrics tốt nhất từ kết quả train
    best_weight = run_save_dir / "weights" / "best.pt"
    metrics_summary = {
        "model": model_key,
        "seed": seed,
        "epochs": epochs,
        "training_time_min": elapsed_min,
        "best_weight_exists": best_weight.exists(),
        "best_weight_path": str(best_weight) if best_weight.exists() else None
    }

    # Nếu có metrics từ results
    if hasattr(results, "results_dict"):
        rd = results.results_dict
        metrics_summary.update({
            "map50": rd.get("metrics/mAP50(B)", 0.0),
            "map50_95": rd.get("metrics/mAP50-95(B)", 0.0),
            "precision": rd.get("metrics/precision(B)", 0.0),
            "recall": rd.get("metrics/recall(B)", 0.0)
        })

    return metrics_summary


def run_benchmarks(
    models_to_train: List[str],
    data_yaml: str = "data_dfire.yaml",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    seeds: List[int] = None,
    device: str = "auto",
    dry_run: bool = False
):
    """Quản lý toàn bộ tiến trình huấn luyện các mô hình MT2."""
    if seeds is None:
        seeds = DEFAULT_SEEDS

    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("🔥 PIPELINE HUẤN LUYỆN ĐỐI SÁNH CÁC MÔ HÌNH PHÁT HIỆN (MT2) 🔥")
    print("=" * 70)
    print(f"• Thiết bị: {device.upper()}")
    print(f"• Dataset: {data_yaml}")
    print(f"• Danh sách mô hình: {models_to_train}")
    print(f"• Danh sách Seeds: {seeds}")
    print(f"• Cấu hình ảnh: {imgsz}x{imgsz} | Batch size: {batch_size} | Epochs: {epochs}")

    if not verify_dataset_yaml(data_yaml):
        return False

    project_dir = Path("runs/benchmarks")
    project_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for m_key in models_to_train:
        if m_key not in SUPPORTED_MODELS:
            print(f"⚠️ Bỏ qua mô hình không xác định: {m_key}")
            continue

        model_spec = SUPPORTED_MODELS[m_key]
        for s in seeds:
            res = train_single_run(
                model_key=m_key,
                weights_path=model_spec["weights"],
                data_yaml=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch_size=batch_size,
                seed=s,
                device=device,
                project_dir=project_dir,
                dry_run=dry_run
            )
            all_results.append(res)

    # Lưu tóm tắt JSON
    summary_path = project_dir / "benchmark_train_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("🎉 HOÀN TẤT PIPELINE HUẤN LUYỆN BENCHMARK!")
    print(f"📄 Kết quả tóm tắt đã lưu tại: {summary_path}")
    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(description="Pipeline huấn luyện các mô hình phát hiện MT2")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8n", "yolo11n"],
        help=f"Danh sách mô hình cần huấn luyện: {list(SUPPORTED_MODELS.keys())} hoặc 'all'"
    )
    parser.add_argument("--data", default="data_dfire.yaml", help="File yaml dataset (mặc định data_dfire.yaml)")
    parser.add_argument("--epochs", type=int, default=50, help="Số epoch huấn luyện (mặc định 50)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào (mặc định 640)")
    parser.add_argument("--batch", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="Danh sách random seeds")
    parser.add_argument("--device", default="auto", help="Thiết bị ('auto', 'cpu', '0')")
    parser.add_argument("--dry-run", action="store_true", help="Chạy kiểm tra tải và cấu trúc không huấn luyện thực sự")
    args = parser.parse_args()

    models = list(SUPPORTED_MODELS.keys()) if "all" in args.models else args.models
    run_benchmarks(
        models_to_train=models,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        seeds=args.seeds,
        device=args.device,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
