import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

# Đảm bảo in tiếng Việt không lỗi bảng mã trên Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def evaluate_single_model(model_path, data_yaml="data.yaml", imgsz=640, device="cpu"):
    print(f"\n⏳ Đang đánh giá mô hình: {model_path} trên tập Validation...")
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file model: {model_path}")
        return None

    try:
        model = YOLO(model_path)
        metrics = model.val(
            data=data_yaml,
            split="val",
            imgsz=imgsz,
            device=device,
            verbose=False
        )

        results = {
            "model_path": model_path,
            "classes": model.names,
            "map50": float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0,
            "map50_95": float(metrics.box.map) if hasattr(metrics.box, 'map') else 0.0,
            "precision": float(metrics.box.mp) if hasattr(metrics.box, 'mp') else 0.0,
            "recall": float(metrics.box.mr) if hasattr(metrics.box, 'mr') else 0.0,
        }
        return results
    except Exception as e:
        print(f"❌ Lỗi khi đánh giá model {model_path}: {e}")
        return None

def print_metrics_table(results_list):
    print("\n" + "="*75)
    print("📊 BẢNG SO SÁNH CHỈ SỐ ĐÁNH GIÁ TRÊN TẬP VALIDATION")
    print("="*75)
    header = f"{'Model':<30} | {'mAP@50':<10} | {'mAP@50-95':<12} | {'Precision':<10} | {'Recall':<10}"
    print(header)
    print("-" * len(header))
    for r in results_list:
        name = Path(r['model_path']).name
        print(f"{name:<30} | {r['map50']*100:>8.2f}% | {r['map50_95']*100:>10.2f}% | {r['precision']*100:>8.2f}% | {r['recall']*100:>8.2f}%")
    print("="*75 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Đánh giá và so sánh mô hình YOLO trên tập validation")
    parser.add_argument("--model", default="best.pt", help="Đường dẫn model cần đánh giá")
    parser.add_argument("--compare", default=None, help="Đường dẫn model cũ để so sánh (tùy chọn)")
    parser.add_argument("--data", default="data.yaml", help="File data.yaml dùng để val")
    parser.add_argument("--device", default="cpu", help="Thiết bị chạy ('0' hoặc 'cpu')")
    args = parser.parse_args()

    results = []
    res1 = evaluate_single_model(args.model, data_yaml=args.data, device=args.device)
    if res1:
        results.append(res1)

    if args.compare and os.path.exists(args.compare):
        res2 = evaluate_single_model(args.compare, data_yaml=args.data, device=args.device)
        if res2:
            results.append(res2)

    if results:
        print_metrics_table(results)

if __name__ == "__main__":
    main()
