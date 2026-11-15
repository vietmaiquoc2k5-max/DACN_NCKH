"""Đánh giá Thích nghi miền & Tính mức quên thảm khốc (Catastrophic Forgetting) (MT4).

Đề tài NCKH: Phát hiện & cảnh báo cháy từ camera giám sát.
Chỉ tiêu MT4:
- mAP@0.5 trên tập Việt Nam: Tăng >= 5 điểm % (+0.05) so với Baseline.
- mAP@0.5 trên tập D-Fire: Giảm <= 2 điểm % (-0.02) so với Baseline.

Quy trình:
1. Đánh giá mô hình Baseline (best.pt) và 3 mô hình thích nghi miền (da_full, da_freeze, da_replay).
2. Thực hiện kiểm định trên 2 tập Test độc lập:
   - Tập Test Việt Nam (data_vn.yaml, split='test', 630 ảnh).
   - Tập Test D-Fire (data_dfire.yaml, split='test', 2.153 ảnh).
3. Tự động tính chênh lệch Delta mAP và kết luận đối chiếu chỉ tiêu MT4.
"""
import os
import sys
import json
import time
import argparse
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Ngưỡng chỉ tiêu MT4 theo đề cương NCKH
MT4_VN_TARGET_DELTA = 0.05      # Tăng >= 5 điểm % (+0.05) trên tập VN
MT4_DFIRE_MAX_FORGETTING = -0.02  # Giảm <= 2 điểm % (-0.02) trên tập D-Fire


def evaluate_model_on_split(
    model_path: Path,
    data_yaml: Path,
    split: str = "test",
    imgsz: int = 640,
    device: str = "auto",
    batch: int = 16
) -> Dict[str, float]:
    """Đánh giá một mô hình YOLO trên tập kiểm định cụ thể và trích xuất metrics."""
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Không tìm thấy data config: {data_yaml}")

    resolved_device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    model = YOLO(str(model_path))
    results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=resolved_device,
        workers=0 if sys.platform == "win32" else 2,
        plots=False,
        save=False,
        verbose=False
    )

    box = results.box
    map50 = float(box.map50)
    map50_95 = float(box.map)
    precision = float(box.mp)
    recall = float(box.mr)

    # Per-class mAP@0.5
    class_map50 = {}
    maps = box.maps  # array of mAP50-95 per class or similar
    # box.map50 per class có trong box.all_ap hoặc box.class_result
    names = results.names
    names_dict = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
    for cls_idx, cls_name in names_dict.items():
        if hasattr(box, "ap50") and len(box.ap50) > cls_idx:
            class_map50[cls_name] = float(box.ap50[cls_idx])
        elif hasattr(box, "maps") and len(box.maps) > cls_idx:
            class_map50[cls_name] = float(box.maps[cls_idx])

    return {
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "class_map50": class_map50
    }


def compute_adaptation_delta(
    baseline_metrics: Dict[str, float],
    da_metrics: Dict[str, float]
) -> Dict[str, float]:
    """Tính toán chênh lệch (Delta) mAP giữa mô hình DA và Baseline."""
    delta_map50 = da_metrics["map50"] - baseline_metrics["map50"]
    delta_map50_95 = da_metrics["map50_95"] - baseline_metrics["map50_95"]
    delta_recall = da_metrics.get("recall", 0.0) - baseline_metrics.get("recall", 0.0)
    delta_precision = da_metrics.get("precision", 0.0) - baseline_metrics.get("precision", 0.0)

    return {
        "delta_map50": delta_map50,
        "delta_map50_percentage": delta_map50 * 100.0,
        "delta_map50_95": delta_map50_95,
        "delta_recall": delta_recall,
        "delta_precision": delta_precision
    }


def verify_mt4_compliance(delta_vn: float, delta_dfire: float) -> Tuple[bool, bool, bool]:
    """Kiểm tra đối chiếu với chỉ tiêu MT4:
    - VN Pass: delta_vn >= +0.05
    - D-Fire Pass: delta_dfire >= -0.02 (mức giảm <= 2 điểm %)
    - Overall Pass: Cả hai điều kiện đều đạt
    """
    vn_passed = delta_vn >= (MT4_VN_TARGET_DELTA - 1e-5)
    dfire_passed = delta_dfire >= (MT4_DFIRE_MAX_FORGETTING - 1e-5)
    overall_passed = vn_passed and dfire_passed
    return vn_passed, dfire_passed, overall_passed


def format_evaluation_table(eval_data: Dict[str, Any]) -> str:
    """Định dạng bảng kết quả so sánh chi tiết dạng Markdown."""
    lines = [
        "| Mô hình / Chiến lược | mAP@0.5 (VN) | Δ mAP VN | mAP@0.5 (D-Fire) | Δ mAP D-Fire (Quên) | Đạt MT4 VN (≥+5%) | Đạt MT4 D-Fire (≤2%) | Kết luận MT4 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    base = eval_data.get("baseline", {})
    base_vn = base.get("vn", {}).get("map50", 0.0)
    base_df = base.get("dfire", {}).get("map50", 0.0)

    lines.append(f"| **Baseline (`best.pt`)** | {base_vn*100:.2f}% | — | {base_df*100:.2f}% | — | — | — | **BASELINE** |")

    strategies = [
        ("Strategy 1: Full", "strategy_1_full"),
        ("Strategy 2: Freeze", "strategy_2_freeze"),
        ("Strategy 3: Freeze + Replay 20%", "strategy_3_replay")
    ]

    for display_name, key in strategies:
        entry = eval_data.get(key)
        if not entry:
            lines.append(f"| {display_name} | *Chưa chạy* | — | *Chưa chạy* | — | — | — | *N/A* |")
            continue

        vn_m = entry.get("vn", {}).get("map50", 0.0)
        df_m = entry.get("dfire", {}).get("map50", 0.0)
        d_vn = entry.get("delta_vn", {}).get("delta_map50", 0.0)
        d_df = entry.get("delta_dfire", {}).get("delta_map50", 0.0)

        vn_ok, df_ok, overall_ok = verify_mt4_compliance(d_vn, d_df)

        vn_badge = "✅ ĐẠT" if vn_ok else "❌ CHƯA ĐẠT"
        df_badge = "✅ ĐẠT" if df_ok else "❌ CHƯA ĐẠT"
        status_badge = "🟢 **PASS**" if overall_ok else "🔴 **FAIL**"

        d_vn_str = f"{d_vn*100:+.2f}%"
        d_df_str = f"{d_df*100:+.2f}%"

        lines.append(f"| {display_name} | {vn_m*100:.2f}% | **{d_vn_str}** | {df_m*100:.2f}% | **{d_df_str}** | {vn_badge} | {df_badge} | {status_badge} |")

    return "\n".join(lines)


def run_complete_evaluation(
    baseline_path: Path = PROJECT_ROOT / "best.pt",
    da_full_path: Optional[Path] = None,
    da_freeze_path: Optional[Path] = None,
    da_replay_path: Optional[Path] = None,
    vn_yaml: Path = PROJECT_ROOT / "data_vn.yaml",
    dfire_yaml: Path = PROJECT_ROOT / "data_dfire.yaml",
    output_json: Path = PROJECT_ROOT / "runs" / "domain_adaptation" / "eval_summary.json",
    device: str = "auto",
    batch: int = 16
) -> Dict[str, Any]:
    """Thực thi toàn bộ quy trình đánh giá 4 mô hình trên 2 tập test và tính delta."""
    print("=" * 75)
    print("🔬 BẮT ĐẦU ĐÁNH GIÁ THÍCH NGHI MIỀN & TÍNH MỨC QUÊN (MT4)")
    print(f"  • Tập Test VN: {vn_yaml}")
    print(f"  • Tập Test D-Fire: {dfire_yaml}")
    print(f"  • Baseline: {baseline_path}")
    print("=" * 75)

    results = {}

    # 1. Đánh giá Baseline
    print("\n▶ Đang đánh giá BASELINE (best.pt)...")
    base_vn = evaluate_model_on_split(baseline_path, vn_yaml, split="test", device=device, batch=batch)
    base_dfire = evaluate_model_on_split(baseline_path, dfire_yaml, split="test", device=device, batch=batch)
    results["baseline"] = {
        "model_path": str(baseline_path),
        "vn": base_vn,
        "dfire": base_dfire
    }
    print(f"  ✓ Baseline VN: mAP50 = {base_vn['map50']*100:.2f}% | mAP50-95 = {base_vn['map50_95']*100:.2f}%")
    print(f"  ✓ Baseline D-Fire: mAP50 = {base_dfire['map50']*100:.2f}% | mAP50-95 = {base_dfire['map50_95']*100:.2f}%")

    # Danh sách các mô hình DA cần đánh giá
    model_configs = [
        ("strategy_1_full", "Strategy 1: Full", da_full_path or (PROJECT_ROOT / "runs" / "domain_adaptation" / "da_full.pt")),
        ("strategy_2_freeze", "Strategy 2: Freeze", da_freeze_path or (PROJECT_ROOT / "runs" / "domain_adaptation" / "da_freeze.pt")),
        ("strategy_3_replay", "Strategy 3: Freeze + Replay", da_replay_path or (PROJECT_ROOT / "runs" / "domain_adaptation" / "da_replay.pt"))
    ]

    for key, name, model_p in model_configs:
        if not model_p.exists():
            print(f"\n⚠️ Bỏ qua {name}: Không tìm thấy file {model_p}")
            continue

        print(f"\n▶ Đang đánh giá {name} ({model_p.name})...")
        m_vn = evaluate_model_on_split(model_p, vn_yaml, split="test", device=device, batch=batch)
        m_dfire = evaluate_model_on_split(model_p, dfire_yaml, split="test", device=device, batch=batch)

        delta_vn = compute_adaptation_delta(base_vn, m_vn)
        delta_dfire = compute_adaptation_delta(base_dfire, m_dfire)

        vn_ok, df_ok, overall_ok = verify_mt4_compliance(delta_vn["delta_map50"], delta_dfire["delta_map50"])

        results[key] = {
            "model_path": str(model_p),
            "vn": m_vn,
            "dfire": m_dfire,
            "delta_vn": delta_vn,
            "delta_dfire": delta_dfire,
            "mt4_compliance": {
                "vn_passed": vn_ok,
                "dfire_passed": df_ok,
                "overall_passed": overall_ok
            }
        }
        print(f"  ✓ {name} VN: mAP50 = {m_vn['map50']*100:.2f}% (Δ = {delta_vn['delta_map50_percentage']:+.2f}%)")
        print(f"  ✓ {name} D-Fire: mAP50 = {m_dfire['map50']*100:.2f}% (Δ = {delta_dfire['delta_map50_percentage']:+.2f}%)")
        print(f"  ✓ Kết luận MT4: {'PASS' if overall_ok else 'FAIL'}")

    # Tạo bảng markdown
    md_table = format_evaluation_table(results)
    print("\n" + "=" * 75)
    print("📊 BẢNG TỔNG HỢP SO SÁNH CHỈ TIÊU MT4:")
    print("=" * 75)
    print(md_table)
    print("=" * 75)

    # Lưu file JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📑 Đã lưu báo cáo đánh giá JSON: {output_json.resolve()}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Đánh giá Thích nghi miền & Tính mức quên thảm khốc (MT4)")
    parser.add_argument("--baseline", type=str, default="best.pt", help="Trọng số baseline gốc")
    parser.add_argument("--da-full", type=str, default="runs/domain_adaptation/da_full.pt", help="Mô hình chiến lược 1 Full")
    parser.add_argument("--da-freeze", type=str, default="runs/domain_adaptation/da_freeze.pt", help="Mô hình chiến lược 2 Freeze")
    parser.add_argument("--da-replay", type=str, default="runs/domain_adaptation/da_replay.pt", help="Mô hình chiến lược 3 Replay")
    parser.add_argument("--vn-yaml", type=str, default="data_vn.yaml", help="YAML tập Việt Nam")
    parser.add_argument("--dfire-yaml", type=str, default="data_dfire.yaml", help="YAML tập D-Fire")
    parser.add_argument("--output-json", type=str, default="runs/domain_adaptation/eval_summary.json", help="File JSON kết quả")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị: auto, cuda:0, cpu")
    parser.add_argument("--batch", type=int, default=16, help="Batch size kiểm định")

    args = parser.parse_args()

    baseline_p = Path(args.baseline)
    if not baseline_p.is_absolute():
        baseline_p = PROJECT_ROOT / baseline_p

    da_full_p = Path(args.da_full)
    if not da_full_p.is_absolute():
        da_full_p = PROJECT_ROOT / da_full_p

    da_freeze_p = Path(args.da_freeze)
    if not da_freeze_p.is_absolute():
        da_freeze_p = PROJECT_ROOT / da_freeze_p

    da_replay_p = Path(args.da_replay)
    if not da_replay_p.is_absolute():
        da_replay_p = PROJECT_ROOT / da_replay_p

    vn_yaml_p = Path(args.vn_yaml)
    if not vn_yaml_p.is_absolute():
        vn_yaml_p = PROJECT_ROOT / vn_yaml_p

    dfire_yaml_p = Path(args.dfire_yaml)
    if not dfire_yaml_p.is_absolute():
        dfire_yaml_p = PROJECT_ROOT / dfire_yaml_p

    output_json_p = Path(args.output_json)
    if not output_json_p.is_absolute():
        output_json_p = PROJECT_ROOT / output_json_p

    run_complete_evaluation(
        baseline_path=baseline_p,
        da_full_path=da_full_p if da_full_p.exists() else None,
        da_freeze_path=da_freeze_p if da_freeze_p.exists() else None,
        da_replay_path=da_replay_p if da_replay_p.exists() else None,
        vn_yaml=vn_yaml_p,
        dfire_yaml=dfire_yaml_p,
        output_json=output_json_p,
        device=args.device,
        batch=args.batch
    )


if __name__ == "__main__":
    main()

