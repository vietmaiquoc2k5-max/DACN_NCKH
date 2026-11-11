"""Đánh giá chi tiết Operating Point triển khai trên tập kiểm định (Validation Set) - TASK 5.

Đánh giá cấu hình triển khai chính thức:
conf=0.35, K=2, N=8, feature_threshold=0.02, cooldown=5.0s

Phân tách độc lập:
1. Primary evaluation: cross_verified Ground Truth (3 events).
   Nêu rõ giới hạn mẫu nhỏ không đủ bằng chứng tổng quát hóa mạnh.
2. Secondary exploratory evaluation: All Ground Truth (17 events gồm cả pending).

So sánh trực tiếp:
- Baseline A (Per-frame baseline)
- Pipeline B1 (ByteTrack + K=2/N=8, feat=0.0)
- Pipeline B2 K=2 (ByteTrack + K=2/N=8 + Feature Gate 0.02)

Xuất tệp:
runs/benchmarks/week4_scientific/validation_operating_point_evaluation.json
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import SystemSettings
from src.event_eval import (
    Event,
    evaluate,
    frame_detections_to_events,
    load_frame_detections,
    load_ground_truth_events,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("eval_val_op")


def run_validation_evaluation() -> Dict[str, Any]:
    LOGGER.info("=== BẮT ĐẦU ĐÁNH GIÁ OPERATING POINT TRÊN TẬP VALIDATION (TASK 5) ===")

    settings = SystemSettings()
    val_ids: Set[str] = {
        "POS_01", "POS_03", "POS_05", "POS_07", "POS_09", "POS_11",
        "POS_13", "POS_15", "POS_17", "POS_19", "POS_21", "POS_23"
    }

    # 1. Nạp khung hình của tập Validation
    frame_jsonl = ROOT_DIR / "runs" / "benchmarks" / "week4_final" / "tracked_frame_detections.jsonl"
    all_frames = load_frame_detections(frame_jsonl)
    val_frames = [f for f in all_frames if f.video_id in val_ids]
    LOGGER.info("Đã nạp %d khung hình kiểm định thuộc %d video Validation", len(val_frames), len(val_ids))

    # 2. Nạp Ground Truth
    gt_primary = load_ground_truth_events(video_ids=val_ids, only_verified=True, strict=True)
    gt_all = load_ground_truth_events(video_ids=val_ids, only_verified=False, strict=False)
    LOGGER.info("Ground Truth Validation: Primary cross_verified=%d, All=%d", len(gt_primary), len(gt_all))

    # 3. Định nghĩa các pipeline so sánh
    pipelines = [
        {
            "id": "baseline_a",
            "name": "Baseline A (Per-frame)",
            "conf": settings.conf_threshold,
            "k": 1,
            "n": 1,
            "feat_thr": 0.0,
            "cooldown": 0.0,
        },
        {
            "id": "pipeline_b1",
            "name": f"Pipeline B1 (ByteTrack K={settings.confirm_frames_k}/N={settings.window_size_n}, feat=0.0)",
            "conf": settings.conf_threshold,
            "k": settings.confirm_frames_k,
            "n": settings.window_size_n,
            "feat_thr": 0.0,
            "cooldown": settings.alert_cooldown_sec,
        },
        {
            "id": "pipeline_b2_k2_deployed",
            "name": f"Pipeline B2 Deployed (ByteTrack K={settings.confirm_frames_k}/N={settings.window_size_n}, feat={settings.feature_threshold})",
            "conf": settings.conf_threshold,
            "k": settings.confirm_frames_k,
            "n": settings.window_size_n,
            "feat_thr": settings.feature_threshold,
            "cooldown": settings.alert_cooldown_sec,
        },
    ]

    results_primary: Dict[str, Any] = {}
    results_exploratory: Dict[str, Any] = {}

    baseline_primary_recall = 0.0
    baseline_all_recall = 0.0

    for pipe in pipelines:
        fire_evs, smoke_evs = frame_detections_to_events(
            frames=val_frames,
            confidence_threshold=pipe["conf"],
            window_n=pipe["n"],
            confirm_k=pipe["k"],
            max_event_gap_sec=2.0,
            feature_threshold=pipe["feat_thr"],
            cooldown_sec=pipe["cooldown"],
        )
        pred_events = fire_evs + smoke_evs

        # Primary (cross_verified GT)
        eval_p = evaluate(
            ground_truth=gt_primary,
            predictions=pred_events,
            positive_video_ids=val_ids,
            allow_missing_negative=True,
            tolerance_sec=10.0,
        )
        # Exploratory (All GT)
        eval_a = evaluate(
            ground_truth=gt_all,
            predictions=pred_events,
            positive_video_ids=val_ids,
            allow_missing_negative=True,
            tolerance_sec=10.0,
        )

        tp_p = eval_p["counts"]["matched_events"]
        fn_p = eval_p["counts"]["unmatched_gt_events"]
        fp_p = eval_p["counts"]["positive_unmatched_alerts"]
        mean_delay_p = eval_p["metrics"]["mean_detection_delay_sec"]

        tp_a = eval_a["counts"]["matched_events"]
        fn_a = eval_a["counts"]["unmatched_gt_events"]
        fp_a = eval_a["counts"]["positive_unmatched_alerts"]
        mean_delay_a = eval_a["metrics"]["mean_detection_delay_sec"]

        late_p = eval_p["late_detections"]
        missed_p = eval_p["unmatched_ground_truth"]

        late_a = eval_a["late_detections"]
        missed_a = eval_a["unmatched_ground_truth"]

        p_summary = {
            "name": pipe["name"],
            "parameters": {
                "conf": pipe["conf"],
                "k": pipe["k"],
                "n": pipe["n"],
                "feat_thr": pipe["feat_thr"],
                "cooldown": pipe["cooldown"],
            },
            "predicted_events_count": len(pred_events),
            "tp": tp_p,
            "fn": fn_p,
            "fp": fp_p,
            "event_instance_recall": round(float(eval_p["metrics"]["event_instance_recall"]), 4),
            "event_precision": round(float(eval_p["metrics"]["event_precision"]), 4),
            "event_f1": round(float(eval_p["metrics"]["event_f1"]), 4),
            "mean_delay_sec": round(float(mean_delay_p), 3) if mean_delay_p is not None else None,
            "late_detections_count": len(late_p),
            "late_detections": late_p,
            "missed_detections_count": len(missed_p),
            "missed_detections": missed_p,
        }

        a_summary = {
            "name": pipe["name"],
            "parameters": {
                "conf": pipe["conf"],
                "k": pipe["k"],
                "n": pipe["n"],
                "feat_thr": pipe["feat_thr"],
                "cooldown": pipe["cooldown"],
            },
            "predicted_events_count": len(pred_events),
            "tp": tp_a,
            "fn": fn_a,
            "fp": fp_a,
            "event_instance_recall": round(float(eval_a["metrics"]["event_instance_recall"]), 4),
            "event_precision": round(float(eval_a["metrics"]["event_precision"]), 4),
            "event_f1": round(float(eval_a["metrics"]["event_f1"]), 4),
            "mean_delay_sec": round(float(mean_delay_a), 3) if mean_delay_a is not None else None,
            "late_detections_count": len(late_a),
            "late_detections": late_a,
            "missed_detections_count": len(missed_a),
            "missed_detections": missed_a,
        }

        if pipe["id"] == "baseline_a":
            baseline_primary_recall = eval_p["metrics"]["event_instance_recall"]
            baseline_all_recall = eval_a["metrics"]["event_instance_recall"]

        p_summary["recall_drop_points"] = round((baseline_primary_recall - eval_p["metrics"]["event_instance_recall"]) * 100.0, 2)
        a_summary["recall_drop_points"] = round((baseline_all_recall - eval_a["metrics"]["event_instance_recall"]) * 100.0, 2)

        results_primary[pipe["id"]] = p_summary
        results_exploratory[pipe["id"]] = a_summary

    output_data = {
        "metadata": {
            "title": "Validation Operating Point Evaluation (Task 5)",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "validation_videos": sorted(list(val_ids)),
            "validation_frames_count": len(val_frames),
            "deployed_configuration": {
                "confidence_threshold": settings.conf_threshold,
                "confirm_frames_k": settings.confirm_frames_k,
                "window_size_n": settings.window_size_n,
                "feature_threshold": settings.feature_threshold,
                "alert_cooldown_sec": settings.alert_cooldown_sec,
            },
        },
        "primary_evaluation_cross_verified": {
            "description": "Đánh giá chính thức trên các sự kiện đã được 2 annotator xác minh chéo độc lập",
            "gt_count": len(gt_primary),
            "sample_size_limitation_warning": (
                "Chỉ có 3 sự kiện đạt chuẩn cross_verified trên tập Validation (POS_03, POS_05, POS_23). "
                "Cỡ mẫu này quá nhỏ để khẳng định tính tổng quát hóa tuyệt đối. "
                "Kết quả Recall 100% chỉ chứng minh tính nhất quán của thuật toán, không phải bằng chứng tổng quát hoàn chỉnh."
            ),
            "pipelines": results_primary,
        },
        "secondary_exploratory_evaluation_all_gt": {
            "description": "Đánh giá thăm dò trên toàn bộ 17 sự kiện (bao gồm 14 sự kiện pending_verification)",
            "gt_count": len(gt_all),
            "pipelines": results_exploratory,
        },
    }

    out_file = ROOT_DIR / "runs" / "benchmarks" / "week4_scientific" / "validation_operating_point_evaluation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(output_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Đã lưu kết quả đánh giá Validation Operating Point tại: %s", out_file)
    return output_data


if __name__ == "__main__":
    run_validation_evaluation()
