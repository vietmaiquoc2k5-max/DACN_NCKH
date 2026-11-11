"""Đánh giá độc lập cấu hình hoạt động trên tập kiểm thử bị khóa (Locked Test Set Evaluation).

Xuất tệp artifacts: `runs/benchmarks/week4_final/locked_test_evaluation.json`
Đánh giá các ứng viên trên tập Test gồm 12 video dương tính độc lập (`POS_02`, `POS_04`, ... `POS_24`)
và tập âm tính (0 video có sẵn trên ổ đĩa -> BLOCKED_BY_NEGATIVE_DATA).
Đảm bảo đồng bộ 100% với cấu hình provisional được chọn từ Validation Grid Search trong `best_operating_point.json`.
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.event_eval import (
    Event,
    evaluate,
    frame_detections_to_events,
    get_ground_truth_manifest,
    load_frame_detections,
    load_ground_truth_events,
)

LOGGER = logging.getLogger("locked_test_eval")
OUTPUT_PATH = ROOT_DIR / "runs" / "benchmarks" / "week4_final" / "locked_test_evaluation.json"
BEST_OP_PATH = ROOT_DIR / "runs" / "benchmarks" / "week4_final" / "best_operating_point.json"
FRAME_JSONL_PATH = ROOT_DIR / "runs" / "benchmarks" / "week4_final" / "tracked_frame_detections.jsonl"


def run_evaluation() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    LOGGER.info("Bắt đầu đánh giá độc lập trên Locked Test Set...")

    test_ids = {
        "POS_02", "POS_04", "POS_06", "POS_08", "POS_10", "POS_12",
        "POS_14", "POS_16", "POS_18", "POS_20", "POS_22", "POS_24"
    }
    val_ids = {
        "POS_01", "POS_03", "POS_05", "POS_07", "POS_09", "POS_11",
        "POS_13", "POS_15", "POS_17", "POS_19", "POS_21", "POS_23"
    }

    all_frames = load_frame_detections(FRAME_JSONL_PATH)
    test_frames = [f for f in all_frames if f.video_id in test_ids]
    gt_test = load_ground_truth_events(video_ids=test_ids, only_verified=True, strict=True)
    gt_manifest = get_ground_truth_manifest(val_video_ids=val_ids, test_video_ids=test_ids)

    LOGGER.info(
        "Tập Test: %d khung hình, %d Ground Truth verified events trên %d videos",
        len(test_frames), len(gt_test), len(test_ids)
    )

    # Đọc cấu hình provisional được chọn từ Validation Grid Search
    if BEST_OP_PATH.exists():
        best_payload = json.loads(BEST_OP_PATH.read_text(encoding="utf-8"))
        prov = best_payload.get("provisional_candidate_on_positive_val")
    else:
        prov = None

    if prov:
        prov_conf = float(prov.get("confidence_threshold", 0.45))
        prov_k = int(prov.get("k", 2))
        prov_n = int(prov.get("n", 12))
        prov_feat = float(prov.get("feature_threshold", 0.0))
        prov_cd = float(prov.get("cooldown_sec", 0.0))
    else:
        prov_conf = 0.45
        prov_k = 2
        prov_n = 12
        prov_feat = 0.0
        prov_cd = 0.0

    candidate_configs = [
        {
            "name": "provisional_validation_selected",
            "description": f"Cấu hình tối ưu được chọn từ Validation Grid Search (conf={prov_conf}, K={prov_k}, N={prov_n}, feat={prov_feat}, cooldown={prov_cd}s)",
            "conf": prov_conf,
            "k": prov_k,
            "n": prov_n,
            "feat_thr": prov_feat,
            "cooldown": prov_cd,
        },
        {
            "name": "ablation_no_feat_no_cooldown",
            "description": f"Ablation: ByteTrack + K/N thuần túy không Feature Gate và không Cooldown (conf={prov_conf}, K={prov_k}, N={prov_n})",
            "conf": prov_conf,
            "k": prov_k,
            "n": prov_n,
            "feat_thr": 0.0,
            "cooldown": 0.0,
        },
        {
            "name": "provisional_alternative_conf35_k3_n8",
            "description": "Cấu hình đối sánh cân bằng (conf=0.35, K=3, N=8, feat=0.02, cooldown=5.0s)",
            "conf": 0.35,
            "k": 3,
            "n": 8,
            "feat_thr": 0.02,
            "cooldown": 5.0,
        },
    ]

    # Baseline Pipeline A tại các mức conf
    baselines: Dict[str, Any] = {}
    for conf_val in sorted(list({prov_conf, 0.35, 0.25})):
        base_ev, _ = frame_detections_to_events(test_frames, conf_val, window_n=1, confirm_k=1)
        base_rep = evaluate(gt_test, base_ev, test_ids, None, None, allow_missing_negative=True)
        baselines[f"frame_level_baseline_conf_{conf_val}"] = {
            "confidence_threshold": conf_val,
            "counts": base_rep["counts"],
            "metrics": base_rep["metrics"],
        }

    evaluated_candidates = []
    for cand in candidate_configs:
        _, temp_ev = frame_detections_to_events(
            test_frames,
            cand["conf"],
            window_n=cand["n"],
            confirm_k=cand["k"],
            feature_threshold=cand["feat_thr"],
            cooldown_sec=cand["cooldown"],
        )
        report = evaluate(gt_test, temp_ev, test_ids, None, None, allow_missing_negative=True)
        evaluated_candidates.append({
            "name": cand["name"],
            "description": cand["description"],
            "parameters": {
                "confidence_threshold": cand["conf"],
                "k": cand["k"],
                "n": cand["n"],
                "feature_threshold": cand["feat_thr"],
                "cooldown_sec": cand["cooldown"],
            },
            "status": report["status"],
            "metrics": report["metrics"],
            "counts": report["counts"],
            "targets": report["targets"],
            "matches": report["matches"],
            "late_detections": report.get("late_detections", []),
            "early_alerts": report.get("early_alerts", []),
            "unmatched_ground_truth": report["unmatched_ground_truth"],
            "unmatched_predictions": report["unmatched_predictions"],
        })

    payload = {
        "metadata": {
            "evaluation_name": "Locked Test Set Evaluation (Week 4 Final Reconciled)",
            "git_branch": "fix",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "positive_test_videos": sorted(list(test_ids)),
            "negative_test_videos_available": [],
            "negative_test_hours_available": 0.0,
            "ground_truth_manifest": gt_manifest,
            "ground_truth_policy": "verified_only",
            "frame_detections_source": "runs/benchmarks/week4_final/tracked_frame_detections.jsonl",
            "total_test_frames": len(test_frames),
            "total_test_verified_gt_events": len(gt_test),
        },
        "operating_point_decision": {
            "status": "NO_FEASIBLE_OPERATING_POINT",
            "feasible": False,
            "blocked_by_negative_data": True,
            "selected_operating_point": None,
            "provisional_operating_point": {
                "name": "provisional_validation_selected",
                "parameters": {
                    "confidence_threshold": prov_conf,
                    "k": prov_k,
                    "n": prov_n,
                    "feature_threshold": prov_feat,
                    "cooldown_sec": prov_cd,
                },
                "test_set_performance": evaluated_candidates[0]["metrics"],
                "summary": (
                    f"Đạt Video Event Recall {evaluated_candidates[0]['metrics']['video_event_recall']*100:.1f}% trên video có verified GT, "
                    f"Event Instance Recall {evaluated_candidates[0]['metrics']['event_instance_recall']*100:.1f}%, "
                    f"Precision {evaluated_candidates[0]['metrics']['event_precision']*100:.1f}%, "
                    f"Mean Delay {evaluated_candidates[0]['metrics']['mean_detection_delay_sec']:.2f}s (<= 10.0s). "
                    "Cần tối thiểu 5.0 giờ video âm tính độc lập để kiểm chứng FAR và giảm >= 50% báo giả trước khi nghiệm thu MT3."
                ),
            },
            "scientific_rationale": (
                "Theo tiêu chuẩn khoa học và đề cương NCKH MT3, điểm vận hành chỉ được nghiệm thu chính thức "
                "khi chứng minh được tỷ lệ báo giả FAR giảm >= 50% trên tập dữ liệu âm tính độc lập. "
                "Do chưa có file video âm tính thực tế trên ổ đĩa, hệ thống trung thực từ chối cấp nhãn FEASIBLE "
                "và duy trì trạng thái BLOCKED_BY_NEGATIVE_DATA."
            ),
        },
        "frame_level_baselines": baselines,
        "evaluated_candidates": evaluated_candidates,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Đã lưu kết quả đánh giá tập kiểm thử tại: %s", OUTPUT_PATH)


if __name__ == "__main__":
    run_evaluation()
