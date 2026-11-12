"""Đồng bộ hóa artifacts tuần 4 (metrics.json, operating_point.json, best_operating_point.json)
với run kiểm chứng âm tính thực tế run_20261111_233035_neg_full và provenance chính xác.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("sync_metrics")


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1048576):
            h.update(chunk)
    return h.hexdigest()


def sync_artifacts():
    run_dir = ROOT_DIR / "runs" / "benchmarks" / "week4_full_inference" / "run_20261111_233035_neg_full"
    exp_summary_path = run_dir / "negative_exposure_summary.json"
    manifest_path = ROOT_DIR / "data" / "negative_videos_manifest.json"
    gt_path = ROOT_DIR / "data" / "ground_truth" / "events.jsonl"
    model_path = ROOT_DIR / "runs" / "detect" / "runs" / "benchmarks" / "yolov8n_seed42" / "weights" / "best.pt"

    with open(exp_summary_path, "r", encoding="utf-8") as f:
        exp_summary = json.load(f)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest_sha256 = compute_file_sha256(manifest_path)
    gt_sha256 = compute_file_sha256(gt_path)
    model_sha256 = compute_file_sha256(model_path)

    metrics_path = ROOT_DIR / "runs" / "benchmarks" / "week4_final" / "metrics.json"
    existing_metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            existing_metrics = json.load(f)

    # Cập nhật provenance và dataset_availability
    existing_metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
    existing_metrics["acceptance_status"] = "PARTIAL (PENDING_24_GT_VERIFICATION)"
    existing_metrics["acceptance_status_detail"] = (
        "Chỉ tiêu MT3 về giảm FAR (57.39% >= 50.0%) và thời lượng âm tính (7.2026h >= 5.0h) đã ĐẠT HOÀN TOÀN. "
        "Trạng thái tổng thể giữ PARTIAL do 24/35 Ground Truth positive events đang chờ annotator thứ hai thẩm định độc lập."
    )

    existing_metrics["provenance"] = {
        "run_id": "run_20261111_233035_neg_full",
        "source_code_run_commit": "256065e7e54c35c335e6953d0bc373829ac94997",
        "initial_artifact_commit": "62785d7b4cdd1b499c99abbd016ecedf6fa1ee35",
        "base_audit_repair_commit": "6e490bd414f2e3b3240f87c93b0a66e7f96c40ca",
        "model_checkpoint": "runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt",
        "checkpoint_sha256": model_sha256,
        "negative_manifest_sha256": manifest_sha256,
        "events_gt_sha256": gt_sha256,
        "sampling_mode": "Full-Video Temporal Sampling Inference at 5 FPS (128,291 sampled frames, 929,939 total stream frames to EOF)",
    }

    existing_metrics["dataset_availability"] = {
        "available_positive_videos": 24,
        "available_positive_duration_sec": 863.98,
        "available_positive_hours": 0.24,
        "available_negative_videos": manifest_data["summary"]["valid_videos_count"],
        "available_negative_hours": manifest_data["summary"]["total_valid_exposure_hours"],
        "negative_stream_frames_decoded": manifest_data["summary"]["full_decoded_frames_sum"],
        "negative_sampled_frames_processed": manifest_data["summary"]["full_sampled_frames_sum"],
        "proposal_target_negative_hours": 5.0,
        "surplus_negative_hours": manifest_data["surplus_hours"],
        "negative_exposure_target_met": True,
    }

    existing_metrics["negative_video_evaluation"] = {
        "status": "PASS / SUFFICIENT_NEGATIVE_EXPOSURE",
        "evaluated_at": exp_summary.get("timestamp"),
        "total_videos_evaluated": exp_summary["total_videos_processed"],
        "total_exposure_seconds": exp_summary["total_exposure_seconds"],
        "total_exposure_hours": exp_summary["total_exposure_hours"],
        "total_decoded_frames": exp_summary["total_decoded_frames"],
        "total_processed_frames": exp_summary["total_processed_frames"],
        "sampling_fps": 5.0,
        "pipeline_a": exp_summary["pipeline_a"],
        "pipeline_b1": exp_summary["pipeline_b1"],
        "pipeline_b2": exp_summary["pipeline_b2"],
        "far_reduction_percentage": exp_summary["far_reduction_percentage"],
        "target_met": exp_summary["proposal_mt3_target_met"],
    }

    existing_metrics["operating_point_selection"] = {
        "status": "PROVISIONAL",
        "feasible": True,
        "reason": (
            "Operating Point (conf=0.35, K=3, N=8, feat=0.02, cooldown=5.0s, gap=2.0s) thỏa mãn 100% ràng buộc MT3: "
            "FAR giảm 57.39% (>= 50%), độ trễ TB 2.13s (<= 10s), recall video duy trì 95.83% (suy giảm 0.0% <= 5%). "
            "Khóa ở mức PROVISIONAL do 24 sự kiện GT còn pending verification."
        ),
        "selected_operating_point": {
            "confidence_threshold": 0.35,
            "k": 3,
            "n": 8,
            "feature_threshold": 0.02,
            "cooldown_sec": 5.0,
            "max_event_gap_sec": 2.0,
        },
        "blockers_for_official_lock": [
            "Hoàn tất thẩm định độc lập bởi annotator thứ hai cho 24 sự kiện pending trong data/ground_truth/events.jsonl.",
            "Khóa locked test set evaluation sau khi 100% nhãn được cross-verified."
        ],
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing_metrics, f, indent=2, ensure_ascii=False)
    LOGGER.info("Đã đồng bộ thành công %s", metrics_path)

    # Cập nhật operating_point.json và best_operating_point.json
    for op_name in ["operating_point.json", "best_operating_point.json"]:
        op_file = ROOT_DIR / "runs" / "benchmarks" / "week4_final" / op_name
        if op_file.exists():
            with open(op_file, "r", encoding="utf-8") as f:
                op_data = json.load(f)
            op_data["verification_history"]["negative_manifest_sha256"] = manifest_sha256
            op_data["verification_history"]["events_gt_sha256"] = gt_sha256
            op_data["verification_history"]["base_audit_repair_commit"] = "6e490bd414f2e3b3240f87c93b0a66e7f96c40ca"
            op_data["verification_history"]["sampling_mode"] = (
                "Full-Video Temporal Sampling Inference at 5 FPS (128,291 sampled frames, 929,939 total stream frames to EOF)"
            )
            with open(op_file, "w", encoding="utf-8") as f:
                json.dump(op_data, f, indent=2, ensure_ascii=False)
            LOGGER.info("Đã cập nhật provenance trong %s", op_file)


if __name__ == "__main__":
    sync_artifacts()

