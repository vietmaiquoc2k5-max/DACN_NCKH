import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.event_eval import (
    Event,
    FrameDetection,
    evaluate,
    extract_frame_level_alarms,
    frame_detections_to_events,
    get_ground_truth_manifest,
    load_frame_detections,
    load_ground_truth_events,
    load_jsonl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("eval_with_negatives")


def run_comprehensive_evaluation():
    LOGGER.info("=== BẮT ĐẦU ĐÁNH GIÁ THỰC NGHIỆM MT3 VỚI DỮ LIỆU ÂM TÍNH THỰC TẾ ===")

    out_dir = PROJECT_ROOT / "runs" / "benchmarks" / "week4_final"
    gt_path = PROJECT_ROOT / "data" / "ground_truth" / "events.jsonl"
    neg_manifest_path = PROJECT_ROOT / "data" / "negative_videos_manifest.json"

    # 1. Nạp Ground Truth
    gt_events = load_ground_truth_events(gt_path, only_verified=True, strict=True)
    gt_manifest = get_ground_truth_manifest(gt_path)
    LOGGER.info("Đã nạp %d Ground Truth verified events.", len(gt_events))

    # 2. Đọc manifest video âm tính
    with open(neg_manifest_path, "r", encoding="utf-8") as f:
        neg_manifest = json.load(f)

    val_neg_ids = {v["video_id"] for v in neg_manifest.get("videos", []) if v["split"] == "val"}
    test_neg_ids = {v["video_id"] for v in neg_manifest.get("videos", []) if v["split"] == "test"}

    val_neg_hours = neg_manifest.get("val_hours", 0.0)
    test_neg_hours = neg_manifest.get("test_hours", 0.0)
    total_neg_hours = neg_manifest.get("actual_hours", 0.0)

    LOGGER.info("Tập âm tính: Val=%s (%.2fh), Test=%s (%.2fh) | Tổng=%.2fh", val_neg_ids, val_neg_hours, test_neg_ids, test_neg_hours, total_neg_hours)

    # 3. Phân chia Video Dương tính Val & Test (12 Val / 12 Test)
    val_pos_ids = {"POS_01", "POS_03", "POS_05", "POS_07", "POS_09", "POS_11", "POS_13", "POS_15", "POS_17", "POS_19", "POS_21", "POS_23"}
    test_pos_ids = {"POS_02", "POS_04", "POS_06", "POS_08", "POS_10", "POS_12", "POS_14", "POS_16", "POS_18", "POS_20", "POS_22", "POS_24"}

    # 4. Hợp nhất frame detections: Dương tính + Âm tính
    pos_raw_path = out_dir / "frame_level_detections.jsonl"
    pos_trk_path = out_dir / "tracked_frame_detections.jsonl"
    neg_raw_path = out_dir / "negative_raw_detections.jsonl"
    neg_trk_path = out_dir / "negative_tracked_detections.jsonl"

    all_raw_frames: List[FrameDetection] = []
    all_tracked_frames: List[FrameDetection] = []

    if pos_raw_path.exists():
        all_raw_frames.extend(load_frame_detections(pos_raw_path))
    if neg_raw_path.exists():
        all_raw_frames.extend(load_frame_detections(neg_raw_path))

    if pos_trk_path.exists():
        all_tracked_frames.extend(load_frame_detections(pos_trk_path))
    if neg_trk_path.exists():
        all_tracked_frames.extend(load_frame_detections(neg_trk_path))

    LOGGER.info("Tổng số khung hình nạp: Raw=%d, Tracked=%d", len(all_raw_frames), len(all_tracked_frames))

    # Tách theo split
    val_raw = [f for f in all_raw_frames if f.video_id in (val_pos_ids | val_neg_ids)]
    val_trk = [f for f in all_tracked_frames if f.video_id in (val_pos_ids | val_neg_ids)]
    test_raw = [f for f in all_raw_frames if f.video_id in (test_pos_ids | test_neg_ids)]
    test_trk = [f for f in all_tracked_frames if f.video_id in (test_pos_ids | test_neg_ids)]

    val_gt = [e for e in gt_events if e.video_id in val_pos_ids]
    test_gt = [e for e in gt_events if e.video_id in test_pos_ids]

    LOGGER.info("Split Val: %d raw frames, %d tracked frames, %d GT events", len(val_raw), len(val_trk), len(val_gt))
    LOGGER.info("Split Test: %d raw frames, %d tracked frames, %d GT events", len(test_raw), len(test_trk), len(test_gt))

    # 5. Đánh giá Pipeline A (Frame Baseline) và Pipeline B1 (ByteTrack K/N) trên Test Set
    frame_events_test, events_b1_test = frame_detections_to_events(
        frames=test_trk,
        confidence_threshold=0.35,
        window_n=12,
        confirm_k=5,
        max_event_gap_sec=2.0,
        feature_threshold=0.0,
        cooldown_sec=0.0,
    )

    eval_frame_baseline = evaluate(
        ground_truth=test_gt,
        predictions=frame_events_test,
        positive_video_ids=test_pos_ids,
        negative_video_ids=test_neg_ids,
        negative_hours=test_neg_hours,
        tolerance_sec=10.0,
        allow_missing_negative=False,
    )

    eval_b1 = evaluate(
        ground_truth=test_gt,
        predictions=events_b1_test,
        positive_video_ids=test_pos_ids,
        negative_video_ids=test_neg_ids,
        negative_hours=test_neg_hours,
        tolerance_sec=10.0,
        allow_missing_negative=False,
    )

    # 6. Đánh giá Pipeline B2 (ByteTrack + K/N + Temporal Feature Gate + Cooldown) trên Test Set
    _, events_b2_test = frame_detections_to_events(
        frames=test_trk,
        confidence_threshold=0.35,
        window_n=8,
        confirm_k=3,
        max_event_gap_sec=2.0,
        feature_threshold=0.02,
        cooldown_sec=5.0,
    )

    eval_b2 = evaluate(
        ground_truth=test_gt,
        predictions=events_b2_test,
        positive_video_ids=test_pos_ids,
        negative_video_ids=test_neg_ids,
        negative_hours=test_neg_hours,
        tolerance_sec=10.0,
        allow_missing_negative=False,
    )

    LOGGER.info("=== KẾT QUẢ THỰC NGHIỆM TRÊN LOCKED TEST SET ===")
    LOGGER.info("Pipeline A (Frame Baseline): Video Recall=%.1f%%, FAR=%.2f/h, Delay=%.2fs",
                eval_frame_baseline["metrics"]["video_event_recall"]*100,
                eval_frame_baseline["metrics"]["false_alarms_per_hour"],
                eval_frame_baseline["metrics"]["mean_detection_delay_sec"])
    LOGGER.info("Pipeline B1 (ByteTrack K/N): Video Recall=%.1f%%, FAR=%.2f/h, Delay=%.2fs",
                eval_b1["metrics"]["video_event_recall"]*100,
                eval_b1["metrics"]["false_alarms_per_hour"],
                eval_b1["metrics"]["mean_detection_delay_sec"])
    LOGGER.info("Pipeline B2 (ByteTrack + Feature Gate): Video Recall=%.1f%%, FAR=%.2f/h, Delay=%.2fs",
                eval_b2["metrics"]["video_event_recall"]*100,
                eval_b2["metrics"]["false_alarms_per_hour"],
                eval_b2["metrics"]["mean_detection_delay_sec"])

    # Tính tỷ lệ giảm FAR và Recall Drop
    far_base = eval_frame_baseline["metrics"]["false_alarms_per_hour"]
    far_b2 = eval_b2["metrics"]["false_alarms_per_hour"]
    far_reduction = ((far_base - far_b2) / far_base * 100.0) if far_base > 0 else 0.0

    rec_base = eval_frame_baseline["metrics"]["video_event_recall"] * 100.0
    rec_b2 = eval_b2["metrics"]["video_event_recall"] * 100.0
    rec_drop = max(0.0, rec_base - rec_b2)
    delay_b2 = eval_b2["metrics"]["mean_detection_delay_sec"]

    mt3_far_met = far_reduction >= 50.0
    mt3_rec_met = rec_drop <= 5.0
    mt3_delay_met = delay_b2 <= 10.0
    mt3_hours_met = total_neg_hours >= 5.0

    decision = {
        "metadata": {
            "evaluation_name": "Reconciled Locked Test Evaluation with Physical Negative Data",
            "git_branch": "fix",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "total_negative_hours_available": total_neg_hours,
            "required_negative_hours": 5.0,
            "missing_negative_hours": round(max(0.0, 5.0 - total_neg_hours), 4),
            "status": "PARTIAL / BLOCKED_BY_NEGATIVE_DATA" if not mt3_hours_met else "FEASIBLE",
            "scientific_rationale": "Theo đề cương NCKH MT3, hệ thống yêu cầu tối thiểu 5.0 giờ video âm tính. Hiện tại ổ đĩa đã có 3.4263 giờ video âm tính thực tế (đạt 68.5% chỉ tiêu), còn thiếu 1.5737 giờ để nghiệm thu toàn phần.",
            "ground_truth_verified_events": len(gt_events),
        },
        "mt3_criteria_check": {
            "far_reduction_percent": round(far_reduction, 2),
            "far_reduction_target_met": mt3_far_met,
            "event_recall_drop_percent": round(rec_drop, 2),
            "recall_drop_target_met": mt3_rec_met,
            "mean_detection_delay_sec": round(delay_b2, 2),
            "delay_target_met": mt3_delay_met,
            "negative_duration_hours": round(total_neg_hours, 2),
            "duration_target_met": mt3_hours_met,
        },
        "pipelines_comparison": {
            "pipeline_a_frame_baseline": eval_frame_baseline["metrics"],
            "pipeline_b1_bytetrack_no_gate": eval_b1["metrics"],
            "pipeline_b2_bytetrack_with_gate": eval_b2["metrics"],
        },
    }

    eval_out_file = out_dir / "reconciled_test_evaluation.json"
    with open(eval_out_file, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)
    LOGGER.info("Kết quả đánh giá đã lưu tại: %s", eval_out_file)


if __name__ == "__main__":
    run_comprehensive_evaluation()
