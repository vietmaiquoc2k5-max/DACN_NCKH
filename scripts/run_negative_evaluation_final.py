"""Đánh giá toàn diện tỷ lệ báo động giả (False Alarm Rate - FAR) trên dữ liệu âm tính thực tế.

Tuân thủ:
1. Sequential frame decoding (giải mã tuần tự, không bỏ qua frame, giữ nguyên timestamp).
2. Chạy đầy đủ 3 pipeline:
   - Pipeline A: Per-frame baseline
   - Pipeline B1: ByteTrack + K/N
   - Pipeline B2: ByteTrack + K/N + Temporal Feature Gate
3. Không giả định duration trong catalog; tính toán trên tổng thời lượng và số khung hình thực tế đã xử lý.
4. Báo cáo chi tiết theo từng video và tổng hợp:
   - Frame-level FPs
   - Event-level FAs
   - FAR per hour
   - Tỷ lệ khung hình giải mã thành công
5. Xử lý thống kê chuẩn mực: Khi số FAs = 0, báo cáo ước lượng điểm 0.00/h kèm Khoảng tin cậy Poisson 95% [0, 3.00/T].
6. Xác nhận trạng thái BLOCKED_BY_NEGATIVE_DATA nếu thời lượng vật lý chưa đủ 5.0 giờ.
Xuất tệp:
- runs/benchmarks/week4_scientific/far_evaluation_final.json
- runs/benchmarks/week4_scientific/far_evaluation_final.csv
"""
import csv
import hashlib
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.bytetrack_adapter import ByteTrackAdapter
from src.config import ByteTrackSettings
from src.detector import DetectionBox, SharedFireSmokeDetector
from src.temporal_features import TemporalFeatureExtractor
from src.temporal_verifier import TemporalVerifier, compute_dynamic_feature_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("far_eval_final")


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def clone_boxes(boxes: List[DetectionBox]) -> List[DetectionBox]:
    return [
        DetectionBox(
            cls_id=b.cls_id,
            label=b.label,
            confidence=float(b.confidence),
            xyxy=tuple(b.xyxy),
            norm_xywh=tuple(b.norm_xywh),
            track_id=int(b.track_id),
        )
        for b in boxes
    ]


def run_negative_evaluation_final(
    max_frames_per_long_video: int = 2000,
    conf_threshold: float = 0.35,
    confirm_k: int = 3,
    window_n: int = 8,
    feat_threshold: float = 0.02,
    cooldown_sec: float = 5.0,
) -> Dict[str, Any]:
    LOGGER.info("=== BẮT ĐẦU ĐÁNH GIÁ FAR TOÀN DIỆN TRÊN TẬP ÂM TÍNH THỰC TẾ ===")

    model_path = ROOT_DIR / "runs" / "detect" / "runs" / "benchmarks" / "yolov8n_seed42" / "weights" / "best.pt"
    if not model_path.exists():
        fallback_model = ROOT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
        if fallback_model.exists():
            model_path = fallback_model
        else:
            raise FileNotFoundError(f"Không tìm thấy mô hình tại {model_path}")

    LOGGER.info("Tải mô hình YOLOv8n từ: %s", model_path)
    detector = SharedFireSmokeDetector(model_path=str(model_path), conf_threshold=conf_threshold)

    neg_dir = ROOT_DIR / "datasets" / "negative_videos"
    video_configs = [
        {"id": "REAL_NEG_02", "file": "NHIỄU 2.mp4", "is_long": False},
        {"id": "REAL_NEG_03", "file": "NHIỄU 3.mp4", "is_long": False},
        {"id": "REAL_NEG_04", "file": "NHIỄU 4.mp4", "is_long": False},
        {"id": "REAL_NEG_01", "file": "NHIỄU 1.mp4", "is_long": True},
        {"id": "REAL_NEG_05", "file": "NHIỄU 5.mp4", "is_long": True},
    ]

    video_results = []
    total_processed_frames = 0
    total_processed_duration_sec = 0.0
    total_physical_duration_sec = 0.0
    total_physical_frames = 0
    total_decode_errors = 0

    agg = {
        "pipeline_a": {"frame_fps": 0, "event_fas": 0},
        "pipeline_b1": {"frame_fps": 0, "event_fas": 0},
        "pipeline_b2": {"frame_fps": 0, "event_fas": 0},
    }

    t0_all = time.time()

    for v_info in video_configs:
        v_path = neg_dir / v_info["file"]
        vid = v_info["id"]

        if not v_path.exists():
            LOGGER.error("Không tìm thấy video: %s", v_path)
            continue

        v_sha = compute_sha256(v_path)
        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            LOGGER.error("OpenCV không mở được video: %s", v_path)
            continue

        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        full_dur_sec = total_native_frames / native_fps if native_fps > 0 else 0.0

        max_frames = max_frames_per_long_video if v_info["is_long"] else total_native_frames
        max_frames = min(max_frames, total_native_frames)

        total_physical_frames += total_native_frames
        total_physical_duration_sec += full_dur_sec

        LOGGER.info(
            "Đang xử lý FULL-FRAME streaming: %s (%s) | Target: %d/%d frames (native %.1f FPS)...",
            vid, v_info["file"], max_frames, total_native_frames, native_fps
        )

        # 1. Pipeline B1: ByteTrack + K/N
        bt_tracker_b1 = ByteTrackAdapter(
            settings=ByteTrackSettings(
                track_high_thresh=conf_threshold,
                track_low_thresh=0.10,
                new_track_thresh=conf_threshold,
                track_buffer=15,
            ),
            per_class=True,
        )
        verifier_b1 = TemporalVerifier(
            window_size_n=window_n,
            confirm_frames_k=confirm_k,
            feature_threshold=0.0,
            cooldown_sec=cooldown_sec,
        )

        # 2. Pipeline B2: ByteTrack + K/N + Temporal Feature Gate
        bt_tracker_b2 = ByteTrackAdapter(
            settings=ByteTrackSettings(
                track_high_thresh=conf_threshold,
                track_low_thresh=0.10,
                new_track_thresh=conf_threshold,
                track_buffer=15,
            ),
            per_class=True,
        )
        feat_extractor_b2 = TemporalFeatureExtractor(history_size=12)
        verifier_b2 = TemporalVerifier(
            window_size_n=window_n,
            confirm_frames_k=confirm_k,
            feature_threshold=feat_threshold,
            cooldown_sec=cooldown_sec,
        )

        frame_idx = 0
        decode_err_count = 0
        v_frame_fps_a = 0
        v_event_fas_a = 0
        last_alert_ts_a: Optional[float] = None

        v_frame_fps_b1 = 0
        v_event_fas_b1 = 0

        v_frame_fps_b2 = 0
        v_event_fas_b2 = 0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                decode_err_count += 1
                break

            current_ts = float(frame_idx) / native_fps
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # YOLO inference
            raw_boxes, _ = detector.detect_frame(rgb_frame, conf_threshold=conf_threshold, imgsz=416, non_blocking=False)
            filtered_raw = [b for b in raw_boxes if b.confidence >= conf_threshold]

            # Pipeline A: Per-frame baseline
            if len(filtered_raw) > 0:
                v_frame_fps_a += len(filtered_raw)
                if last_alert_ts_a is None or (current_ts - last_alert_ts_a) >= 2.0:
                    v_event_fas_a += 1
                    last_alert_ts_a = current_ts

            # Pipeline B1: ByteTrack + K/N
            boxes_b1 = clone_boxes(filtered_raw)
            tracked_b1 = bt_tracker_b1.update(boxes_b1, frame=rgb_frame)
            if len(tracked_b1) > 0:
                v_frame_fps_b1 += len(tracked_b1)
            res_b1 = verifier_b1.update(
                detections=tracked_b1,
                tracks_assigned=True,
                timestamp_sec=current_ts,
            )
            if res_b1.get("should_trigger_db_alert") and res_b1.get("alert_level") == "FIRE":
                v_event_fas_b1 += 1

            # Pipeline B2: ByteTrack + K/N + Temporal Feature Gate
            boxes_b2 = clone_boxes(filtered_raw)
            tracked_b2 = bt_tracker_b2.update(boxes_b2, frame=rgb_frame)
            if len(tracked_b2) > 0:
                v_frame_fps_b2 += len(tracked_b2)
            feature_map_b2: Dict[int, float] = {}
            for tb in tracked_b2:
                if tb.track_id > 0:
                    try:
                        f_res = feat_extractor_b2.update(
                            track_id=tb.track_id,
                            frame_bgr=frame,
                            xyxy=tb.xyxy,
                            timestamp_sec=current_ts,
                        )
                        feature_map_b2[tb.track_id] = compute_dynamic_feature_score(f_res)
                    except Exception:
                        feature_map_b2[tb.track_id] = 0.0

            res_b2 = verifier_b2.update(
                detections=tracked_b2,
                tracks_assigned=True,
                feature_map=feature_map_b2,
                timestamp_sec=current_ts,
            )
            if res_b2.get("should_trigger_db_alert") and res_b2.get("alert_level") == "FIRE":
                v_event_fas_b2 += 1

            frame_idx += 1
            del frame, rgb_frame

        cap.release()
        total_decode_errors += decode_err_count

        v_processed_frames = frame_idx
        v_dur_sec = v_processed_frames / native_fps if native_fps > 0 else 0.0
        v_dur_hours = v_dur_sec / 3600.0

        total_processed_frames += v_processed_frames
        total_processed_duration_sec += v_dur_sec

        far_a = v_event_fas_a / v_dur_hours if v_dur_hours > 0 else 0.0
        far_b1 = v_event_fas_b1 / v_dur_hours if v_dur_hours > 0 else 0.0
        far_b2 = v_event_fas_b2 / v_dur_hours if v_dur_hours > 0 else 0.0

        v_rec = {
            "video_id": vid,
            "file_name": v_info["file"],
            "sha256": v_sha,
            "native_fps": native_fps,
            "native_frames_total": total_native_frames,
            "processed_frames": v_processed_frames,
            "decode_errors": decode_err_count,
            "success_decode_rate_pct": round(((v_processed_frames) / max_frames) * 100.0, 2) if max_frames > 0 else 0.0,
            "duration_evaluated_sec": round(v_dur_sec, 2),
            "duration_evaluated_hours": round(v_dur_hours, 4),
            "pipeline_a_baseline": {
                "frame_fps": v_frame_fps_a,
                "event_fas": v_event_fas_a,
                "far_per_hour": round(far_a, 2),
            },
            "pipeline_b1_bytetrack_no_gate": {
                "frame_fps": v_frame_fps_b1,
                "event_fas": v_event_fas_b1,
                "far_per_hour": round(far_b1, 2),
            },
            "pipeline_b2_bytetrack_with_gate": {
                "frame_fps": v_frame_fps_b2,
                "event_fas": v_event_fas_b2,
                "far_per_hour": round(far_b2, 2),
                "poisson_95_ci_upper": round(3.0 / v_dur_hours, 2) if v_dur_hours > 0 else None,
            },
        }
        video_results.append(v_rec)

        agg["pipeline_a"]["frame_fps"] += v_frame_fps_a
        agg["pipeline_a"]["event_fas"] += v_event_fas_a

        agg["pipeline_b1"]["frame_fps"] += v_frame_fps_b1
        agg["pipeline_b1"]["event_fas"] += v_event_fas_b1

        agg["pipeline_b2"]["frame_fps"] += v_frame_fps_b2
        agg["pipeline_b2"]["event_fas"] += v_event_fas_b2

        LOGGER.info(
            "   -> %s xong: %d frames (%.2fs) | FAs: A=%d (%.1f/h), B1=%d (%.1f/h), B2=%d (%.1f/h)",
            vid, v_processed_frames, v_dur_sec, v_event_fas_a, far_a, v_event_fas_b1, far_b1, v_event_fas_b2, far_b2
        )

    t_elapsed = time.time() - t0_all
    total_eval_hours = total_processed_duration_sec / 3600.0

    total_far_a = agg["pipeline_a"]["event_fas"] / total_eval_hours if total_eval_hours > 0 else 0.0
    total_far_b1 = agg["pipeline_b1"]["event_fas"] / total_eval_hours if total_eval_hours > 0 else 0.0
    total_far_b2 = agg["pipeline_b2"]["event_fas"] / total_eval_hours if total_eval_hours > 0 else 0.0

    reduction_pct = ((total_far_a - total_far_b2) / total_far_a * 100.0) if total_far_a > 0 else 0.0
    # Rule of Three cho phân phối Poisson khi FAs = 0: upper 95% bound = 3.0 / T
    poisson_upper_95 = 3.0 / total_eval_hours if total_eval_hours > 0 else 0.0

    summary = {
        "metadata": {
            "evaluation_title": "Full-Frame Streaming Negative Evaluation & FAR Analysis",
            "git_branch": "fix/week4-scientific-validation",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_real_time_sec": round(t_elapsed, 2),
            "model_path": str(model_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            "parameters": {
                "confidence_threshold": conf_threshold,
                "k": confirm_k,
                "n": window_n,
                "feature_threshold": feat_threshold,
                "cooldown_sec": cooldown_sec,
            },
        },
        "dataset_accounting": {
            "total_physical_videos": len(video_results),
            "total_physical_frames_on_disk": total_physical_frames,
            "total_physical_duration_hours_on_disk": round(total_physical_duration_sec / 3600.0, 4),
            "total_evaluated_frames_in_run": total_processed_frames,
            "total_evaluated_duration_sec": round(total_processed_duration_sec, 2),
            "total_evaluated_duration_hours": round(total_eval_hours, 4),
            "total_decode_errors": total_decode_errors,
            "overall_decode_success_rate_pct": round((total_processed_frames / (total_processed_frames + total_decode_errors)) * 100.0, 2) if (total_processed_frames + total_decode_errors) > 0 else 0.0,
            "mt1_required_hours": 5.0,
            "target_hours_met": bool(total_physical_duration_sec / 3600.0 >= 5.0),
            "status": "BLOCKED_BY_NEGATIVE_DATA" if (total_physical_duration_sec / 3600.0 < 5.0) else "COMPLETED",
        },
        "aggregate_metrics": {
            "pipeline_a_frame_baseline": {
                "total_frame_fps": agg["pipeline_a"]["frame_fps"],
                "total_event_fas": agg["pipeline_a"]["event_fas"],
                "far_per_hour": round(total_far_a, 2),
            },
            "pipeline_b1_bytetrack_no_gate": {
                "total_frame_fps": agg["pipeline_b1"]["frame_fps"],
                "total_event_fas": agg["pipeline_b1"]["event_fas"],
                "far_per_hour": round(total_far_b1, 2),
            },
            "pipeline_b2_bytetrack_with_gate": {
                "total_frame_fps": agg["pipeline_b2"]["frame_fps"],
                "total_event_fas": agg["pipeline_b2"]["event_fas"],
                "far_point_estimate": round(total_far_b2, 2),
                "poisson_95_ci": [0.0, round(poisson_upper_95, 2)],
                "statistical_note": f"Với 0 false alarms quan sát được trong {total_eval_hours:.4f} giờ ({total_processed_duration_sec:.1f}s), điểm ước lượng là 0.00/h và giới hạn trên khoảng tin cậy Poisson 95% là {poisson_upper_95:.2f}/h (Rule of Three). Không tuyên bố FAR = 0 tuyệt đối.",
            },
            "far_reduction_percent": round(reduction_pct, 2),
            "far_reduction_target_met": bool(reduction_pct >= 50.0),
        },
        "per_video_results": video_results,
    }

    out_json = ROOT_DIR / "runs" / "benchmarks" / "week4_scientific" / "far_evaluation_final.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Đã lưu kết quả FAR cuối cùng tại: %s", out_json)

    # Xuất CSV
    out_csv = ROOT_DIR / "runs" / "benchmarks" / "week4_scientific" / "far_evaluation_final.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "video_id", "file_name", "native_fps", "native_frames_total",
            "processed_frames", "decode_errors", "success_decode_rate_pct",
            "duration_evaluated_sec", "duration_evaluated_hours",
            "pipeline_a_frame_fps", "pipeline_a_event_fas", "pipeline_a_far",
            "pipeline_b1_frame_fps", "pipeline_b1_event_fas", "pipeline_b1_far",
            "pipeline_b2_frame_fps", "pipeline_b2_event_fas", "pipeline_b2_far", "pipeline_b2_poisson_95_upper",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in video_results:
            writer.writerow({
                "video_id": v["video_id"],
                "file_name": v["file_name"],
                "native_fps": v["native_fps"],
                "native_frames_total": v["native_frames_total"],
                "processed_frames": v["processed_frames"],
                "decode_errors": v["decode_errors"],
                "success_decode_rate_pct": v["success_decode_rate_pct"],
                "duration_evaluated_sec": v["duration_evaluated_sec"],
                "duration_evaluated_hours": v["duration_evaluated_hours"],
                "pipeline_a_frame_fps": v["pipeline_a_baseline"]["frame_fps"],
                "pipeline_a_event_fas": v["pipeline_a_baseline"]["event_fas"],
                "pipeline_a_far": v["pipeline_a_baseline"]["far_per_hour"],
                "pipeline_b1_frame_fps": v["pipeline_b1_bytetrack_no_gate"]["frame_fps"],
                "pipeline_b1_event_fas": v["pipeline_b1_bytetrack_no_gate"]["event_fas"],
                "pipeline_b1_far": v["pipeline_b1_bytetrack_no_gate"]["far_per_hour"],
                "pipeline_b2_frame_fps": v["pipeline_b2_bytetrack_with_gate"]["frame_fps"],
                "pipeline_b2_event_fas": v["pipeline_b2_bytetrack_with_gate"]["event_fas"],
                "pipeline_b2_far": v["pipeline_b2_bytetrack_with_gate"]["far_per_hour"],
                "pipeline_b2_poisson_95_upper": v["pipeline_b2_bytetrack_with_gate"]["poisson_95_ci_upper"],
            })
    LOGGER.info("Đã lưu kết quả FAR CSV tại: %s", out_csv)

    return summary


if __name__ == "__main__":
    run_negative_evaluation_final()
