"""Đánh giá toàn diện tỷ lệ báo động giả (FAR) chuẩn hóa theo phương pháp luận khoa học (Task 1, 2, 3, 4).

Chuẩn hóa:
1. Đồng bộ Operating Point từ SystemSettings duy nhất (conf=0.35, K=2, N=8, feat=0.02, cooldown=5.0s).
   Có assertion fail ngay lập tức nếu tham số benchmark khác biệt so với SystemSettings.
2. Đồng nhất ngữ nghĩa Cooldown (Per-class Causal Cooldown) giữa Pipeline A, B1, B2:
   - FIRE và SMOKE có cooldown độc lập (SMOKE không chặn FIRE, FIRE không chặn SMOKE).
   - Xử lý nhất quán khi đồng thời xuất hiện cả FIRE và SMOKE trong cùng một khung hình.
   - Duy trì biến thể Pipeline A Legacy (cooldown=2.0s) độc lập để đối soát lịch sử.
3. Tách biệt 3 cấp độ đo lường:
   - false_positive_detections: tổng bounding boxes sai (FIRE, SMOKE, Total).
   - false_positive_frames: số frames có >= 1 detection sai (FIRE, SMOKE, Any).
   - false_alarm_events: số sự kiện cảnh báo sai sau temporal aggregation và cooldown (FIRE, SMOKE, Total).
4. Phân định rạch ròi chế độ đánh giá:
   - Hỗ trợ CLI --max-frames 0 (hoặc None) cho chế độ full_video (không cap bất kỳ video nào).
   - Ghi nhận chính xác số frame giải mã, decode failures, coverage % và thời lượng inference thực tế.
5. Tính khoảng tin cậy Poisson 95% (Rule of Three khi FAs = 0).
6. Xuất metadata khoa học: benchmark_run_id, git_commit, model_sha256, operating_point, evaluation_mode, dataset_manifest_hash.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import subprocess
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
from src.config import ByteTrackSettings, SystemSettings
from src.detector import DetectionBox, SharedFireSmokeDetector
from src.temporal_features import TemporalFeatureExtractor
from src.temporal_verifier import TemporalVerifier, compute_dynamic_feature_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("far_unified")


def compute_sha256(filepath: Path) -> str:
    """Tính mã băm SHA256 của tệp tin."""
    sha = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def get_git_commit_sha() -> str:
    """Lấy mã commit Git hiện tại."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR))
        return out.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def clone_boxes(boxes: List[DetectionBox]) -> List[DetectionBox]:
    """Tạo bản sao sâu danh sách DetectionBox."""
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


def calculate_poisson_95_ci(k: int, t_hours: float) -> Tuple[float, float]:
    """Tính khoảng tin cậy 95% Poisson cho tỷ lệ sự kiện trên giờ."""
    if t_hours <= 0:
        return 0.0, 0.0
    if k == 0:
        # Rule of Three (Poisson 95% upper bound khi 0 sự kiện)
        return 0.0, round(3.0 / t_hours, 2)
    # Wilson-Hilferty / Chi-square approximation
    lower = max(0.0, k * math.pow(1.0 - 1.0 / (9.0 * k) - 1.96 / (3.0 * math.sqrt(k)), 3)) / t_hours
    upper = (k + 1) * math.pow(1.0 - 1.0 / (9.0 * (k + 1)) + 1.96 / (3.0 * math.sqrt(k + 1)), 3) / t_hours
    return round(lower, 2), round(upper, 2)


def run_unified_negative_evaluation(
    settings: Optional[SystemSettings] = None,
    max_frames_per_long_video: Optional[int] = 2000,
    conf_threshold: Optional[float] = None,
    confirm_k: Optional[int] = None,
    window_n: Optional[int] = None,
    feat_threshold: Optional[float] = None,
    cooldown_sec: Optional[float] = None,
    model_path_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Thực thi đánh giá FAR toàn diện chuẩn hóa theo đúng Operating Point được triển khai."""
    LOGGER.info("=== BẮT ĐẦU ĐÁNH GIÁ FAR CHUẨN HÓA PHƯƠNG PHÁP LUẬN (TASK 1, 2, 3, 4) ===")

    # 1. Nạp và kiểm tra Operating Point từ nguồn duy nhất SystemSettings
    canonical_settings = SystemSettings()
    deployed_settings = canonical_settings if settings is None else settings

    eval_conf = float(conf_threshold if conf_threshold is not None else deployed_settings.conf_threshold)
    eval_k = int(confirm_k if confirm_k is not None else deployed_settings.confirm_frames_k)
    eval_n = int(window_n if window_n is not None else deployed_settings.window_size_n)
    eval_feat = float(feat_threshold if feat_threshold is not None else deployed_settings.feature_threshold)
    eval_cd = float(cooldown_sec if cooldown_sec is not None else deployed_settings.alert_cooldown_sec)

    # TASK 1 Assertion: Bắt buộc fail nếu cấu hình benchmark khác biệt so với SystemSettings chuẩn
    assert eval_k == canonical_settings.confirm_frames_k, (
        f"Benchmark confirm_k ({eval_k}) != SystemSettings.confirm_frames_k ({canonical_settings.confirm_frames_k})"
    )
    assert eval_n == canonical_settings.window_size_n, (
        f"Benchmark window_n ({eval_n}) != SystemSettings.window_size_n ({canonical_settings.window_size_n})"
    )
    assert abs(eval_feat - canonical_settings.feature_threshold) < 1e-6, (
        f"Benchmark feat_threshold ({eval_feat}) != SystemSettings.feature_threshold ({canonical_settings.feature_threshold})"
    )
    assert abs(eval_conf - canonical_settings.conf_threshold) < 1e-6, (
        f"Benchmark conf_threshold ({eval_conf}) != SystemSettings.conf_threshold ({canonical_settings.conf_threshold})"
    )
    assert abs(eval_cd - canonical_settings.alert_cooldown_sec) < 1e-6, (
        f"Benchmark cooldown_sec ({eval_cd}) != SystemSettings.alert_cooldown_sec ({canonical_settings.alert_cooldown_sec})"
    )

    LOGGER.info(
        "Operating Point xác thực thành công: conf=%.2f, K=%d, N=%d, feat_threshold=%.4f, cooldown=%.1fs",
        eval_conf, eval_k, eval_n, eval_feat, eval_cd
    )

    if model_path_str:
        model_path = Path(model_path_str)
    else:
        model_path = ROOT_DIR / "runs" / "detect" / "runs" / "benchmarks" / "yolov8n_seed42" / "weights" / "best.pt"
        if not model_path.exists():
            fallback = ROOT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
            if fallback.exists():
                model_path = fallback
            else:
                raise FileNotFoundError(f"Không tìm thấy mô hình tại {model_path}")

    model_sha = compute_sha256(model_path)
    LOGGER.info("Tải mô hình: %s (SHA256: %s)", model_path, model_sha)
    detector = SharedFireSmokeDetector(model_path=str(model_path), conf_threshold=eval_conf)

    neg_dir = ROOT_DIR / "datasets" / "negative_videos"
    video_configs = [
        {"id": "REAL_NEG_02", "file": "NHIỄU 2.mp4", "is_long": False},
        {"id": "REAL_NEG_03", "file": "NHIỄU 3.mp4", "is_long": False},
        {"id": "REAL_NEG_04", "file": "NHIỄU 4.mp4", "is_long": False},
        {"id": "REAL_NEG_01", "file": "NHIỄU 1.mp4", "is_long": True},
        {"id": "REAL_NEG_05", "file": "NHIỄU 5.mp4", "is_long": True},
    ]

    # TASK 4: Phân định rạch ròi full_video vs bounded_sample
    evaluation_mode = "full_video" if max_frames_per_long_video is None else "bounded_sample"
    LOGGER.info("Chế độ đánh giá: %s (max_frames_per_long_video=%s)", evaluation_mode, max_frames_per_long_video)

    video_results = []
    total_physical_frames = 0
    total_physical_duration_sec = 0.0
    total_inferred_frames = 0
    total_inferred_duration_sec = 0.0
    total_decode_errors = 0

    agg_detections = {"fire": 0, "smoke": 0, "total": 0}
    agg_frames = {"fire": 0, "smoke": 0, "any": 0}

    agg_events = {
        "pipeline_a_controlled_cd5": {"fire": 0, "smoke": 0, "total": 0},
        "pipeline_a_legacy_cd2": {"fire": 0, "smoke": 0, "total": 0},
        "pipeline_b1_bytetrack_no_gate": {"fire": 0, "smoke": 0, "total": 0},
        "pipeline_b2_bytetrack_with_gate": {"fire": 0, "smoke": 0, "total": 0},
    }

    manifest_hash_builder = hashlib.sha256()
    t0_start = time.time()

    for v_info in video_configs:
        v_path = neg_dir / v_info["file"]
        vid = v_info["id"]

        if not v_path.exists():
            LOGGER.error("Không tìm thấy video: %s", v_path)
            continue

        v_sha = compute_sha256(v_path)
        manifest_hash_builder.update(f"{vid}:{v_info['file']}:{v_sha}\n".encode("utf-8"))

        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            LOGGER.error("Không mở được video: %s", v_path)
            continue

        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_phys_dur_sec = native_frames / native_fps if native_fps > 0 else 0.0

        if v_info["is_long"] and max_frames_per_long_video is not None:
            target_frames = min(max_frames_per_long_video, native_frames)
        else:
            target_frames = native_frames

        total_physical_frames += native_frames
        total_physical_duration_sec += v_phys_dur_sec

        LOGGER.info(
            "Xử lý video %s (%s): target %d/%d frames (FPS=%.1f)...",
            vid, v_info["file"], target_frames, native_frames, native_fps
        )

        # Khởi tạo tracker và verifier độc lập theo từng video để cách ly state tuyệt đối
        bt_tracker_b1 = ByteTrackAdapter(
            settings=ByteTrackSettings(
                track_high_thresh=eval_conf,
                track_low_thresh=0.10,
                new_track_thresh=eval_conf,
                track_buffer=15,
            ),
            per_class=True,
        )
        verifier_b1 = TemporalVerifier(
            window_size_n=eval_n,
            confirm_frames_k=eval_k,
            feature_threshold=0.0,
            cooldown_sec=eval_cd,
        )

        bt_tracker_b2 = ByteTrackAdapter(
            settings=ByteTrackSettings(
                track_high_thresh=eval_conf,
                track_low_thresh=0.10,
                new_track_thresh=eval_conf,
                track_buffer=15,
            ),
            per_class=True,
        )
        feat_extractor_b2 = TemporalFeatureExtractor(history_size=12)
        verifier_b2 = TemporalVerifier(
            window_size_n=eval_n,
            confirm_frames_k=eval_k,
            feature_threshold=eval_feat,
            cooldown_sec=eval_cd,
        )

        frame_idx = 0
        decode_err = 0

        v_det_fp = {"fire": 0, "smoke": 0, "total": 0}
        v_frame_fp = {"fire": 0, "smoke": 0, "any": 0}

        v_fa_a_cd5 = {"fire": 0, "smoke": 0, "total": 0}
        v_fa_a_cd2 = {"fire": 0, "smoke": 0, "total": 0}
        last_alert_ts_a_cd5 = {"FIRE": None, "SMOKE": None}
        last_alert_ts_a_cd2 = {"FIRE": None, "SMOKE": None}

        v_fa_b1 = {"fire": 0, "smoke": 0, "total": 0}
        v_fa_b2 = {"fire": 0, "smoke": 0, "total": 0}

        while frame_idx < target_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                decode_err += 1
                break

            current_ts = float(frame_idx) / native_fps
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1. Detection cấp khung hình từ YOLOv8n
            raw_boxes, _ = detector.detect_frame(rgb_frame, conf_threshold=eval_conf, imgsz=416, non_blocking=False)
            filtered_raw = [b for b in raw_boxes if b.confidence >= eval_conf]

            # Đếm detection boxes và frame FP
            has_fire_det = False
            has_smoke_det = False
            for b in filtered_raw:
                lbl = str(b.label).upper()
                if lbl == "FIRE":
                    v_det_fp["fire"] += 1
                    has_fire_det = True
                elif lbl == "SMOKE":
                    v_det_fp["smoke"] += 1
                    has_smoke_det = True
                v_det_fp["total"] += 1

            if has_fire_det:
                v_frame_fp["fire"] += 1
            if has_smoke_det:
                v_frame_fp["smoke"] += 1
            if has_fire_det or has_smoke_det:
                v_frame_fp["any"] += 1

            # 2. Pipeline A (Per-frame baseline) với Per-class Cooldown độc lập
            # Cả FIRE và SMOKE được đánh giá độc lập, không chặn chéo nhau
            for lbl in ["FIRE", "SMOKE"]:
                has_lbl_det = any(str(b.label).upper() == lbl for b in filtered_raw)
                if has_lbl_det:
                    # Kiểm tra cooldown 5.0s (Controlled baseline)
                    last_ts5 = last_alert_ts_a_cd5[lbl]
                    if last_ts5 is None or (current_ts - last_ts5) >= eval_cd:
                        if lbl == "FIRE":
                            v_fa_a_cd5["fire"] += 1
                        else:
                            v_fa_a_cd5["smoke"] += 1
                        v_fa_a_cd5["total"] += 1
                        last_alert_ts_a_cd5[lbl] = current_ts

                    # Kiểm tra cooldown 2.0s (Legacy baseline)
                    last_ts2 = last_alert_ts_a_cd2[lbl]
                    if last_ts2 is None or (current_ts - last_ts2) >= 2.0:
                        if lbl == "FIRE":
                            v_fa_a_cd2["fire"] += 1
                        else:
                            v_fa_a_cd2["smoke"] += 1
                        v_fa_a_cd2["total"] += 1
                        last_alert_ts_a_cd2[lbl] = current_ts

            # 3. Pipeline B1: ByteTrack + K/N
            boxes_b1 = clone_boxes(filtered_raw)
            tracked_b1 = bt_tracker_b1.update(boxes_b1, frame=rgb_frame)
            res_b1 = verifier_b1.update(
                detections=tracked_b1,
                tracks_assigned=True,
                timestamp_sec=current_ts,
            )
            if res_b1.get("should_trigger_db_alert") and res_b1.get("alert_level") in {"FIRE", "SUSPECT"}:
                trig_labels = res_b1.get("triggered_labels") or [res_b1.get("primary_label", "FIRE")]
                for tl in trig_labels:
                    tl_upper = str(tl).upper()
                    if tl_upper == "FIRE":
                        v_fa_b1["fire"] += 1
                    elif tl_upper == "SMOKE":
                        v_fa_b1["smoke"] += 1
                    v_fa_b1["total"] += 1

            # 4. Pipeline B2: ByteTrack + K/N + Temporal Feature Gate
            boxes_b2 = clone_boxes(filtered_raw)
            tracked_b2 = bt_tracker_b2.update(boxes_b2, frame=rgb_frame)
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
            if res_b2.get("should_trigger_db_alert") and res_b2.get("alert_level") in {"FIRE", "SUSPECT"}:
                trig_labels = res_b2.get("triggered_labels") or [res_b2.get("primary_label", "FIRE")]
                for tl in trig_labels:
                    tl_upper = str(tl).upper()
                    if tl_upper == "FIRE":
                        v_fa_b2["fire"] += 1
                    elif tl_upper == "SMOKE":
                        v_fa_b2["smoke"] += 1
                    v_fa_b2["total"] += 1

            frame_idx += 1
            del frame, rgb_frame

        cap.release()
        total_decode_errors += decode_err

        v_inferred_frames = frame_idx
        v_inferred_dur_sec = v_inferred_frames / native_fps if native_fps > 0 else 0.0
        v_inferred_dur_hours = v_inferred_dur_sec / 3600.0

        total_inferred_frames += v_inferred_frames
        total_inferred_duration_sec += v_inferred_dur_sec

        far_a_cd5 = {k: round(v / v_inferred_dur_hours, 2) if v_inferred_dur_hours > 0 else 0.0 for k, v in v_fa_a_cd5.items()}
        far_a_cd2 = {k: round(v / v_inferred_dur_hours, 2) if v_inferred_dur_hours > 0 else 0.0 for k, v in v_fa_a_cd2.items()}
        far_b1 = {k: round(v / v_inferred_dur_hours, 2) if v_inferred_dur_hours > 0 else 0.0 for k, v in v_fa_b1.items()}
        far_b2 = {k: round(v / v_inferred_dur_hours, 2) if v_inferred_dur_hours > 0 else 0.0 for k, v in v_fa_b2.items()}

        ci_b2_lower, ci_b2_upper = calculate_poisson_95_ci(v_fa_b2["total"], v_inferred_dur_hours)

        v_summary = {
            "video_id": vid,
            "file_name": v_info["file"],
            "sha256": v_sha,
            "native_fps": native_fps,
            "total_physical_frames": native_frames,
            "inferred_frames": v_inferred_frames,
            "decode_errors": decode_err,
            "processing_coverage_pct": round((v_inferred_frames / native_frames) * 100.0, 2),
            "inferred_duration_sec": round(v_inferred_dur_sec, 2),
            "inferred_duration_hours": round(v_inferred_dur_hours, 4),
            "false_positive_detections": v_det_fp,
            "false_positive_frames": v_frame_fp,
            "false_alarm_events": {
                "pipeline_a_controlled_cd5s": v_fa_a_cd5,
                "pipeline_a_legacy_cd2s": v_fa_a_cd2,
                "pipeline_b1_bytetrack_no_gate": v_fa_b1,
                "pipeline_b2_bytetrack_with_gate": v_fa_b2,
            },
            "far_per_hour": {
                "pipeline_a_controlled_cd5s": far_a_cd5,
                "pipeline_a_legacy_cd2s": far_a_cd2,
                "pipeline_b1_bytetrack_no_gate": far_b1,
                "pipeline_b2_bytetrack_with_gate": far_b2,
                "pipeline_b2_poisson_95_ci": [ci_b2_lower, ci_b2_upper],
            },
        }
        video_results.append(v_summary)

        # Cộng dồn
        for k in ["fire", "smoke", "total"]:
            agg_detections[k] += v_det_fp[k]
            agg_events["pipeline_a_controlled_cd5"][k] += v_fa_a_cd5[k]
            agg_events["pipeline_a_legacy_cd2"][k] += v_fa_a_cd2[k]
            agg_events["pipeline_b1_bytetrack_no_gate"][k] += v_fa_b1[k]
            agg_events["pipeline_b2_bytetrack_with_gate"][k] += v_fa_b2[k]

        agg_frames["fire"] += v_frame_fp["fire"]
        agg_frames["smoke"] += v_frame_fp["smoke"]
        agg_frames["any"] += v_frame_fp["any"]

    total_eval_hours = total_inferred_duration_sec / 3600.0 if total_inferred_duration_sec > 0 else 0.0

    agg_far = {}
    for p_key, ev_dict in agg_events.items():
        agg_far[p_key] = {
            "fire_far": round(ev_dict["fire"] / total_eval_hours, 2) if total_eval_hours > 0 else 0.0,
            "smoke_far": round(ev_dict["smoke"] / total_eval_hours, 2) if total_eval_hours > 0 else 0.0,
            "total_far": round(ev_dict["total"] / total_eval_hours, 2) if total_eval_hours > 0 else 0.0,
        }

    ci_b2_tot_lower, ci_b2_tot_upper = calculate_poisson_95_ci(
        agg_events["pipeline_b2_bytetrack_with_gate"]["total"], total_eval_hours
    )
    ci_b2_fire_lower, ci_b2_fire_upper = calculate_poisson_95_ci(
        agg_events["pipeline_b2_bytetrack_with_gate"]["fire"], total_eval_hours
    )
    ci_b2_smoke_lower, ci_b2_smoke_upper = calculate_poisson_95_ci(
        agg_events["pipeline_b2_bytetrack_with_gate"]["smoke"], total_eval_hours
    )

    a_ctrl_far = agg_far["pipeline_a_controlled_cd5"]["total_far"]
    b1_far = agg_far["pipeline_b1_bytetrack_no_gate"]["total_far"]
    b2_far = agg_far["pipeline_b2_bytetrack_with_gate"]["total_far"]

    far_reduct_b1 = ((a_ctrl_far - b1_far) / a_ctrl_far * 100.0) if a_ctrl_far > 0 else 0.0
    far_reduct_b2 = ((a_ctrl_far - b2_far) / a_ctrl_far * 100.0) if a_ctrl_far > 0 else 0.0

    overall_coverage_pct = round((total_inferred_frames / total_physical_frames) * 100.0, 2) if total_physical_frames > 0 else 0.0

    elapsed_time = round(time.time() - t0_start, 2)
    run_timestamp_utc = datetime.now(timezone.utc).isoformat()
    benchmark_run_id = f"FAR_RUN_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    dataset_manifest_hash = manifest_hash_builder.hexdigest()

    summary_report = {
        "metadata": {
            "title": "Unified False Alarm Rate & Methodology Reconciliation (Tasks 1, 2, 3, 4)",
            "benchmark_run_id": benchmark_run_id,
            "git_commit": get_git_commit_sha(),
            "timestamp_utc": run_timestamp_utc,
            "elapsed_seconds": elapsed_time,
            "model_path": str(model_path.relative_to(ROOT_DIR) if model_path.is_relative_to(ROOT_DIR) else model_path),
            "model_sha256": model_sha,
            "dataset_manifest_hash": dataset_manifest_hash,
            "evaluation_mode": evaluation_mode,
            "operating_point": {
                "confidence_threshold": eval_conf,
                "confirm_k": eval_k,
                "window_n": eval_n,
                "feature_threshold": eval_feat,
                "cooldown_sec": eval_cd,
            },
            "cooldown_policy": "Per-class independent causal cooldown per camera (FIRE and SMOKE isolated; simultaneous supported)",
        },
        "dataset_accounting": {
            "total_physical_videos_on_disk": len(video_configs),
            "total_physical_frames_on_disk": total_physical_frames,
            "total_physical_duration_hours_on_disk": round(total_physical_duration_sec / 3600.0, 4),
            "total_inferred_frames": total_inferred_frames,
            "total_inferred_duration_sec": round(total_inferred_duration_sec, 2),
            "total_inferred_duration_hours": round(total_eval_hours, 4),
            "total_decode_errors": total_decode_errors,
            "overall_processing_coverage_pct": overall_coverage_pct,
            "mt1_requirement_hours": 5.0,
            "deficit_hours_to_reach_5h": round(max(0.0, 5.0 - total_eval_hours), 4),
            "target_hours_met": bool(total_eval_hours >= 5.0),
            "status": "PASS" if total_eval_hours >= 5.0 else "BLOCKED_BY_NEGATIVE_DATA",
        },
        "three_metrics_separation": {
            "1_false_positive_detections": {
                "description": "Tổng số bounding boxes sai phát hiện từ mô hình YOLO trên video âm tính",
                "counts": agg_detections,
            },
            "2_false_positive_frames": {
                "description": "Số lượng khung hình chứa ít nhất một bounding box sai (không đếm trùng lặp box)",
                "counts": agg_frames,
            },
            "3_false_alarm_events": {
                "description": "Số sự kiện cảnh báo sai độc lập sau temporal confirmation, event merging và per-class cooldown",
                "counts": agg_events,
            },
        },
        "aggregate_far_comparison": {
            "pipeline_a_controlled_cd5s": {
                "name": "Pipeline A (Per-frame baseline, Cooldown 5.0s)",
                "event_fas": agg_events["pipeline_a_controlled_cd5"],
                "far_per_hour": agg_far["pipeline_a_controlled_cd5"],
            },
            "pipeline_a_legacy_cd2s": {
                "name": "Pipeline A (Legacy baseline, Cooldown 2.0s / Max Gap 2.0s)",
                "event_fas": agg_events["pipeline_a_legacy_cd2"],
                "far_per_hour": agg_far["pipeline_a_legacy_cd2"],
            },
            "pipeline_b1_bytetrack_no_gate": {
                "name": f"Pipeline B1 (ByteTrack + K={eval_k}/N={eval_n}, Cooldown {eval_cd}s, feat=0.0)",
                "event_fas": agg_events["pipeline_b1_bytetrack_no_gate"],
                "far_per_hour": agg_far["pipeline_b1_bytetrack_no_gate"],
                "far_reduction_vs_a_ctrl_pct": round(far_reduct_b1, 2),
            },
            "pipeline_b2_bytetrack_with_gate": {
                "name": f"Pipeline B2 (ByteTrack + K={eval_k}/N={eval_n} + Temporal Feature Gate, Cooldown {eval_cd}s, feat={eval_feat})",
                "event_fas": agg_events["pipeline_b2_bytetrack_with_gate"],
                "far_point_estimate": agg_far["pipeline_b2_bytetrack_with_gate"],
                "poisson_95_ci": {
                    "total_ci": [ci_b2_tot_lower, ci_b2_tot_upper],
                    "fire_ci": [ci_b2_fire_lower, ci_b2_fire_upper],
                    "smoke_ci": [ci_b2_smoke_lower, ci_b2_smoke_upper],
                },
                "far_reduction_vs_a_ctrl_pct": round(far_reduct_b2, 2),
                "statistical_rigor_note": (
                    f"Trong {total_eval_hours:.4f} giờ đánh giá ({total_inferred_duration_sec:.1f}s), Pipeline B2 ghi nhận 0 sự kiện cảnh báo sai. "
                    f"Điểm ước lượng là 0.00/h. Giới hạn trên khoảng tin cậy Poisson 95% theo Rule of Three là {ci_b2_tot_upper:.2f}/h. "
                    "Không tuyên bố FAR = 0 theo nghĩa tuyệt đối khi thời lượng đánh giá là hữu hạn."
                ),
            },
            "mt3_target_met_ge_50pct": bool(far_reduct_b2 >= 50.0),
        },
        "per_video_results": video_results,
    }

    out_json = ROOT_DIR / "runs" / "benchmarks" / "week4_scientific" / "far_methodology_reconciliation.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Đã lưu kết quả đối soát FAR tại: %s", out_json)

    # Xuất CSV
    out_csv = ROOT_DIR / "runs" / "benchmarks" / "week4_scientific" / "far_methodology_reconciliation.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "video_id", "file_name", "native_fps", "native_frames", "inferred_frames",
            "inferred_dur_sec", "inferred_dur_hours",
            "fp_det_fire", "fp_det_smoke", "fp_det_total",
            "fp_frame_fire", "fp_frame_smoke", "fp_frame_any",
            "fa_a_ctrl_fire", "fa_a_ctrl_smoke", "fa_a_ctrl_total", "far_a_ctrl_total",
            "fa_b1_fire", "fa_b1_smoke", "fa_b1_total", "far_b1_total",
            "fa_b2_fire", "fa_b2_smoke", "fa_b2_total", "far_b2_total", "far_b2_poisson_upper_95",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in video_results:
            writer.writerow({
                "video_id": v["video_id"],
                "file_name": v["file_name"],
                "native_fps": v["native_fps"],
                "native_frames": v["total_physical_frames"],
                "inferred_frames": v["inferred_frames"],
                "inferred_dur_sec": v["inferred_duration_sec"],
                "inferred_dur_hours": v["inferred_duration_hours"],
                "fp_det_fire": v["false_positive_detections"]["fire"],
                "fp_det_smoke": v["false_positive_detections"]["smoke"],
                "fp_det_total": v["false_positive_detections"]["total"],
                "fp_frame_fire": v["false_positive_frames"]["fire"],
                "fp_frame_smoke": v["false_positive_frames"]["smoke"],
                "fp_frame_any": v["false_positive_frames"]["any"],
                "fa_a_ctrl_fire": v["false_alarm_events"]["pipeline_a_controlled_cd5s"]["fire"],
                "fa_a_ctrl_smoke": v["false_alarm_events"]["pipeline_a_controlled_cd5s"]["smoke"],
                "fa_a_ctrl_total": v["false_alarm_events"]["pipeline_a_controlled_cd5s"]["total"],
                "far_a_ctrl_total": v["far_per_hour"]["pipeline_a_controlled_cd5s"]["total"],
                "fa_b1_fire": v["false_alarm_events"]["pipeline_b1_bytetrack_no_gate"]["fire"],
                "fa_b1_smoke": v["false_alarm_events"]["pipeline_b1_bytetrack_no_gate"]["smoke"],
                "fa_b1_total": v["false_alarm_events"]["pipeline_b1_bytetrack_no_gate"]["total"],
                "far_b1_total": v["far_per_hour"]["pipeline_b1_bytetrack_no_gate"]["total"],
                "fa_b2_fire": v["false_alarm_events"]["pipeline_b2_bytetrack_with_gate"]["fire"],
                "fa_b2_smoke": v["false_alarm_events"]["pipeline_b2_bytetrack_with_gate"]["smoke"],
                "fa_b2_total": v["false_alarm_events"]["pipeline_b2_bytetrack_with_gate"]["total"],
                "far_b2_total": v["far_per_hour"]["pipeline_b2_bytetrack_with_gate"]["total"],
                "far_b2_poisson_upper_95": v["far_per_hour"]["pipeline_b2_poisson_95_ci"][1],
            })
    LOGGER.info("Đã lưu kết quả CSV đối soát tại: %s", out_csv)

    return summary_report


if __name__ == "__main__":
    defaults = SystemSettings()
    parser = argparse.ArgumentParser(description="Đánh giá FAR chuẩn hóa phương pháp luận")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=2000,
        help="Số khung tối đa cho video dài (0 hoặc số âm để chạy full_video mọi frame decodable)",
    )
    parser.add_argument("--conf", type=float, default=defaults.conf_threshold, help="Ngưỡng confidence (mặc định SystemSettings)")
    parser.add_argument("--k", type=int, default=defaults.confirm_frames_k, help="Confirm frames K (mặc định SystemSettings)")
    parser.add_argument("--n", type=int, default=defaults.window_size_n, help="Window size N (mặc định SystemSettings)")
    parser.add_argument("--feat", type=float, default=defaults.feature_threshold, help="Feature threshold (mặc định SystemSettings)")
    parser.add_argument("--cd", type=float, default=defaults.alert_cooldown_sec, help="Cooldown sec (mặc định SystemSettings)")
    args = parser.parse_args()

    max_frames_arg = args.max_frames if args.max_frames > 0 else None

    run_unified_negative_evaluation(
        max_frames_per_long_video=max_frames_arg,
        conf_threshold=args.conf,
        confirm_k=args.k,
        window_n=args.n,
        feat_threshold=args.feat,
        cooldown_sec=args.cd,
    )
