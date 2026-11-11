"""Ablation Grid Search trên tập Validation để cải thiện Event Instance Recall (Task 3).

Thực hiện:
1. Đánh giá Baseline A trên Validation Set (12 video: POS_01, POS_03, ..., POS_23).
2. Thử nghiệm các không gian siêu tham số:
   - K/N (cửa sổ trượt xác nhận thời gian)
   - Confidence threshold
   - Feature threshold (ngưỡng biến động động học)
   - Cooldown (giãn cách cảnh báo)
   - Cấu hình phân tách độc lập giữa FIRE và SMOKE (Per-Class Separation)
3. Tính toán: TP, FN, FP, Recall, Precision, F1, Mean Delay.
4. Đo lường mức suy giảm Recall so với Baseline A: Target delta <= 5.0 điểm %.
5. Tuyệt đối KHÔNG sử dụng Locked Test Set để chọn cấu hình.
Xuất tệp:
- runs/benchmarks/week4_scientific/validation_recall_ablation.json
- runs/benchmarks/week4_scientific/validation_recall_ablation.csv
"""
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.event_eval import (
    Event,
    FrameDetection,
    evaluate,
    extract_frame_level_alarms,
    load_frame_detections,
    load_ground_truth_events,
    merge_frame_alarms_to_events,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("ablation_recall")


def _merge_timestamps(
    video_id: str,
    camera_id: str,
    label: str,
    samples: List[Tuple[float, float]],
    max_gap: float,
) -> List[Event]:
    if not samples:
        return []
    sorted_samples = sorted(samples, key=lambda s: s[0])
    events: List[Event] = []
    c_start = sorted_samples[0][0]
    c_end = sorted_samples[0][0]
    c_peak = sorted_samples[0][1]

    for ts, conf in sorted_samples[1:]:
        if (ts - c_end) <= max_gap:
            c_end = ts
            c_peak = max(c_peak, conf)
        else:
            events.append(
                Event(
                    video_id=video_id,
                    camera_id=camera_id,
                    label=label,
                    start_sec=c_start,
                    end_sec=c_end,
                    confidence=c_peak,
                )
            )
            c_start = ts
            c_end = ts
            c_peak = conf

    events.append(
        Event(
            video_id=video_id,
            camera_id=camera_id,
            label=label,
            start_sec=c_start,
            end_sec=c_end,
            confidence=c_peak,
        )
    )
    return events


def custom_frame_detections_to_events(
    frames: List[FrameDetection],
    default_conf: float = 0.35,
    default_k: int = 3,
    default_n: int = 8,
    default_feat: float = 0.02,
    default_cooldown: float = 5.0,
    max_event_gap_sec: float = 2.0,
    per_class_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Event], List[Event]]:
    """Tạo sự kiện hỗ trợ cấu hình per-class riêng biệt giữa FIRE và SMOKE."""
    per_class_params = per_class_params or {}

    # 1. Pipeline A: Baseline theo frame
    raw_alarms = []
    for frame in frames:
        for det in frame.detections:
            lbl = str(det.get("label", "")).upper()
            conf = float(det.get("confidence", 0.0))
            cfg = per_class_params.get(lbl, {})
            req_conf = cfg.get("confidence_threshold", default_conf)
            if conf >= req_conf:
                raw_alarms.append({
                    "video_id": frame.video_id,
                    "camera_id": frame.camera_id,
                    "label": lbl,
                    "timestamp_sec": frame.timestamp_sec,
                    "confidence": conf,
                })
    baseline = merge_frame_alarms_to_events(raw_alarms, max_event_gap_sec=max_event_gap_sec)

    # 2. Pipeline B: Temporal verification
    stream_frames: Dict[Tuple[str, str], List[FrameDetection]] = {}
    for frame in sorted(frames, key=lambda f: (f.video_id, f.camera_id, f.timestamp_sec)):
        stream_frames.setdefault((frame.video_id, frame.camera_id), []).append(frame)

    temporal: List[Event] = []
    for (video_id, camera_id), cam_frames in stream_frames.items():
        track_keys = set()
        per_frame_hits = []

        for frame in cam_frames:
            present = {}
            for det in frame.detections:
                lbl = str(det.get("label", "")).upper()
                conf = float(det.get("confidence", 0.0))
                cfg = per_class_params.get(lbl, {})
                req_conf = cfg.get("confidence_threshold", default_conf)
                if conf < req_conf:
                    continue

                tid_raw = det.get("track_id", "untracked")
                if isinstance(tid_raw, int) and tid_raw <= 0:
                    continue
                track = str(tid_raw)
                feat_score = float(
                    det.get("temporal_score", det.get("feature_score", det.get("flicker_score", 0.0)))
                )
                key = (lbl, track)
                prev_conf, prev_feat = present.get(key, (0.0, 0.0))
                present[key] = (max(prev_conf, conf), max(prev_feat, feat_score))
                track_keys.add(key)
            per_frame_hits.append(present)

        label_confirmed_samples: Dict[str, List[Tuple[float, float]]] = {}
        for label, track in sorted(track_keys):
            cfg = per_class_params.get(label, {})
            k = cfg.get("confirm_k", default_k)
            n = cfg.get("window_n", default_n)
            feat_thr = cfg.get("feature_threshold", default_feat)
            cd = cfg.get("cooldown_sec", default_cooldown)

            rolling = []
            was_confirmed = False
            in_active_alert = False
            last_alert_ts: Optional[float] = None

            for frame, present in zip(cam_frames, per_frame_hits):
                hit = (label, track) in present
                conf, feat_val = present.get((label, track), (0.0, 0.0))
                rolling.append((frame.timestamp_sec, hit, conf, feat_val))
                rolling = rolling[-n:]

                hit_count = sum(1 for _, h, _, _ in rolling if h)
                mean_feat = (
                    sum(fv for _, h, _, fv in rolling if h) / float(hit_count)
                    if hit_count > 0 else 0.0
                )
                passes_feat = (feat_thr <= 0.0) or (mean_feat >= feat_thr)
                confirmed = (hit_count >= k) and passes_feat

                if confirmed:
                    if not was_confirmed:
                        if cd <= 0.0 or last_alert_ts is None or (frame.timestamp_sec - last_alert_ts) >= cd:
                            in_active_alert = True
                            last_alert_ts = frame.timestamp_sec
                        else:
                            in_active_alert = False
                    if in_active_alert:
                        peak_conf = max(c for _, h, c, _ in rolling if h)
                        label_confirmed_samples.setdefault(label, []).append((frame.timestamp_sec, peak_conf))
                else:
                    in_active_alert = False
                was_confirmed = confirmed

        for label, samples in label_confirmed_samples.items():
            temporal.extend(_merge_timestamps(video_id, camera_id, label, samples, max_event_gap_sec))

    temporal.sort(key=lambda e: (e.video_id, e.camera_id, e.start_sec, e.label))
    return baseline, temporal


def run_ablation() -> Dict[str, Any]:
    LOGGER.info("=== BẮT ĐẦU ABLATION GRID SEARCH TRÊN TẬP VALIDATION (TASK 3) ===")

    val_ids = {
        "POS_01", "POS_03", "POS_05", "POS_07", "POS_09", "POS_11",
        "POS_13", "POS_15", "POS_17", "POS_19", "POS_21", "POS_23"
    }

    frame_jsonl_path = ROOT_DIR / "runs" / "benchmarks" / "week4_final" / "tracked_frame_detections.jsonl"
    all_frames = load_frame_detections(frame_jsonl_path)
    val_frames = [f for f in all_frames if f.video_id in val_ids]
    LOGGER.info("Nạp %d khung hình kiểm định thuộc 12 video Validation", len(val_frames))

    gt_all = load_ground_truth_events(video_ids=val_ids, only_verified=False, strict=False)
    gt_ver = load_ground_truth_events(video_ids=val_ids, only_verified=True, strict=True)
    LOGGER.info("Validation Ground Truth: %d all events, %d cross_verified events", len(gt_all), len(gt_ver))

    # 1. Đánh giá Mốc Chuẩn Baseline A (Per-frame baseline)
    base_events, _ = custom_frame_detections_to_events(val_frames, default_conf=0.35, default_k=1, default_n=1, default_feat=0.0, default_cooldown=0.0)
    rep_base_all = evaluate(gt_all, base_events, val_ids, allow_missing_negative=True)
    rep_base_ver = evaluate(gt_ver, base_events, val_ids, allow_missing_negative=True)

    base_recall_all = rep_base_all["metrics"]["event_instance_recall"]
    base_recall_ver = rep_base_ver["metrics"]["event_instance_recall"]
    base_delay = rep_base_all["metrics"]["mean_detection_delay_sec"]

    LOGGER.info(
        "BASELINE A (Validation): All Recall=%.1f%% (TP=%d/%d), Verified Recall=%.1f%% (TP=%d/%d), Mean Delay=%.2fs",
        base_recall_all * 100, rep_base_all["counts"]["matched_events"], len(gt_all),
        base_recall_ver * 100, rep_base_ver["counts"]["matched_events"], len(gt_ver),
        base_delay
    )

    # 2. Định nghĩa không gian thử nghiệm Ablation
    # A. Cấu hình Toàn cục (Global Configurations)
    configs = [
        # Nhóm tham chiếu cũ
        {
            "id": "current_pipeline_b2",
            "name": "Current Pipeline B2 (conf=0.35, K=3, N=8, feat=0.02, cd=5.0s)",
            "conf": 0.35, "k": 3, "n": 8, "feat": 0.02, "cd": 5.0,
            "per_class": None,
        },
        {
            "id": "current_pipeline_b1",
            "name": "Current Pipeline B1 (conf=0.35, K=3, N=8, feat=0.0, cd=5.0s)",
            "conf": 0.35, "k": 3, "n": 8, "feat": 0.0, "cd": 5.0,
            "per_class": None,
        },
        # Nhóm quét K và N
        {"id": "kn_k2_n6", "name": "Global K=2, N=6 (conf=0.35, feat=0.02, cd=5.0s)", "conf": 0.35, "k": 2, "n": 6, "feat": 0.02, "cd": 5.0, "per_class": None},
        {"id": "kn_k2_n8", "name": "Global K=2, N=8 (conf=0.35, feat=0.02, cd=5.0s)", "conf": 0.35, "k": 2, "n": 8, "feat": 0.02, "cd": 5.0, "per_class": None},
        {"id": "kn_k3_n6", "name": "Global K=3, N=6 (conf=0.35, feat=0.02, cd=5.0s)", "conf": 0.35, "k": 3, "n": 6, "feat": 0.02, "cd": 5.0, "per_class": None},
        {"id": "kn_k4_n10", "name": "Global K=4, N=10 (conf=0.35, feat=0.02, cd=5.0s)", "conf": 0.35, "k": 4, "n": 10, "feat": 0.02, "cd": 5.0, "per_class": None},
        # Nhóm quét Feature Threshold
        {"id": "feat_001", "name": "Global feat=0.01 (conf=0.35, K=3, N=8, cd=5.0s)", "conf": 0.35, "k": 3, "n": 8, "feat": 0.01, "cd": 5.0, "per_class": None},
        {"id": "feat_0005", "name": "Global feat=0.005 (conf=0.35, K=3, N=8, cd=5.0s)", "conf": 0.35, "k": 3, "n": 8, "feat": 0.005, "cd": 5.0, "per_class": None},
        {"id": "feat_000", "name": "Global feat=0.00 (conf=0.35, K=2, N=6, cd=5.0s)", "conf": 0.35, "k": 2, "n": 6, "feat": 0.00, "cd": 5.0, "per_class": None},
        # Nhóm quét Cooldown
        {"id": "cd_3s", "name": "Global cooldown=3.0s (conf=0.35, K=3, N=8, feat=0.02)", "conf": 0.35, "k": 3, "n": 8, "feat": 0.02, "cd": 3.0, "per_class": None},
        {"id": "cd_2s", "name": "Global cooldown=2.0s (conf=0.35, K=2, N=6, feat=0.01)", "conf": 0.35, "k": 2, "n": 6, "feat": 0.01, "cd": 2.0, "per_class": None},
        # B. Nhóm Cấu hình Phân tách Lớp (Per-Class Separation)
        {
            "id": "per_class_opt_1",
            "name": "Per-Class Opt 1: FIRE(K=3,N=8,feat=0.02) + SMOKE(K=2,N=8,feat=0.005)",
            "conf": 0.35, "k": 3, "n": 8, "feat": 0.02, "cd": 5.0,
            "per_class": {
                "FIRE": {"confirm_k": 3, "window_n": 8, "feature_threshold": 0.02, "cooldown_sec": 5.0, "confidence_threshold": 0.35},
                "SMOKE": {"confirm_k": 2, "window_n": 8, "feature_threshold": 0.005, "cooldown_sec": 5.0, "confidence_threshold": 0.30},
            },
        },
        {
            "id": "per_class_opt_2",
            "name": "Per-Class Opt 2: FIRE(K=2,N=6,feat=0.02) + SMOKE(K=2,N=8,feat=0.005)",
            "conf": 0.35, "k": 2, "n": 6, "feat": 0.02, "cd": 5.0,
            "per_class": {
                "FIRE": {"confirm_k": 2, "window_n": 6, "feature_threshold": 0.02, "cooldown_sec": 4.0, "confidence_threshold": 0.35},
                "SMOKE": {"confirm_k": 2, "window_n": 8, "feature_threshold": 0.005, "cooldown_sec": 4.0, "confidence_threshold": 0.28},
            },
        },
        {
            "id": "per_class_opt_3",
            "name": "Per-Class Opt 3: FIRE(K=2,N=6,feat=0.015) + SMOKE(K=2,N=6,feat=0.00)",
            "conf": 0.35, "k": 2, "n": 6, "feat": 0.015, "cd": 3.0,
            "per_class": {
                "FIRE": {"confirm_k": 2, "window_n": 6, "feature_threshold": 0.015, "cooldown_sec": 3.0, "confidence_threshold": 0.32},
                "SMOKE": {"confirm_k": 2, "window_n": 6, "feature_threshold": 0.000, "cooldown_sec": 3.0, "confidence_threshold": 0.25},
            },
        },
        {
            "id": "per_class_opt_4",
            "name": "Per-Class Opt 4 (Balanced): FIRE(conf=0.35,K=2,N=6,feat=0.02,cd=4.0) + SMOKE(conf=0.28,K=2,N=8,feat=0.00,cd=4.0)",
            "conf": 0.35, "k": 2, "n": 6, "feat": 0.02, "cd": 4.0,
            "per_class": {
                "FIRE": {"confirm_k": 2, "window_n": 6, "feature_threshold": 0.02, "cooldown_sec": 4.0, "confidence_threshold": 0.35},
                "SMOKE": {"confirm_k": 2, "window_n": 8, "feature_threshold": 0.00, "cooldown_sec": 4.0, "confidence_threshold": 0.28},
            },
        },
    ]

    ablation_results = []

    for cfg in configs:
        cid = cfg["id"]
        cname = cfg["name"]
        per_class = cfg.get("per_class")

        _, temp_events = custom_frame_detections_to_events(
            val_frames,
            default_conf=cfg["conf"],
            default_k=cfg["k"],
            default_n=cfg["n"],
            default_feat=cfg["feat"],
            default_cooldown=cfg["cd"],
            per_class_params=per_class,
        )

        rep_all = evaluate(gt_all, temp_events, val_ids, allow_missing_negative=True)
        rep_ver = evaluate(gt_ver, temp_events, val_ids, allow_missing_negative=True)

        m_all = rep_all["metrics"]
        c_all = rep_all["counts"]
        m_ver = rep_ver["metrics"]

        rec_all = m_all["event_instance_recall"]
        prec_all = m_all["event_precision"]
        f1_all = m_all["event_f1"]
        delay_all = m_all["mean_detection_delay_sec"]

        rec_ver = m_ver["event_instance_recall"]
        delay_ver = m_ver["mean_detection_delay_sec"]

        # Độ sụt giảm Recall so với Baseline A
        delta_rec_all_pts = (base_recall_all - rec_all) * 100.0
        delta_rec_ver_pts = (base_recall_ver - rec_ver) * 100.0

        meets_mt3_all = bool(delta_rec_all_pts <= 5.0 and delay_all <= 10.0)
        meets_mt3_ver = bool(delta_rec_ver_pts <= 5.0 and delay_ver <= 10.0)

        entry = {
            "id": cid,
            "name": cname,
            "has_per_class_params": bool(per_class is not None),
            "parameters": {
                "default": {
                    "confidence_threshold": cfg["conf"],
                    "k": cfg["k"],
                    "n": cfg["n"],
                    "feature_threshold": cfg["feat"],
                    "cooldown_sec": cfg["cd"],
                },
                "per_class": per_class,
            },
            "predicted_events_count": len(temp_events),
            "all_gt_evaluation": {
                "tp_matched": c_all["matched_events"],
                "fn_missed": c_all["unmatched_gt_events"],
                "fp_unmatched": c_all["unmatched_predicted_events"],
                "recall": round(rec_all, 4),
                "precision": round(prec_all, 4),
                "f1": round(f1_all, 4),
                "mean_delay_sec": round(delay_all, 3),
                "recall_drop_points": round(delta_rec_all_pts, 2),
                "meets_recall_target": bool(delta_rec_all_pts <= 5.0),
            },
            "verified_gt_evaluation": {
                "tp_matched": rep_ver["counts"]["matched_events"],
                "fn_missed": rep_ver["counts"]["unmatched_gt_events"],
                "recall": round(rec_ver, 4),
                "precision": round(rep_ver["metrics"]["event_precision"], 4),
                "f1": round(rep_ver["metrics"]["event_f1"], 4),
                "mean_delay_sec": round(delay_ver, 3),
                "recall_drop_points": round(delta_rec_ver_pts, 2),
                "meets_recall_target": bool(delta_rec_ver_pts <= 5.0),
            },
            "meets_mt3_overall": meets_mt3_all,
        }
        ablation_results.append(entry)

        LOGGER.info(
            "[%s] Rec All=%.1f%% (drop=%.1f pts | Target %s) | F1=%.1f%% | Delay=%.2fs | Preds=%d",
            cid, rec_all * 100, delta_rec_all_pts, "PASS" if delta_rec_all_pts <= 5.0 else "FAIL",
            f1_all * 100, delay_all, len(temp_events)
        )

    # Lựa chọn cấu hình tốt nhất trên Validation Set
    # Ưu tiên: meets_recall_target == True trên all GT, sau đó sắp xếp theo F1 giảm dần
    valid_candidates = [r for r in ablation_results if r["all_gt_evaluation"]["meets_recall_target"]]
    if not valid_candidates:
        # Nếu không có ứng viên nào đạt <= 5 điểm %, chọn ứng viên có độ sụt giảm nhỏ nhất
        best_candidate = min(ablation_results, key=lambda r: r["all_gt_evaluation"]["recall_drop_points"])
    else:
        best_candidate = max(valid_candidates, key=lambda r: (r["all_gt_evaluation"]["f1"], -r["all_gt_evaluation"]["mean_delay_sec"]))

    LOGGER.info("=== CẤU HÌNH TỐI ƯU ĐƯỢC CHỌN TỪ VALIDATION SET: %s ===", best_candidate["id"])

    payload = {
        "metadata": {
            "title": "Validation Set Event Instance Recall Ablation Study",
            "git_branch": "fix/week4-scientific-validation",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "validation_videos": sorted(list(val_ids)),
            "validation_frames_count": len(val_frames),
            "baseline_metrics": {
                "all_gt_recall": round(base_recall_all, 4),
                "verified_gt_recall": round(base_recall_ver, 4),
                "mean_delay_sec": round(base_delay, 3),
            },
        },
        "best_candidate_selected_on_validation": best_candidate,
        "ablation_results": ablation_results,
    }

    out_json = ROOT_DIR / "runs" / "benchmarks" / "week4_scientific" / "validation_recall_ablation.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Đã lưu kết quả Ablation JSON tại: %s", out_json)

    # Xuất CSV
    out_csv = ROOT_DIR / "runs" / "benchmarks" / "week4_scientific" / "validation_recall_ablation.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id", "name", "has_per_class_params",
            "tp_matched_all", "fn_missed_all", "fp_unmatched_all",
            "recall_all", "precision_all", "f1_all", "mean_delay_all_sec", "recall_drop_all_pts",
            "tp_matched_ver", "fn_missed_ver",
            "recall_ver", "precision_ver", "f1_ver", "mean_delay_ver_sec", "recall_drop_ver_pts",
            "meets_recall_target_all", "meets_mt3_overall"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in ablation_results:
            writer.writerow({
                "id": r["id"],
                "name": r["name"],
                "has_per_class_params": r["has_per_class_params"],
                "tp_matched_all": r["all_gt_evaluation"]["tp_matched"],
                "fn_missed_all": r["all_gt_evaluation"]["fn_missed"],
                "fp_unmatched_all": r["all_gt_evaluation"]["fp_unmatched"],
                "recall_all": r["all_gt_evaluation"]["recall"],
                "precision_all": r["all_gt_evaluation"]["precision"],
                "f1_all": r["all_gt_evaluation"]["f1"],
                "mean_delay_all_sec": r["all_gt_evaluation"]["mean_delay_sec"],
                "recall_drop_all_pts": r["all_gt_evaluation"]["recall_drop_points"],
                "tp_matched_ver": r["verified_gt_evaluation"]["tp_matched"],
                "fn_missed_ver": r["verified_gt_evaluation"]["fn_missed"],
                "recall_ver": r["verified_gt_evaluation"]["recall"],
                "precision_ver": r["verified_gt_evaluation"]["precision"],
                "f1_ver": r["verified_gt_evaluation"]["f1"],
                "mean_delay_ver_sec": r["verified_gt_evaluation"]["mean_delay_sec"],
                "recall_drop_ver_pts": r["verified_gt_evaluation"]["recall_drop_points"],
                "meets_recall_target_all": r["all_gt_evaluation"]["meets_recall_target"],
                "meets_mt3_overall": r["meets_mt3_overall"],
            })
    LOGGER.info("Đã lưu kết quả Ablation CSV tại: %s", out_csv)

    return payload


if __name__ == "__main__":
    run_ablation()

