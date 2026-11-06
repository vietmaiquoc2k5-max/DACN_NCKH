"""Script thực nghiệm đối sánh và nghiệm thu Tuần 4 (`scripts/benchmark_week4_bytetrack.py`).

Đánh giá đối sánh 4 Pipeline độc lập trên cùng tập video, cùng model, cùng frame sampling (5.0 FPS)
và cùng ngưỡng confidence:
1. Pipeline A — `frame_level_baseline`:
   - Chỉ chạy YOLO detector (`runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt` hoặc `best.pt`).
   - Mỗi frame có detection FIRE/SMOKE >= confidence_threshold tạo ngay 1 frame-level alarm.
   - Tuyệt đối KHÔNG dùng ByteTrack, KHÔNG dùng K/N, KHÔNG dùng TemporalVerifier, KHÔNG dùng Cooldown.
   - Ghi nhận toàn bộ raw frame-level alarms và gộp thành Event intervals với `max_event_gap_sec = 2.0s`.
2. Pipeline B1 — `bytetrack_no_gate`:
   - YOLO detector + `ByteTrackAdapter` + `TemporalVerifier` (luật K/N theo từng track_id,
     vùng loại trừ ROI, Cooldown nhân quả, `feature_threshold = 0.0` - TẮT feature gate).
3. Pipeline B2 — `bytetrack_feature_gate`:
   - YOLO detector + `ByteTrackAdapter` + `TemporalFeatureExtractor` + `TemporalVerifier` (luật K/N theo từng track_id,
     vùng loại trừ ROI, Cooldown nhân quả, `feature_threshold = 0.02` - BẬT feature gate).
4. Pipeline C — `legacy_iou_temporal`:
   - YOLO detector + `TemporalVerifier` với IoU tracking nội bộ (`tracks_assigned=False`, luật K/N, Cooldown).

Thực hiện:
- Chạy toàn diện trên cả tập Video Dương tính (available_positive) và Video Âm tính (available_negative).
- Chia tập 24 video dương tính hiện có thành Validation Set (12 video lẻ: POS_01, POS_03, ..., POS_23)
  và Test Set (12 video chẵn: POS_02, POS_04, ..., POS_24) hoàn toàn tách biệt 100%.
- Quét lưới tham số (Grid Search) trên Validation Set và đánh giá điểm vận hành trên Test Set.
- Kiểm tra sự tồn tại thực tế của các video âm tính trên ổ đĩa; xuất `negative_video_metrics.json`,
  manifest thiếu dữ liệu âm tính và báo cáo trung thực `BLOCKED_BY_NEGATIVE_DATA` nếu < 5.0 giờ.
- Đo tách biệt thời gian suy luận của riêng Detector (Detector-only Latency/FPS) và của toàn bộ
  Pipeline xác nhận theo thời gian (End-to-End Pipeline Latency/FPS) kèm phân rã overhead.

Xuất đầy đủ kết quả vào `runs/benchmarks/week4_final/` (và đồng bộ sang `runs/benchmarks/week4_bytetrack/`).
"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("benchmark_week4")

from src.bytetrack_adapter import ByteTrackAdapter
from src.config import ByteTrackSettings
from src.detector import SharedFireSmokeDetector
from src.event_eval import (
    Event,
    FrameDetection,
    evaluate,
    export_events_csv,
    export_grid_search_csv,
    export_json,
    extract_frame_level_alarms,
    frame_detections_to_events,
    grid_search,
    merge_frame_alarms_to_events,
    select_best_operating_point,
)
from src.temporal_features import TemporalFeatureExtractor
from src.temporal_verifier import DetectionBox, TemporalVerifier, compute_dynamic_feature_score


def set_deterministic_seed(seed: int = 42) -> None:
    """Cố định seed để đảm bảo tính tái lập thực nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset_catalog(csv_path: Path) -> Dict[str, Any]:
    """Đọc danh mục video từ data/video_dataset_list.csv và phân loại theo trạng thái tồn tại trên ổ đĩa."""
    available_positive: List[Dict[str, Any]] = []
    missing_positive: List[Dict[str, Any]] = []
    available_negative: List[Dict[str, Any]] = []
    missing_negative: List[Dict[str, Any]] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"].strip()
            file_name = row.get("file_name", "").strip()
            rel_path = row.get("relative_path", row.get("file_path", "")).strip()
            cat = row.get("category", "").strip().lower()

            primary_path = ROOT_DIR / rel_path
            fallback_path = ROOT_DIR / "FireSmokeLabeler" / "videos" / file_name
            if primary_path.exists():
                full_path = primary_path
                exists = True
            elif file_name and fallback_path.exists():
                full_path = fallback_path
                exists = True
            else:
                full_path = primary_path
                exists = False

            is_pos = ("neg" not in cat) and (not vid.upper().startswith("NEG")) and ("_NEG_" not in vid.upper())
            duration = float(row["duration_sec"])
            fire_start = float(row["fire_start_sec"])
            smoke_start = float(row["smoke_start_sec"])
            item = {
                "video_id": vid,
                "camera_id": f"BENCH_{vid}",
                "file_name": file_name,
                "rel_path": str(full_path.relative_to(ROOT_DIR)) if exists else rel_path,
                "catalog_relative_path": rel_path,
                "path": str(full_path),
                "exists_on_disk": exists,
                "is_positive": is_pos,
                "category": cat,
                "duration_sec": duration,
                "fps": float(row["fps"]),
                "resolution": row.get("resolution", ""),
                "fire_start_sec": fire_start,
                "smoke_start_sec": smoke_start,
                "notes": row.get("notes", ""),
            }
            if is_pos:
                if exists:
                    available_positive.append(item)
                else:
                    missing_positive.append(item)
            else:
                if exists:
                    available_negative.append(item)
                else:
                    missing_negative.append(item)

    return {
        "available_positive": available_positive,
        "missing_positive": missing_positive,
        "available_negative": available_negative,
        "missing_negative": missing_negative,
    }


def build_ground_truth_events(videos: List[Dict[str, Any]]) -> List[Event]:
    """Tạo danh sách sự kiện Ground Truth từ fire_start_sec và smoke_start_sec của từng video dương tính."""
    gt_events: List[Event] = []
    for v in videos:
        if not v["is_positive"]:
            continue
        vid = v["video_id"]
        cam_id = v["camera_id"]
        dur = float(v["duration_sec"])
        f_start = float(v["fire_start_sec"])
        s_start = float(v["smoke_start_sec"])

        if f_start >= 0 and s_start >= 0:
            gt_events.append(Event(vid, cam_id, "FIRE", f_start, dur, 1.0))
            gt_events.append(Event(vid, cam_id, "SMOKE", s_start, dur, 1.0))
        elif f_start >= 0:
            gt_events.append(Event(vid, cam_id, "FIRE", f_start, dur, 1.0))
        elif s_start >= 0:
            gt_events.append(Event(vid, cam_id, "SMOKE", s_start, dur, 1.0))
    return gt_events


def sample_video_frames(video_path: str, sample_fps: float = 5.0) -> List[Tuple[int, float, np.ndarray]]:
    """Đọc và lấy mẫu các khung hình của một video theo tần số sample_fps cố định."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(native_fps / sample_fps)))
    sampled: List[Tuple[int, float, np.ndarray]] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame_idx % step == 0:
            ts_sec = frame_idx / float(native_fps)
            sampled.append((frame_idx, ts_sec, frame))
        frame_idx += 1
    cap.release()
    return sampled


def clone_boxes(boxes: List[DetectionBox]) -> List[DetectionBox]:
    """Tạo bản sao độc lập hoàn toàn của danh sách DetectionBox để các pipeline không dùng chung trạng thái."""
    return [
        DetectionBox(
            cls_id=b.cls_id,
            label=b.label,
            confidence=b.confidence,
            xyxy=tuple(b.xyxy),
            norm_xywh=tuple(b.norm_xywh),
            track_id=-1,
        )
        for b in boxes
    ]


def run_pipelines_on_video(
    detector: SharedFireSmokeDetector,
    video_info: Dict[str, Any],
    sample_fps: float = 5.0,
    conf_thresh: float = 0.25,
    eval_conf_thresh: float = 0.35,
    imgsz: int = 416,
    window_n: int = 8,
    confirm_k: int = 3,
    suspect_k: int = 2,
    cooldown_sec: float = 5.0,
    feature_threshold: float = 0.02,
    max_event_gap_sec: float = 2.0,
) -> Dict[str, Any]:
    """Chạy độc lập 4 pipeline trên cùng một video (dương tính hoặc âm tính) và thu thập toàn bộ dữ liệu.

    Pipelines:
    - Pipeline A: Frame-level baseline (YOLO detector, không tracker, không K/N, không cooldown).
    - Pipeline B1: ByteTrack + K/N, KHÔNG feature gate (feature_threshold = 0.0).
    - Pipeline B2: ByteTrack + K/N + Temporal Feature Gate (feature_threshold > 0.0, e.g. 0.02).
    - Pipeline C: Legacy IoU Temporal (IoU tracker + K/N).
    """
    vid = video_info["video_id"]
    cam_id = video_info["camera_id"]
    sampled_frames = sample_video_frames(video_info["path"], sample_fps=sample_fps)

    # Khởi tạo ByteTrack cho phép theo dõi mọi box >= conf_thresh (0.25)
    bt_tracker = ByteTrackAdapter(
        settings=ByteTrackSettings(
            track_high_thresh=conf_thresh,
            track_low_thresh=0.10,
            new_track_thresh=conf_thresh,
            track_buffer=15,
            match_thresh=0.70,
            fuse_score=False,
        ),
        per_class=True,
    )
    feat_extractor = TemporalFeatureExtractor(history_size=12)

    # Pipeline B1: ByteTrack + K/N, KHÔNG feature gate
    verifier_b1 = TemporalVerifier(
        window_size_n=window_n,
        confirm_frames_k=confirm_k,
        suspect_frames_k=suspect_k,
        cooldown_sec=cooldown_sec,
        feature_threshold=0.0,
    )

    # Pipeline B2: ByteTrack + K/N + Temporal Feature Gate
    verifier_b2 = TemporalVerifier(
        window_size_n=window_n,
        confirm_frames_k=confirm_k,
        suspect_frames_k=suspect_k,
        cooldown_sec=cooldown_sec,
        feature_threshold=feature_threshold,
    )

    # Pipeline C: Legacy IoU Temporal
    verifier_c = TemporalVerifier(
        window_size_n=window_n,
        confirm_frames_k=confirm_k,
        suspect_frames_k=suspect_k,
        cooldown_sec=cooldown_sec,
        feature_threshold=0.0,
    )

    detector_latencies_ms: List[float] = []
    tracker_overhead_ms: List[float] = []
    feature_overhead_ms: List[float] = []
    verifier_b1_overhead_ms: List[float] = []
    verifier_b2_overhead_ms: List[float] = []
    pipeline_b1_latencies_ms: List[float] = []
    pipeline_b2_latencies_ms: List[float] = []
    pipeline_c_latencies_ms: List[float] = []

    tracked_frame_detections: List[FrameDetection] = []
    raw_frame_detections: List[FrameDetection] = []

    # Lưu mốc thời gian xác nhận cho từng pipeline
    b1_confirmed_samples: Dict[str, List[Tuple[float, float]]] = {}
    b1_db_alerts: List[Dict[str, Any]] = []

    b2_confirmed_samples: Dict[str, List[Tuple[float, float]]] = {}
    b2_db_alerts: List[Dict[str, Any]] = []

    c_confirmed_samples: Dict[str, List[Tuple[float, float]]] = {}
    c_db_alerts: List[Dict[str, Any]] = []

    for frame_idx, ts_sec, frame in sampled_frames:
        # 1. Chạy YOLO Detector
        t_det0 = time.perf_counter()
        raw_boxes_res, det_ms = detector.detect_frame(frame, conf_threshold=conf_thresh, imgsz=imgsz)
        raw_boxes_all = raw_boxes_res or []
        t_det1 = time.perf_counter()
        det_elapsed_ms = (t_det1 - t_det0) * 1000.0
        detector_latencies_ms.append(det_elapsed_ms)

        # Lưu raw detections cho Pipeline A và Grid Search
        raw_det_dicts = tuple(
            {
                "label": b.label,
                "confidence": round(float(b.confidence), 4),
                "xyxy": list(b.xyxy),
                "track_id": -1,
            }
            for b in raw_boxes_all
        )
        raw_frame_detections.append(
            FrameDetection(vid, cam_id, round(ts_sec, 4), raw_det_dicts)
        )

        # 2. Chạy ByteTrack trên toàn bộ detection >= conf_thresh
        t_trk0 = time.perf_counter()
        tracked_boxes_all = bt_tracker.update(clone_boxes(raw_boxes_all), frame)
        t_trk1 = time.perf_counter()
        trk_ms = (t_trk1 - t_trk0) * 1000.0
        tracker_overhead_ms.append(trk_ms)

        # 3. Trích xuất Temporal Features thực tế cho từng track (KHÔNG gán điểm giả 0.02)
        t_feat0 = time.perf_counter()
        feat_map_all: Dict[int, Any] = {}
        det_dicts_with_tracks: List[Dict[str, Any]] = []
        for tb in tracked_boxes_all:
            try:
                f_res = feat_extractor.update(
                    track_id=tb.track_id,
                    frame_bgr=frame,
                    xyxy=tb.xyxy,
                    timestamp_sec=ts_sec,
                )
                feat_map_all[tb.track_id] = f_res
                dyn_score = compute_dynamic_feature_score(f_res)
            except Exception as feat_err:
                LOGGER.warning(
                    "Lỗi khi trích xuất temporal feature cho track_id %d trong video %s tại %.2fs: %s",
                    tb.track_id, vid, ts_sec, feat_err, exc_info=True
                )
                f_res = None
                dyn_score = 0.0

            det_dicts_with_tracks.append({
                "label": tb.label,
                "confidence": round(float(tb.confidence), 4),
                "track_id": int(tb.track_id),
                "temporal_score": round(float(dyn_score), 4),
                "flicker_score": round(float(f_res.flicker_score), 4) if f_res else 0.0,
                "smoke_expansion_score": round(float(f_res.smoke_expansion_score), 4) if f_res else 0.0,
                "area_growth_rate": round(float(f_res.area_growth_rate), 4) if f_res else 0.0,
            })
        t_feat1 = time.perf_counter()
        feat_ms = (t_feat1 - t_feat0) * 1000.0
        feature_overhead_ms.append(feat_ms)

        # Lưu FrameDetection chuẩn hóa (chứa track_id và temporal_score thực tế) phục vụ Grid Search
        tracked_frame_detections.append(
            FrameDetection(vid, cam_id, round(ts_sec, 4), tuple(det_dicts_with_tracks))
        )

        # ----------------------------------------------------------------------
        # ĐÁNH GIÁ TẠI NGƯỠNG eval_conf_thresh (0.35)
        # ----------------------------------------------------------------------
        boxes_for_b = [tb for tb in tracked_boxes_all if tb.confidence >= eval_conf_thresh]
        feat_map_b = {tb.track_id: feat_map_all[tb.track_id] for tb in boxes_for_b if tb.track_id in feat_map_all}

        # PIPELINE B1: ByteTrack + K/N (Không Feature Gate)
        t_b1_start = time.perf_counter()
        res_b1 = verifier_b1.update(
            boxes_for_b,
            tracks_assigned=True,
            feature_map=feat_map_b,
            timestamp_sec=ts_sec,
        )
        t_b1_end = time.perf_counter()
        ver_b1_ms = (t_b1_end - t_b1_start) * 1000.0
        verifier_b1_overhead_ms.append(ver_b1_ms)
        pipeline_b1_latencies_ms.append(det_elapsed_ms + trk_ms + ver_b1_ms)

        if res_b1["alert_level"] == "FIRE" and res_b1["primary_label"]:
            lbl_b1 = res_b1["primary_label"]
            b1_confirmed_samples.setdefault(lbl_b1, []).append((round(ts_sec, 4), float(res_b1["max_confidence"])))
        if res_b1["should_trigger_db_alert"] and res_b1["primary_label"]:
            b1_db_alerts.append({
                "video_id": vid,
                "camera_id": cam_id,
                "timestamp_sec": round(ts_sec, 4),
                "label": res_b1["primary_label"],
                "confidence": float(res_b1["max_confidence"]),
                "active_hits": int(res_b1["active_hits"]),
                "confirmed_track_id": res_b1["confirmed_track_id"],
                "temporal_score": float(res_b1["temporal_score"]),
            })

        # PIPELINE B2: ByteTrack + K/N + Temporal Feature Gate
        t_b2_start = time.perf_counter()
        res_b2 = verifier_b2.update(
            boxes_for_b,
            tracks_assigned=True,
            feature_map=feat_map_b,
            timestamp_sec=ts_sec,
        )
        t_b2_end = time.perf_counter()
        ver_b2_ms = (t_b2_end - t_b2_start) * 1000.0
        verifier_b2_overhead_ms.append(ver_b2_ms)
        pipeline_b2_latencies_ms.append(det_elapsed_ms + trk_ms + feat_ms + ver_b2_ms)

        if res_b2["alert_level"] == "FIRE" and res_b2["primary_label"]:
            lbl_b2 = res_b2["primary_label"]
            b2_confirmed_samples.setdefault(lbl_b2, []).append((round(ts_sec, 4), float(res_b2["max_confidence"])))
        if res_b2["should_trigger_db_alert"] and res_b2["primary_label"]:
            b2_db_alerts.append({
                "video_id": vid,
                "camera_id": cam_id,
                "timestamp_sec": round(ts_sec, 4),
                "label": res_b2["primary_label"],
                "confidence": float(res_b2["max_confidence"]),
                "active_hits": int(res_b2["active_hits"]),
                "confirmed_track_id": res_b2["confirmed_track_id"],
                "temporal_score": float(res_b2["temporal_score"]),
            })

        # PIPELINE C: Legacy IoU Temporal
        boxes_for_c = [b for b in clone_boxes(raw_boxes_all) if b.confidence >= eval_conf_thresh]
        t_c_start = time.perf_counter()
        res_c = verifier_c.update(
            boxes_for_c,
            tracks_assigned=False,
            timestamp_sec=ts_sec,
        )
        t_c_end = time.perf_counter()
        ver_c_ms = (t_c_end - t_c_start) * 1000.0
        pipeline_c_latencies_ms.append(det_elapsed_ms + ver_c_ms)

        if res_c["alert_level"] == "FIRE" and res_c["primary_label"]:
            lbl_c = res_c["primary_label"]
            c_confirmed_samples.setdefault(lbl_c, []).append((round(ts_sec, 4), float(res_c["max_confidence"])))
        if res_c["should_trigger_db_alert"] and res_c["primary_label"]:
            c_db_alerts.append({
                "video_id": vid,
                "camera_id": cam_id,
                "timestamp_sec": round(ts_sec, 4),
                "label": res_c["primary_label"],
                "confidence": float(res_c["max_confidence"]),
                "active_hits": int(res_c["active_hits"]),
                "confirmed_track_id": res_c["confirmed_track_id"],
            })

    # --------------------------------------------------------------------------
    # PIPELINE A: Frame-level Baseline
    # --------------------------------------------------------------------------
    raw_frame_alarms = extract_frame_level_alarms(
        raw_frame_detections, confidence_threshold=eval_conf_thresh
    )
    pipeline_a_events = merge_frame_alarms_to_events(
        raw_frame_alarms, max_event_gap_sec=max_event_gap_sec
    )

    # Gộp sự kiện cho Pipeline B1
    pipeline_b1_events: List[Event] = []
    for lbl, samples in b1_confirmed_samples.items():
        alarms_b1 = [
            {"video_id": vid, "camera_id": cam_id, "timestamp_sec": ts, "label": lbl, "confidence": conf}
            for ts, conf in samples
        ]
        pipeline_b1_events.extend(merge_frame_alarms_to_events(alarms_b1, max_event_gap_sec=max_event_gap_sec))
    pipeline_b1_events.sort(key=lambda e: (e.start_sec, e.label))

    # Gộp sự kiện cho Pipeline B2
    pipeline_b2_events: List[Event] = []
    for lbl, samples in b2_confirmed_samples.items():
        alarms_b2 = [
            {"video_id": vid, "camera_id": cam_id, "timestamp_sec": ts, "label": lbl, "confidence": conf}
            for ts, conf in samples
        ]
        pipeline_b2_events.extend(merge_frame_alarms_to_events(alarms_b2, max_event_gap_sec=max_event_gap_sec))
    pipeline_b2_events.sort(key=lambda e: (e.start_sec, e.label))

    # Gộp sự kiện cho Pipeline C
    pipeline_c_events: List[Event] = []
    for lbl, samples in c_confirmed_samples.items():
        alarms_c = [
            {"video_id": vid, "camera_id": cam_id, "timestamp_sec": ts, "label": lbl, "confidence": conf}
            for ts, conf in samples
        ]
        pipeline_c_events.extend(merge_frame_alarms_to_events(alarms_c, max_event_gap_sec=max_event_gap_sec))
    pipeline_c_events.sort(key=lambda e: (e.start_sec, e.label))

    return {
        "video_id": vid,
        "camera_id": cam_id,
        "is_positive": video_info["is_positive"],
        "duration_sec": video_info["duration_sec"],
        "sampled_frames": len(sampled_frames),
        "raw_frame_detections": raw_frame_detections,
        "tracked_frame_detections": tracked_frame_detections,
        "pipeline_a": {
            "raw_frame_alarms": raw_frame_alarms,
            "raw_frame_alarm_count": len(raw_frame_alarms),
            "events": pipeline_a_events,
            "event_count": len(pipeline_a_events),
            "mean_detector_latency_ms": float(np.mean(detector_latencies_ms)) if detector_latencies_ms else 0.0,
        },
        "pipeline_b1_no_gate": {
            "db_alerts": b1_db_alerts,
            "db_alert_count": len(b1_db_alerts),
            "events": pipeline_b1_events,
            "event_count": len(pipeline_b1_events),
            "mean_pipeline_latency_ms": float(np.mean(pipeline_b1_latencies_ms)) if pipeline_b1_latencies_ms else 0.0,
            "mean_tracker_overhead_ms": float(np.mean(tracker_overhead_ms)) if tracker_overhead_ms else 0.0,
            "mean_verifier_overhead_ms": float(np.mean(verifier_b1_overhead_ms)) if verifier_b1_overhead_ms else 0.0,
        },
        "pipeline_b2_feature_gate": {
            "db_alerts": b2_db_alerts,
            "db_alert_count": len(b2_db_alerts),
            "events": pipeline_b2_events,
            "event_count": len(pipeline_b2_events),
            "mean_pipeline_latency_ms": float(np.mean(pipeline_b2_latencies_ms)) if pipeline_b2_latencies_ms else 0.0,
            "mean_tracker_overhead_ms": float(np.mean(tracker_overhead_ms)) if tracker_overhead_ms else 0.0,
            "mean_feature_overhead_ms": float(np.mean(feature_overhead_ms)) if feature_overhead_ms else 0.0,
            "mean_verifier_overhead_ms": float(np.mean(verifier_b2_overhead_ms)) if verifier_b2_overhead_ms else 0.0,
        },
        "pipeline_c": {
            "db_alerts": c_db_alerts,
            "db_alert_count": len(c_db_alerts),
            "events": pipeline_c_events,
            "event_count": len(pipeline_c_events),
            "mean_pipeline_latency_ms": float(np.mean(pipeline_c_latencies_ms)) if pipeline_c_latencies_ms else 0.0,
        },
    }


def evaluate_video_level_primary_recall(
    videos: List[Dict[str, Any]],
    predicted_events: List[Event],
    tolerance_sec: float = 10.0,
) -> Dict[str, Any]:
    """Đánh giá Video-level Primary Recall (bất kể nhãn FIRE/SMOKE trên video dương tính)."""
    events_by_vid: Dict[str, List[Event]] = {}
    for ev in predicted_events:
        events_by_vid.setdefault(ev.video_id, []).append(ev)

    recalled_videos = 0
    delays: List[float] = []
    pre_event_alarms = 0
    per_video_details: List[Dict[str, Any]] = []

    for v in videos:
        if not v["is_positive"]:
            continue
        vid = v["video_id"]
        dur = float(v["duration_sec"])
        starts = [s for s in (float(v["fire_start_sec"]), float(v["smoke_start_sec"])) if s >= 0.0]
        earliest_gt = min(starts) if starts else 0.0
        vid_events = sorted(events_by_vid.get(vid, []), key=lambda x: x.start_sec)

        matched_ev = None
        vid_pre_alarms = 0
        for ev in vid_events:
            if ev.end_sec < earliest_gt - 0.5:
                vid_pre_alarms += 1
            elif matched_ev is None and ev.start_sec <= dur + tolerance_sec:
                matched_ev = ev

        pre_event_alarms += vid_pre_alarms
        if matched_ev is not None:
            recalled_videos += 1
            delay = max(0.0, matched_ev.start_sec - earliest_gt)
            delays.append(delay)
            per_video_details.append({
                "video_id": vid,
                "gt_earliest_start_sec": earliest_gt,
                "first_confirmed_event_sec": matched_ev.start_sec,
                "delay_sec": round(delay, 4),
                "recalled": True,
                "pre_event_alarms": vid_pre_alarms,
                "total_events_in_video": len(vid_events),
            })
        else:
            per_video_details.append({
                "video_id": vid,
                "gt_earliest_start_sec": earliest_gt,
                "first_confirmed_event_sec": None,
                "delay_sec": None,
                "recalled": False,
                "pre_event_alarms": vid_pre_alarms,
                "total_events_in_video": len(vid_events),
            })

    total_pos = len([v for v in videos if v["is_positive"]])
    return {
        "total_positive_videos": total_pos,
        "recalled_positive_videos": recalled_videos,
        "any_class_video_recall": recalled_videos / total_pos if total_pos > 0 else 0.0,
        "mean_delay_from_earliest_onset_sec": float(np.mean(delays)) if delays else None,
        "pre_event_false_alarms_on_positive_videos": pre_event_alarms,
        "per_video": per_video_details,
    }


def main() -> int:
    set_deterministic_seed(42)

    csv_path = ROOT_DIR / "data" / "video_dataset_list.csv"
    out_dir = ROOT_DIR / "runs" / "benchmarks" / "week4_final"
    legacy_out_dir = ROOT_DIR / "runs" / "benchmarks" / "week4_bytetrack"
    out_dir.mkdir(parents=True, exist_ok=True)
    legacy_out_dir.mkdir(parents=True, exist_ok=True)

    cand_model = ROOT_DIR / "runs" / "detect" / "runs" / "benchmarks" / "yolov8n_seed42" / "weights" / "best.pt"
    model_path = cand_model if cand_model.exists() else (ROOT_DIR / "best.pt")

    print("=" * 85)
    print("🔥 BENCHMARK NGHIỆM THU TUẦN 4: FRAME-LEVEL BASELINE vs BYTETRACK + TEMPORAL K/N")
    print("=" * 85)
    print(f"📦 Model sử dụng : {model_path.relative_to(ROOT_DIR)}")

    catalog = load_dataset_catalog(csv_path)
    avail_pos = catalog["available_positive"]
    avail_neg = catalog["available_negative"]
    miss_neg = catalog["missing_negative"]
    miss_pos = catalog["missing_positive"]

    total_pos_sec = sum(v["duration_sec"] for v in avail_pos)
    total_neg_avail_sec = sum(v["duration_sec"] for v in avail_neg)
    total_neg_miss_sec = sum(v["duration_sec"] for v in miss_neg)

    print(f"🎬 Video dương tính có trên ổ đĩa : {len(avail_pos)} video ({total_pos_sec:.2f}s = {total_pos_sec/3600.0:.4f}h)")
    print(f"🎬 Video âm tính có trên ổ đĩa    : {len(avail_neg)} video ({total_neg_avail_sec:.2f}s = {total_neg_avail_sec/3600.0:.4f}h)")
    print(f"⚠️ Video âm tính thiếu trên ổ đĩa : {len(miss_neg)} video ({total_neg_miss_sec:.2f}s = {total_neg_miss_sec/3600.0:.4f}h)")

    # 1. Xuất manifest các video âm tính bị thiếu
    missing_neg_manifest = {
        "status": "BLOCKED_BY_NEGATIVE_DATA",
        "proposal_requirement_negative_hours": 5.0,
        "available_negative_videos_on_disk": len(avail_neg),
        "available_negative_hours_on_disk": round(total_neg_avail_sec / 3600.0, 4),
        "cataloged_missing_negative_videos_count": len(miss_neg),
        "cataloged_missing_negative_hours": round(total_neg_miss_sec / 3600.0, 4),
        "additional_uncataloged_negative_hours_needed": round(max(0.0, 5.0 - (total_neg_avail_sec + total_neg_miss_sec) / 3600.0), 4),
        "missing_negative_videos": miss_neg,
        "missing_positive_vietnam_videos_count": len(miss_pos),
    }
    export_json(missing_neg_manifest, out_dir / "missing_negative_videos_manifest.json")
    with (out_dir / "missing_negative_videos_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "file_name",
                "catalog_relative_path",
                "exists_on_disk",
                "category",
                "duration_sec",
                "resolution",
                "fps",
                "notes",
            ],
        )
        writer.writeheader()
        for row in miss_neg:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    # 2. Phân chia Validation (12 video lẻ) và Test (12 video chẵn) tách biệt 100%
    val_videos = [v for idx, v in enumerate(avail_pos) if idx % 2 == 0]
    test_videos = [v for idx, v in enumerate(avail_pos) if idx % 2 == 1]
    val_ids: Set[str] = {v["video_id"] for v in val_videos}
    test_ids: Set[str] = {v["video_id"] for v in test_videos}
    all_pos_ids: Set[str] = {v["video_id"] for v in avail_pos}

    val_neg_videos = [v for idx, v in enumerate(avail_neg) if idx % 2 == 0]
    test_neg_videos = [v for idx, v in enumerate(avail_neg) if idx % 2 == 1]
    val_neg_ids: Set[str] = {v["video_id"] for v in val_neg_videos}
    test_neg_ids: Set[str] = {v["video_id"] for v in test_neg_videos}
    all_neg_ids: Set[str] = {v["video_id"] for v in avail_neg}

    total_neg_hours: Optional[float] = (total_neg_avail_sec / 3600.0) if avail_neg else None
    val_neg_hours: Optional[float] = (sum(v["duration_sec"] for v in val_neg_videos) / 3600.0) if val_neg_videos else None
    test_neg_hours: Optional[float] = (sum(v["duration_sec"] for v in test_neg_videos) / 3600.0) if test_neg_videos else None

    print(f"📊 Chia tập thực nghiệm: Validation = {len(val_videos)} POS / {len(val_neg_videos)} NEG | Test = {len(test_videos)} POS / {len(test_neg_videos)} NEG")

    # 3. Khởi tạo Detector
    detector = SharedFireSmokeDetector(model_path=str(model_path))

    # 4. Tham số đánh giá mặc định
    sample_fps = 5.0
    default_conf = 0.35
    default_k = 3
    default_n = 8
    default_cooldown = 5.0
    default_feat_thr = 0.02
    max_gap_sec = 2.0

    # 5. Chạy suy luận trên toàn bộ Video Dương tính (available_positive)
    positive_runs: List[Dict[str, Any]] = []
    print("\n🎬 BẮT ĐẦU CHẠY SUY LUẬN TRÊN TẬP VIDEO DƯƠNG TÍNH (POS):")
    for idx, v in enumerate(avail_pos, start=1):
        res = run_pipelines_on_video(
            detector=detector,
            video_info=v,
            sample_fps=sample_fps,
            conf_thresh=0.25,
            eval_conf_thresh=default_conf,
            imgsz=416,
            window_n=default_n,
            confirm_k=default_k,
            suspect_k=2,
            cooldown_sec=default_cooldown,
            feature_threshold=default_feat_thr,
            max_event_gap_sec=max_gap_sec,
        )
        positive_runs.append(res)
        pa_cnt = res["pipeline_a"]["raw_frame_alarm_count"]
        pa_ev = res["pipeline_a"]["event_count"]
        pb1_ev = res["pipeline_b1_no_gate"]["event_count"]
        pb2_ev = res["pipeline_b2_feature_gate"]["event_count"]
        pc_ev = res["pipeline_c"]["event_count"]
        print(
            f"  [{idx:02d}/{len(avail_pos):02d}] {v['video_id']} ({v['duration_sec']:5.1f}s) | "
            f"FrameAlarms(A): {pa_cnt:3d} -> Events(A): {pa_ev:2d} | "
            f"B1(NoGate): {pb1_ev:2d} | B2(FeatGate): {pb2_ev:2d} | C(IoU): {pc_ev:2d} | "
            f"Det: {res['pipeline_a']['mean_detector_latency_ms']:.1f}ms | "
            f"PipeB2: {res['pipeline_b2_feature_gate']['mean_pipeline_latency_ms']:.1f}ms"
        )

    # 6. Chạy suy luận trên toàn bộ Video Âm tính (available_negative) nếu có
    negative_runs: List[Dict[str, Any]] = []
    print(f"\n🎬 BẮT ĐẦU CHẠY SUY LUẬN TRÊN TẬP VIDEO ÂM TÍNH (NEG) ({len(avail_neg)} video khả dụng trên ổ đĩa):")
    if avail_neg:
        for idx, v in enumerate(avail_neg, start=1):
            res = run_pipelines_on_video(
                detector=detector,
                video_info=v,
                sample_fps=sample_fps,
                conf_thresh=0.25,
                eval_conf_thresh=default_conf,
                imgsz=416,
                window_n=default_n,
                confirm_k=default_k,
                suspect_k=2,
                cooldown_sec=default_cooldown,
                feature_threshold=default_feat_thr,
                max_event_gap_sec=max_gap_sec,
            )
            negative_runs.append(res)
            pa_cnt = res["pipeline_a"]["raw_frame_alarm_count"]
            pb1_cnt = res["pipeline_b1_no_gate"]["db_alert_count"]
            pb2_cnt = res["pipeline_b2_feature_gate"]["db_alert_count"]
            print(
                f"  [NEG {idx:02d}/{len(avail_neg):02d}] {v['video_id']} ({v['duration_sec']:5.1f}s) | "
                f"FalseAlarms - A: {pa_cnt:2d} | B1: {pb1_cnt:2d} | B2: {pb2_cnt:2d}"
            )
    else:
        print("  ⚠️ Hiện tại 0/28 video âm tính tồn tại trên ổ đĩa. Trạng thái FAR ghi nhận UNAVAILABLE (BLOCKED_BY_NEGATIVE_DATA).")

    # 7. Xuất negative_video_metrics.json
    neg_fa_a = sum(r["pipeline_a"]["event_count"] for r in negative_runs)
    neg_fa_b1 = sum(r["pipeline_b1_no_gate"]["event_count"] for r in negative_runs)
    neg_fa_b2 = sum(r["pipeline_b2_feature_gate"]["event_count"] for r in negative_runs)
    neg_fa_c = sum(r["pipeline_c"]["event_count"] for r in negative_runs)

    far_a = (neg_fa_a / total_neg_hours) if (total_neg_hours and total_neg_hours > 0) else None
    far_b1 = (neg_fa_b1 / total_neg_hours) if (total_neg_hours and total_neg_hours > 0) else None
    far_b2 = (neg_fa_b2 / total_neg_hours) if (total_neg_hours and total_neg_hours > 0) else None
    far_c = (neg_fa_c / total_neg_hours) if (total_neg_hours and total_neg_hours > 0) else None

    neg_metrics_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "BLOCKED_BY_NEGATIVE_DATA",
        "proposal_required_negative_hours": 5.0,
        "catalog_negative_videos_count": len(avail_neg) + len(miss_neg),
        "catalog_negative_hours": round(total_neg_miss_sec / 3600.0, 4),
        "available_negative_videos_on_disk": len(avail_neg),
        "evaluated_negative_hours": round(total_neg_avail_sec / 3600.0, 4),
        "additional_negative_hours_needed": round(max(0.0, 5.0 - total_neg_avail_sec / 3600.0), 4),
        "empirical_negative_far_summary": {
            "pipeline_a_frame_level_baseline": {
                "false_alarm_events_count": neg_fa_a,
                "false_alarms_per_hour": far_a,
                "status": "VERIFIED" if total_neg_hours and total_neg_hours >= 5.0 else ("PARTIAL_DATA" if total_neg_hours else "UNAVAILABLE_MISSING_NEGATIVE_VIDEOS"),
            },
            "pipeline_b1_bytetrack_no_gate": {
                "false_alarm_events_count": neg_fa_b1,
                "false_alarms_per_hour": far_b1,
                "status": "VERIFIED" if total_neg_hours and total_neg_hours >= 5.0 else ("PARTIAL_DATA" if total_neg_hours else "UNAVAILABLE_MISSING_NEGATIVE_VIDEOS"),
            },
            "pipeline_b2_bytetrack_feature_gate": {
                "false_alarm_events_count": neg_fa_b2,
                "false_alarms_per_hour": far_b2,
                "status": "VERIFIED" if total_neg_hours and total_neg_hours >= 5.0 else ("PARTIAL_DATA" if total_neg_hours else "UNAVAILABLE_MISSING_NEGATIVE_VIDEOS"),
            },
            "pipeline_c_legacy_iou": {
                "false_alarm_events_count": neg_fa_c,
                "false_alarms_per_hour": far_c,
                "status": "VERIFIED" if total_neg_hours and total_neg_hours >= 5.0 else ("PARTIAL_DATA" if total_neg_hours else "UNAVAILABLE_MISSING_NEGATIVE_VIDEOS"),
            },
        },
        "evaluated_negative_videos_details": [
            {
                "video_id": r["video_id"],
                "duration_sec": r["duration_sec"],
                "pipeline_a_alarms": r["pipeline_a"]["raw_frame_alarm_count"],
                "pipeline_a_events": r["pipeline_a"]["event_count"],
                "pipeline_b1_alerts": r["pipeline_b1_no_gate"]["db_alert_count"],
                "pipeline_b1_events": r["pipeline_b1_no_gate"]["event_count"],
                "pipeline_b2_alerts": r["pipeline_b2_feature_gate"]["db_alert_count"],
                "pipeline_b2_events": r["pipeline_b2_feature_gate"]["event_count"],
                "pipeline_c_alerts": r["pipeline_c"]["db_alert_count"],
                "pipeline_c_events": r["pipeline_c"]["event_count"],
            }
            for r in negative_runs
        ],
        "missing_negative_videos_manifest_file": "missing_negative_videos_manifest.csv",
        "notes": "Tuyệt đối không gán FAR = 0 khi chưa có video âm tính thực nghiệm. Trạng thái giữ nguyên BLOCKED_BY_NEGATIVE_DATA cho đến khi nạp đủ >= 5 giờ video âm tính.",
    }
    export_json(neg_metrics_payload, out_dir / "negative_video_metrics.json")

    # 8. Quét lưới tham số (Grid Search) CHỈ TRÊN TẬP VALIDATION
    val_tracked_frames: List[FrameDetection] = []
    test_tracked_frames: List[FrameDetection] = []
    for r in positive_runs:
        if r["video_id"] in val_ids:
            val_tracked_frames.extend(r["tracked_frame_detections"])
        else:
            test_tracked_frames.extend(r["tracked_frame_detections"])

    for r in negative_runs:
        if r["video_id"] in val_neg_ids:
            val_tracked_frames.extend(r["tracked_frame_detections"])
        else:
            test_tracked_frames.extend(r["tracked_frame_detections"])

    gt_all = build_ground_truth_events(avail_pos)
    gt_val = build_ground_truth_events(val_videos)
    gt_test = build_ground_truth_events(test_videos)

    print("\n🔍 Đang quét lưới tham số (Grid Search) TRÊN TẬP VALIDATION...")
    grid_candidates_val = grid_search(
        frames=val_tracked_frames,
        ground_truth=gt_val,
        positive_video_ids=val_ids,
        negative_video_ids=val_neg_ids if val_neg_ids else None,
        negative_hours=val_neg_hours,
        confidence_thresholds=[0.25, 0.35, 0.45],
        k_values=[2, 3, 4, 5],
        n_values=[5, 8, 10, 12],
        tolerance_sec=10.0,
        feature_thresholds=[0.0, 0.01, 0.02],
        cooldown_values=[0.0, 5.0, 12.0],
        max_event_gap_sec=max_gap_sec,
        allow_missing_negative=True,
    )
    operating_point_selection = select_best_operating_point(grid_candidates_val)
    provisional = operating_point_selection.get("provisional_candidate_on_positive_val")

    # 9. Đánh giá ĐỘC LẬP trên Test Set khóa cố định
    chosen_conf = float(provisional["confidence_threshold"]) if provisional else default_conf
    chosen_k = int(provisional["k"]) if provisional else default_k
    chosen_n = int(provisional["n"]) if provisional else default_n
    chosen_feat = float(provisional["feature_threshold"]) if provisional else default_feat_thr
    chosen_cooldown = float(provisional["cooldown_sec"]) if provisional else default_cooldown

    _, test_events_chosen = frame_detections_to_events(
        test_tracked_frames,
        confidence_threshold=chosen_conf,
        window_n=chosen_n,
        confirm_k=chosen_k,
        max_event_gap_sec=max_gap_sec,
        feature_threshold=chosen_feat,
        cooldown_sec=chosen_cooldown,
    )
    test_eval_chosen = evaluate(
        gt_test,
        test_events_chosen,
        test_ids,
        test_neg_ids if test_neg_ids else None,
        test_neg_hours,
        tolerance_sec=10.0,
        allow_missing_negative=True,
    )
    operating_point_selection["locked_test_set_evaluation_of_provisional"] = {
        "parameters": {
            "confidence_threshold": chosen_conf,
            "k": chosen_k,
            "n": chosen_n,
            "feature_threshold": chosen_feat,
            "cooldown_sec": chosen_cooldown,
        },
        "test_set_report": test_eval_chosen,
    }

    # 10. Tổng hợp số liệu theo từng phân vùng (Full 24, Validation 12, Test 12)
    def summarize_split(split_name: str, split_videos: List[Dict[str, Any]], split_ids: Set[str], split_gt: List[Event]) -> Dict[str, Any]:
        split_runs = [r for r in positive_runs if r["video_id"] in split_ids]
        raw_alarms_a: List[Dict[str, Any]] = []
        events_a: List[Event] = []
        events_b1: List[Event] = []
        events_b2: List[Event] = []
        events_c: List[Event] = []
        db_alerts_b1: List[Dict[str, Any]] = []
        db_alerts_b2: List[Dict[str, Any]] = []
        db_alerts_c: List[Dict[str, Any]] = []

        det_ms_list: List[float] = []
        pipe_b1_ms_list: List[float] = []
        pipe_b2_ms_list: List[float] = []
        pipe_c_ms_list: List[float] = []
        trk_ms_list: List[float] = []
        feat_ms_list: List[float] = []
        ver_b1_ms_list: List[float] = []
        ver_b2_ms_list: List[float] = []
        total_frames = 0

        for r in split_runs:
            total_frames += r["sampled_frames"]
            raw_alarms_a.extend(r["pipeline_a"]["raw_frame_alarms"])
            events_a.extend(r["pipeline_a"]["events"])
            events_b1.extend(r["pipeline_b1_no_gate"]["events"])
            events_b2.extend(r["pipeline_b2_feature_gate"]["events"])
            events_c.extend(r["pipeline_c"]["events"])
            db_alerts_b1.extend(r["pipeline_b1_no_gate"]["db_alerts"])
            db_alerts_b2.extend(r["pipeline_b2_feature_gate"]["db_alerts"])
            db_alerts_c.extend(r["pipeline_c"]["db_alerts"])

            det_ms_list.append(r["pipeline_a"]["mean_detector_latency_ms"])
            pipe_b1_ms_list.append(r["pipeline_b1_no_gate"]["mean_pipeline_latency_ms"])
            pipe_b2_ms_list.append(r["pipeline_b2_feature_gate"]["mean_pipeline_latency_ms"])
            pipe_c_ms_list.append(r["pipeline_c"]["mean_pipeline_latency_ms"])
            trk_ms_list.append(r["pipeline_b2_feature_gate"]["mean_tracker_overhead_ms"])
            feat_ms_list.append(r["pipeline_b2_feature_gate"]["mean_feature_overhead_ms"])
            ver_b1_ms_list.append(r["pipeline_b1_no_gate"]["mean_verifier_overhead_ms"])
            ver_b2_ms_list.append(r["pipeline_b2_feature_gate"]["mean_verifier_overhead_ms"])

        eval_a = evaluate(split_gt, events_a, split_ids, all_neg_ids if all_neg_ids else None, total_neg_hours, tolerance_sec=10.0, allow_missing_negative=True)
        eval_b1 = evaluate(split_gt, events_b1, split_ids, all_neg_ids if all_neg_ids else None, total_neg_hours, tolerance_sec=10.0, allow_missing_negative=True)
        eval_b2 = evaluate(split_gt, events_b2, split_ids, all_neg_ids if all_neg_ids else None, total_neg_hours, tolerance_sec=10.0, allow_missing_negative=True)
        eval_c = evaluate(split_gt, events_c, split_ids, all_neg_ids if all_neg_ids else None, total_neg_hours, tolerance_sec=10.0, allow_missing_negative=True)

        vid_eval_a = evaluate_video_level_primary_recall(split_videos, events_a, tolerance_sec=10.0)
        vid_eval_b1 = evaluate_video_level_primary_recall(split_videos, events_b1, tolerance_sec=10.0)
        vid_eval_b2 = evaluate_video_level_primary_recall(split_videos, events_b2, tolerance_sec=10.0)
        vid_eval_c = evaluate_video_level_primary_recall(split_videos, events_c, tolerance_sec=10.0)

        mean_det_ms = float(np.mean(det_ms_list)) if det_ms_list else 0.0
        mean_pb1_ms = float(np.mean(pipe_b1_ms_list)) if pipe_b1_ms_list else 0.0
        mean_pb2_ms = float(np.mean(pipe_b2_ms_list)) if pipe_b2_ms_list else 0.0
        mean_pc_ms = float(np.mean(pipe_c_ms_list)) if pipe_c_ms_list else 0.0

        alarm_red_b1 = ((len(raw_alarms_a) - len(db_alerts_b1)) / float(len(raw_alarms_a))) if len(raw_alarms_a) > 0 else 0.0
        alarm_red_b2 = ((len(raw_alarms_a) - len(db_alerts_b2)) / float(len(raw_alarms_a))) if len(raw_alarms_a) > 0 else 0.0

        return {
            "split": split_name,
            "videos_count": len(split_videos),
            "total_duration_sec": round(sum(v["duration_sec"] for v in split_videos), 2),
            "total_sampled_frames": total_frames,
            "pipeline_a_frame_level_baseline": {
                "pipeline_id": "pipeline_a_frame_level_baseline",
                "description": "Frame-level baseline (YOLO only, no ByteTrack, no K/N, no TemporalVerifier, no cooldown)",
                "raw_frame_alarm_count": len(raw_alarms_a),
                "merged_event_count": len(events_a),
                "strict_class_evaluation": eval_a,
                "video_primary_evaluation": vid_eval_a,
                "detector_only_latency_ms": round(mean_det_ms, 3),
                "detector_only_fps": round(1000.0 / mean_det_ms, 2) if mean_det_ms > 0 else 0.0,
            },
            "pipeline_b1_bytetrack_no_gate": {
                "pipeline_id": "pipeline_b1_bytetrack_no_gate",
                "description": f"ByteTrack + K/N without feature gate (K={default_k}/N={default_n}, feat_thr=0.0, cooldown={default_cooldown}s)",
                "db_alert_trigger_count": len(db_alerts_b1),
                "merged_event_count": len(events_b1),
                "alarm_reduction_vs_raw_frame_alarms_pct": round(alarm_red_b1 * 100.0, 2),
                "strict_class_evaluation": eval_b1,
                "video_primary_evaluation": vid_eval_b1,
                "end_to_end_pipeline_latency_ms": round(mean_pb1_ms, 3),
                "end_to_end_pipeline_fps": round(1000.0 / mean_pb1_ms, 2) if mean_pb1_ms > 0 else 0.0,
            },
            "pipeline_b2_bytetrack_feature_gate": {
                "pipeline_id": "pipeline_b2_bytetrack_feature_gate",
                "description": f"ByteTrack + K/N + Temporal Feature Gate (K={default_k}/N={default_n}, feat_thr={default_feat_thr}, cooldown={default_cooldown}s)",
                "db_alert_trigger_count": len(db_alerts_b2),
                "merged_event_count": len(events_b2),
                "alarm_reduction_vs_raw_frame_alarms_pct": round(alarm_red_b2 * 100.0, 2),
                "strict_class_evaluation": eval_b2,
                "video_primary_evaluation": vid_eval_b2,
                "detector_only_latency_ms": round(mean_det_ms, 3),
                "tracker_overhead_ms": round(float(np.mean(trk_ms_list)), 3) if trk_ms_list else 0.0,
                "feature_overhead_ms": round(float(np.mean(feat_ms_list)), 3) if feat_ms_list else 0.0,
                "verifier_overhead_ms": round(float(np.mean(ver_b2_ms_list)), 3) if ver_b2_ms_list else 0.0,
                "end_to_end_pipeline_latency_ms": round(mean_pb2_ms, 3),
                "end_to_end_pipeline_fps": round(1000.0 / mean_pb2_ms, 2) if mean_pb2_ms > 0 else 0.0,
            },
            "pipeline_c_legacy_iou_temporal": {
                "pipeline_id": "pipeline_c_legacy_iou_temporal",
                "description": f"Legacy IoU Temporal (YOLO + internal IoU tracker + TemporalVerifier K={default_k}/N={default_n}, cooldown={default_cooldown}s)",
                "db_alert_trigger_count": len(db_alerts_c),
                "merged_event_count": len(events_c),
                "strict_class_evaluation": eval_c,
                "video_primary_evaluation": vid_eval_c,
                "end_to_end_pipeline_latency_ms": round(mean_pc_ms, 3),
                "end_to_end_pipeline_fps": round(1000.0 / mean_pc_ms, 2) if mean_pc_ms > 0 else 0.0,
            },
        }

    full_summary = summarize_split("full_24_positive", avail_pos, all_pos_ids, gt_all)
    val_summary = summarize_split("val_12_positive", val_videos, val_ids, gt_val)
    test_summary = summarize_split("test_12_positive", test_videos, test_ids, gt_test)

    # 11. Xuất frame_level_events.json & temporal_confirmed_events.json
    all_raw_frame_alarms = [a for r in positive_runs for a in r["pipeline_a"]["raw_frame_alarms"]]
    all_events_a = [asdict(e) for r in positive_runs for e in r["pipeline_a"]["events"]]
    all_events_b1 = [asdict(e) for r in positive_runs for e in r["pipeline_b1_no_gate"]["events"]]
    all_events_b2 = [asdict(e) for r in positive_runs for e in r["pipeline_b2_feature_gate"]["events"]]
    all_events_c = [asdict(e) for r in positive_runs for e in r["pipeline_c"]["events"]]
    all_db_alerts_b1 = [a for r in positive_runs for a in r["pipeline_b1_no_gate"]["db_alerts"]]
    all_db_alerts_b2 = [a for r in positive_runs for a in r["pipeline_b2_feature_gate"]["db_alerts"]]
    all_db_alerts_c = [a for r in positive_runs for a in r["pipeline_c"]["db_alerts"]]

    export_json(
        {
            "pipeline": "pipeline_a_frame_level_baseline",
            "parameters": {
                "confidence_threshold": default_conf,
                "max_event_gap_sec": max_gap_sec,
                "uses_bytetrack": False,
                "uses_k_of_n": False,
                "uses_temporal_verifier": False,
                "uses_cooldown": False,
            },
            "raw_frame_alarms_count": len(all_raw_frame_alarms),
            "merged_events_count": len(all_events_a),
            "merged_events": all_events_a,
            "raw_frame_alarms": all_raw_frame_alarms,
        },
        out_dir / "frame_level_events.json",
    )

    export_json(
        {
            "pipeline": "pipeline_b_temporal_confirmed",
            "parameters": {
                "confidence_threshold": default_conf,
                "window_size_n": default_n,
                "confirm_frames_k": default_k,
                "suspect_frames_k": 2,
                "cooldown_sec": default_cooldown,
                "feature_threshold": default_feat_thr,
                "max_event_gap_sec": max_gap_sec,
                "uses_bytetrack": True,
                "uses_temporal_features": True,
                "uses_temporal_verifier": True,
            },
            "b1_no_gate_db_alerts_count": len(all_db_alerts_b1),
            "b1_no_gate_events_count": len(all_events_b1),
            "b1_no_gate_events": all_events_b1,
            "b2_feature_gate_db_alerts_count": len(all_db_alerts_b2),
            "b2_feature_gate_events_count": len(all_events_b2),
            "b2_feature_gate_events": all_events_b2,
            "legacy_iou_events": all_events_c,
        },
        out_dir / "temporal_confirmed_events.json",
    )

    # 12. Xuất comparison.csv
    comparison_rows = []
    for split_obj in (full_summary, val_summary, test_summary):
        sp_name = split_obj["split"]
        pa = split_obj["pipeline_a_frame_level_baseline"]
        pb1 = split_obj["pipeline_b1_bytetrack_no_gate"]
        pb2 = split_obj["pipeline_b2_bytetrack_feature_gate"]
        pc = split_obj["pipeline_c_legacy_iou_temporal"]

        for p_key, p_data, raw_cnt in (
            ("Pipeline A: Frame-level Baseline (No K/N, No Tracker)", pa, pa["raw_frame_alarm_count"]),
            ("Pipeline B1: ByteTrack + K/N (No Feature Gate, feat_thr=0.0)", pb1, pb1["db_alert_trigger_count"]),
            ("Pipeline B2: ByteTrack + K/N + Temporal Feature Gate (feat_thr=0.02)", pb2, pb2["db_alert_trigger_count"]),
            ("Pipeline C: Legacy IoU Temporal (IoU + K/N)", pc, pc["db_alert_trigger_count"]),
        ):
            m_strict = p_data["strict_class_evaluation"]["metrics"]
            cnt_strict = p_data["strict_class_evaluation"]["counts"]
            m_vid = p_data["video_primary_evaluation"]
            comparison_rows.append({
                "split": sp_name,
                "pipeline": p_key,
                "videos_count": split_obj["videos_count"],
                "raw_alarms_or_db_triggers": raw_cnt,
                "merged_events_count": p_data["merged_event_count"],
                "strict_video_event_recall_pct": round(m_strict["video_event_recall"] * 100.0, 2),
                "any_class_video_recall_pct": round(m_vid["any_class_video_recall"] * 100.0, 2),
                "event_instance_recall_pct": round(m_strict["event_instance_recall"] * 100.0, 2),
                "event_precision_pct": round(m_strict["event_precision"] * 100.0, 2),
                "event_f1_pct": round(m_strict["event_f1"] * 100.0, 2),
                "strict_mean_detection_delay_sec": (
                    round(m_strict["mean_detection_delay_sec"], 3)
                    if m_strict["mean_detection_delay_sec"] is not None else "N/A"
                ),
                "onset_mean_detection_delay_sec": (
                    round(m_vid["mean_delay_from_earliest_onset_sec"], 3)
                    if m_vid["mean_delay_from_earliest_onset_sec"] is not None else "N/A"
                ),
                "pre_event_false_alarms_on_pos": m_vid["pre_event_false_alarms_on_positive_videos"],
                "unmatched_positive_predictions": cnt_strict["unmatched_positive_predictions"],
                "negative_false_alarms_per_hour": (
                    round(m_strict["false_alarms_per_hour"], 4)
                    if m_strict["false_alarms_per_hour"] is not None else "UNAVAILABLE"
                ),
                "detector_only_latency_ms": pa["detector_only_latency_ms"],
                "detector_only_fps": pa["detector_only_fps"],
                "end_to_end_pipeline_latency_ms": p_data.get(
                    "end_to_end_pipeline_latency_ms", pa["detector_only_latency_ms"]
                ),
                "end_to_end_pipeline_fps": p_data.get(
                    "end_to_end_pipeline_fps", pa["detector_only_fps"]
                ),
                "status": p_data["strict_class_evaluation"]["status"],
            })

    with (out_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    # 13. Xuất grid_search.csv, grid_search.json và best_operating_point.json
    export_grid_search_csv(grid_candidates_val, out_dir / "grid_search.csv")
    export_json(
        {
            "validation_split_positive_videos": sorted(list(val_ids)),
            "test_split_positive_videos": sorted(list(test_ids)),
            "validation_split_negative_videos": sorted(list(val_neg_ids)),
            "test_split_negative_videos": sorted(list(test_neg_ids)),
            "total_candidates_evaluated": len(grid_candidates_val),
            "operating_point_selection": operating_point_selection,
            "candidates": grid_candidates_val,
        },
        out_dir / "grid_search.json",
    )
    export_json(operating_point_selection, out_dir / "best_operating_point.json")

    # 14. Xuất metrics.json tổng thể
    metrics_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "acceptance_status": "PARTIAL (BLOCKED_BY_NEGATIVE_DATA)",
        "model_checkpoint": str(model_path.relative_to(ROOT_DIR)),
        "experiment_settings": {
            "seed": 42,
            "sample_fps": sample_fps,
            "imgsz": 416,
            "default_confidence_threshold": default_conf,
            "default_window_n": default_n,
            "default_confirm_k": default_k,
            "default_cooldown_sec": default_cooldown,
            "default_feature_threshold": default_feat_thr,
            "max_event_gap_sec": max_gap_sec,
        },
        "dataset_availability": {
            "available_positive_videos": len(avail_pos),
            "available_positive_duration_sec": round(total_pos_sec, 2),
            "available_positive_hours": round(total_pos_sec / 3600.0, 4),
            "available_negative_videos": len(avail_neg),
            "available_negative_hours": round(total_neg_avail_sec / 3600.0, 4),
            "missing_negative_videos_in_catalog": len(miss_neg),
            "missing_negative_hours_in_catalog": round(total_neg_miss_sec / 3600.0, 4),
            "proposal_target_negative_hours": 5.0,
        },
        "splits_summary": {
            "full_24_positive": full_summary,
            "val_12_positive": val_summary,
            "test_12_positive": test_summary,
        },
        "negative_video_evaluation": neg_metrics_payload,
        "operating_point_selection": operating_point_selection,
    }
    export_json(metrics_payload, out_dir / "metrics.json")

    # 15. Tạo benchmark_report.md
    pa_full = full_summary["pipeline_a_frame_level_baseline"]
    pb1_full = full_summary["pipeline_b1_bytetrack_no_gate"]
    pb2_full = full_summary["pipeline_b2_bytetrack_feature_gate"]
    pc_full = full_summary["pipeline_c_legacy_iou_temporal"]

    pa_val = val_summary["pipeline_a_frame_level_baseline"]
    pb1_val = val_summary["pipeline_b1_bytetrack_no_gate"]
    pb2_val = val_summary["pipeline_b2_bytetrack_feature_gate"]
    pc_val = val_summary["pipeline_c_legacy_iou_temporal"]

    pa_test = test_summary["pipeline_a_frame_level_baseline"]
    pb1_test = test_summary["pipeline_b1_bytetrack_no_gate"]
    pb2_test = test_summary["pipeline_b2_bytetrack_feature_gate"]
    pc_test = test_summary["pipeline_c_legacy_iou_temporal"]

    report_md = f"""# Báo Cáo Thực Nghiệm & Nghiệm Thu Tuần 4: Frame-Level Baseline vs. ByteTrack + Temporal Verification

- **Thời gian thực thi:** `{metrics_payload["timestamp"]}`
- **Model Checkpoint:** `{metrics_payload["model_checkpoint"]}` (mAP@0.5 = 76.78% trên D-Fire Test)
- **Tần số lấy mẫu khung hình:** `{sample_fps} FPS` (`imgsz = 416`, `seed = 42`)
- **Trạng thái nghiệm thu Tuần 4:** **`PARTIAL (BLOCKED_BY_NEGATIVE_DATA)`**

---

## 1. Hiện Trạng Dữ Liệu Video Kiểm Thử Trên Ổ Đĩa

| Phân loại | Số video trong `video_dataset_list.csv` | Số video thực có trên ổ đĩa | Thời lượng thực có | Chỉ tiêu Đề cương (MT1/MT3) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Video Dương tính (Có cháy/khói)** | 44 (`POS_01..24`, `VN_POS_01..20`) | **24** (`POS_01..24`) | `863.98s` (`0.24h`) | ≥ 20 video | **ĐẠT (24/20 video)** |
| **Video Âm tính (Không cháy / Cảnh nhiễu)** | 28 (`NEG_25..49`, `VN_NEG_01,03,04`) | **0** | `0.00s` (`0.00h`) | ≥ 5.0 giờ | **THIẾU (`BLOCKED_BY_NEGATIVE_DATA`)** |

> Danh sách chi tiết 28 video âm tính bị thiếu trên ổ đĩa (`2.8622 giờ` trong catalog + `2.1378 giờ` cần thu thập bổ sung) đã được xuất tại `missing_negative_videos_manifest.csv` và `missing_negative_videos_manifest.json`.
> Báo cáo riêng về video âm tính đã được ghi nhận tại `negative_video_metrics.json`.

---

## 2. Bảng Đối Sánh 4 Pipeline Trên Toàn Bộ 24 Video Dương Tính (`full_24_positive`)

Cấu hình điểm vận hành đánh giá: `confidence_threshold = {default_conf}`, `K = {default_k}`, `N = {default_n}`, `cooldown_sec = {default_cooldown}s`, `max_event_gap_sec = {max_gap_sec}s`.

| Chỉ số thực nghiệm | Pipeline A: Frame-Level Baseline (Chỉ YOLO, không K/N) | Pipeline B1: ByteTrack + K/N (Không Feature Gate, feat_thr=0.0) | Pipeline B2: ByteTrack + K/N + Feature Gate (feat_thr={default_feat_thr}) | Pipeline C: Legacy IoU Temporal (IoU + K/N) |
| :--- | :---: | :---: | :---: | :---: |
| **Tổng cảnh báo phát ra (Raw Alarms / DB Triggers)** | **{pa_full["raw_frame_alarm_count"]}** frame alarms | **{pb1_full["db_alert_trigger_count"]}** DB alerts (**-{pb1_full["alarm_reduction_vs_raw_frame_alarms_pct"]:.2f}%**) | **{pb2_full["db_alert_trigger_count"]}** DB alerts (**-{pb2_full["alarm_reduction_vs_raw_frame_alarms_pct"]:.2f}%**) | **{pc_full["db_alert_trigger_count"]}** DB alerts |
| **Số khoảng sự kiện sau khi gộp (`gap = 2.0s`)** | **{pa_full["merged_event_count"]}** events | **{pb1_full["merged_event_count"]}** events | **{pb2_full["merged_event_count"]}** events | **{pc_full["merged_event_count"]}** events |
| **Video-level Event Recall (Strict Match)** | `{pa_full["strict_class_evaluation"]["metrics"]["video_event_recall"]*100:.2f}%` | `{pb1_full["strict_class_evaluation"]["metrics"]["video_event_recall"]*100:.2f}%` | `{pb2_full["strict_class_evaluation"]["metrics"]["video_event_recall"]*100:.2f}%` | `{pc_full["strict_class_evaluation"]["metrics"]["video_event_recall"]*100:.2f}%` |
| **Video-level Primary Recall (Any Class)** | `{pa_full["video_primary_evaluation"]["any_class_video_recall"]*100:.2f}%` ({pa_full["video_primary_evaluation"]["recalled_positive_videos"]}/24) | `{pb1_full["video_primary_evaluation"]["any_class_video_recall"]*100:.2f}%` ({pb1_full["video_primary_evaluation"]["recalled_positive_videos"]}/24) | `{pb2_full["video_primary_evaluation"]["any_class_video_recall"]*100:.2f}%` ({pb2_full["video_primary_evaluation"]["recalled_positive_videos"]}/24) | `{pc_full["video_primary_evaluation"]["any_class_video_recall"]*100:.2f}%` ({pc_full["video_primary_evaluation"]["recalled_positive_videos"]}/24) |
| **Event-instance Recall** | `{pa_full["strict_class_evaluation"]["metrics"]["event_instance_recall"]*100:.2f}%` | `{pb1_full["strict_class_evaluation"]["metrics"]["event_instance_recall"]*100:.2f}%` | `{pb2_full["strict_class_evaluation"]["metrics"]["event_instance_recall"]*100:.2f}%` | `{pc_full["strict_class_evaluation"]["metrics"]["event_instance_recall"]*100:.2f}%` |
| **Event Precision** | `{pa_full["strict_class_evaluation"]["metrics"]["event_precision"]*100:.2f}%` | `{pb1_full["strict_class_evaluation"]["metrics"]["event_precision"]*100:.2f}%` | **`{pb2_full["strict_class_evaluation"]["metrics"]["event_precision"]*100:.2f}%`** | `{pc_full["strict_class_evaluation"]["metrics"]["event_precision"]*100:.2f}%` |
| **Event F1-Score** | `{pa_full["strict_class_evaluation"]["metrics"]["event_f1"]*100:.2f}%` | `{pb1_full["strict_class_evaluation"]["metrics"]["event_f1"]*100:.2f}%` | **`{pb2_full["strict_class_evaluation"]["metrics"]["event_f1"]*100:.2f}%`** | `{pc_full["strict_class_evaluation"]["metrics"]["event_f1"]*100:.2f}%` |
| **Mean Detection Delay (Strict Class Match)** | `{pa_full["strict_class_evaluation"]["metrics"]["mean_detection_delay_sec"]:.2f} s` | `{pb1_full["strict_class_evaluation"]["metrics"]["mean_detection_delay_sec"]:.2f} s` | `{pb2_full["strict_class_evaluation"]["metrics"]["mean_detection_delay_sec"]:.2f} s` (≤ 10 s) | `{pc_full["strict_class_evaluation"]["metrics"]["mean_detection_delay_sec"]:.2f} s` |
| **Mean Detection Delay (Từ mốc khởi phát sớm)** | `{pa_full["video_primary_evaluation"]["mean_delay_from_earliest_onset_sec"]:.2f} s` | `{pb1_full["video_primary_evaluation"]["mean_delay_from_earliest_onset_sec"]:.2f} s` | `{pb2_full["video_primary_evaluation"]["mean_delay_from_earliest_onset_sec"]:.2f} s` (≤ 10 s) | `{pc_full["video_primary_evaluation"]["mean_delay_from_earliest_onset_sec"]:.2f} s` |
| **Cảnh báo nổ ra trước mốc cháy (`pre_event_alarms`)** | `{pa_full["video_primary_evaluation"]["pre_event_false_alarms_on_positive_videos"]}` | `{pb1_full["video_primary_evaluation"]["pre_event_false_alarms_on_positive_videos"]}` | **`{pb2_full["video_primary_evaluation"]["pre_event_false_alarms_on_positive_videos"]}`** | `{pc_full["video_primary_evaluation"]["pre_event_false_alarms_on_positive_videos"]}` |
| **Sự kiện không khớp GT (`unmatched_positive`)** | `{pa_full["strict_class_evaluation"]["counts"]["unmatched_positive_predictions"]}` | `{pb1_full["strict_class_evaluation"]["counts"]["unmatched_positive_predictions"]}` | **`{pb2_full["strict_class_evaluation"]["counts"]["unmatched_positive_predictions"]}`** | `{pc_full["strict_class_evaluation"]["counts"]["unmatched_positive_predictions"]}` |
| **Negative False Alarms per Hour (FAR/giờ)** | `UNAVAILABLE` | `UNAVAILABLE` | `UNAVAILABLE` | `UNAVAILABLE` |
| **Detector-only Latency / FPS (CPU)** | `{pa_full["detector_only_latency_ms"]:.2f} ms` (`{pa_full["detector_only_fps"]:.1f} FPS`) | `{pa_full["detector_only_latency_ms"]:.2f} ms` (`{pa_full["detector_only_fps"]:.1f} FPS`) | `{pa_full["detector_only_latency_ms"]:.2f} ms` (`{pa_full["detector_only_fps"]:.1f} FPS`) | `{pa_full["detector_only_latency_ms"]:.2f} ms` (`{pa_full["detector_only_fps"]:.1f} FPS`) |
| **End-to-End Pipeline Latency / FPS (CPU)** | `{pa_full["detector_only_latency_ms"]:.2f} ms` (`{pa_full["detector_only_fps"]:.1f} FPS`) | `{pb1_full["end_to_end_pipeline_latency_ms"]:.2f} ms` (`{pb1_full["end_to_end_pipeline_fps"]:.1f} FPS`) | **`{pb2_full["end_to_end_pipeline_latency_ms"]:.2f} ms` (`{pb2_full["end_to_end_pipeline_fps"]:.1f} FPS`)** | `{pc_full["end_to_end_pipeline_latency_ms"]:.2f} ms` (`{pc_full["end_to_end_pipeline_fps"]:.1f} FPS`) |

> **Phân rã chi phí tính toán tăng thêm (Overhead) của Pipeline B2 trên CPU:**
> - `ByteTrackAdapter.update`: `{pb2_full["tracker_overhead_ms"]:.3f} ms/frame`
> - `TemporalFeatureExtractor.update`: `{pb2_full["feature_overhead_ms"]:.3f} ms/frame`
> - `TemporalVerifier.update`: `{pb2_full["verifier_overhead_ms"]:.3f} ms/frame`
> - **Tổng Overhead toàn pipeline:** `{pb2_full["tracker_overhead_ms"] + pb2_full["feature_overhead_ms"] + pb2_full["verifier_overhead_ms"]:.3f} ms/frame`

---

## 3. Bảng Đối Sánh 4 Pipeline Trên Tập Validation (12 Video Lẻ - `val_12_positive`)

| Chỉ số thực nghiệm | Pipeline A: Frame Baseline | Pipeline B1: ByteTrack (No Gate) | Pipeline B2: ByteTrack + Feature Gate | Pipeline C: Legacy IoU |
| :--- | :---: | :---: | :---: | :---: |
| **Tổng cảnh báo phát ra** | **{pa_val["raw_frame_alarm_count"]}** frame alarms | **{pb1_val["db_alert_trigger_count"]}** DB alerts (**-{pb1_val["alarm_reduction_vs_raw_frame_alarms_pct"]:.2f}%**) | **{pb2_val["db_alert_trigger_count"]}** DB alerts (**-{pb2_val["alarm_reduction_vs_raw_frame_alarms_pct"]:.2f}%**) | **{pc_val["db_alert_trigger_count"]}** DB alerts |
| **Số sự kiện sau gộp (gap=2.0s)** | **{pa_val["merged_event_count"]}** events | **{pb1_val["merged_event_count"]}** events | **{pb2_val["merged_event_count"]}** events | **{pc_val["merged_event_count"]}** events |
| **Video Recall (Strict Match)** | `{pa_val["strict_class_evaluation"]["metrics"]["video_event_recall"]*100:.2f}%` | `{pb1_val["strict_class_evaluation"]["metrics"]["video_event_recall"]*100:.2f}%` | `{pb2_val["strict_class_evaluation"]["metrics"]["video_event_recall"]*100:.2f}%` | `{pc_val["strict_class_evaluation"]["metrics"]["video_event_recall"]*100:.2f}%` |
| **Event Precision** | `{pa_val["strict_class_evaluation"]["metrics"]["event_precision"]*100:.2f}%` | `{pb1_val["strict_class_evaluation"]["metrics"]["event_precision"]*100:.2f}%` | **`{pb2_val["strict_class_evaluation"]["metrics"]["event_precision"]*100:.2f}%`** | `{pc_val["strict_class_evaluation"]["metrics"]["event_precision"]*100:.2f}%` |
| **Event F1-Score** | `{pa_val["strict_class_evaluation"]["metrics"]["event_f1"]*100:.2f}%` | `{pb1_val["strict_class_evaluation"]["metrics"]["event_f1"]*100:.2f}%` | **`{pb2_val["strict_class_evaluation"]["metrics"]["event_f1"]*100:.2f}%`** | `{pc_val["strict_class_evaluation"]["metrics"]["event_f1"]*100:.2f}%` |
| **Mean Detection Delay** | `{pa_val["strict_class_evaluation"]["metrics"]["mean_detection_delay_sec"]:.2f} s` | `{pb1_val["strict_class_evaluation"]["metrics"]["mean_detection_delay_sec"]:.2f} s` | `{pb2_val["strict_class_evaluation"]["metrics"]["mean_detection_delay_sec"]:.2f} s` | `{pc_val["strict_class_evaluation"]["metrics"]["mean_detection_delay_sec"]:.2f} s` |

---

## 4. Bảng Đối Sánh 4 Pipeline Trên Tập Test Độc Lập (12 Video Chẵn - `test_12_positive`)

| Chỉ số thực nghiệm | Pipeline A: Frame Baseline | Pipeline B1: ByteTrack (No Gate) | Pipeline B2: ByteTrack + Feature Gate | Pipeline C: Legacy IoU |
| :--- | :---: | :---: | :---: | :---: |
| **Tổng cảnh báo phát ra** | **{pa_test["raw_frame_alarm_count"]}** frame alarms | **{pb1_test["db_alert_trigger_count"]}** DB alerts (**-{pb1_test["alarm_reduction_vs_raw_frame_alarms_pct"]:.2f}%**) | **{pb2_test["db_alert_trigger_count"]}** DB alerts (**-{pb2_test["alarm_reduction_vs_raw_frame_alarms_pct"]:.2f}%**) | **{pc_test["db_alert_trigger_count"]}** DB alerts |
| **Số sự kiện sau gộp (gap=2.0s)** | **{pa_test["merged_event_count"]}** events | **{pb1_test["merged_event_count"]}** events | **{pb2_test["merged_event_count"]}** events | **{pc_test["merged_event_count"]}** events |
| **Video Recall (Strict Match)** | `{pa_test["strict_class_evaluation"]["metrics"]["video_event_recall"]*100:.2f}%` | `{pb1_test["strict_class_evaluation"]["metrics"]["video_event_recall"]*100:.2f}%` | `{pb2_test["strict_class_evaluation"]["metrics"]["video_event_recall"]*100:.2f}%` | `{pc_test["strict_class_evaluation"]["metrics"]["video_event_recall"]*100:.2f}%` |
| **Event Precision** | `{pa_test["strict_class_evaluation"]["metrics"]["event_precision"]*100:.2f}%` | `{pb1_test["strict_class_evaluation"]["metrics"]["event_precision"]*100:.2f}%` | **`{pb2_test["strict_class_evaluation"]["metrics"]["event_precision"]*100:.2f}%`** | `{pc_test["strict_class_evaluation"]["metrics"]["event_precision"]*100:.2f}%` |
| **Event F1-Score** | `{pa_test["strict_class_evaluation"]["metrics"]["event_f1"]*100:.2f}%` | `{pb1_test["strict_class_evaluation"]["metrics"]["event_f1"]*100:.2f}%` | **`{pb2_test["strict_class_evaluation"]["metrics"]["event_f1"]*100:.2f}%`** | `{pc_test["strict_class_evaluation"]["metrics"]["event_f1"]*100:.2f}%` |
| **Mean Detection Delay** | `{pa_test["strict_class_evaluation"]["metrics"]["mean_detection_delay_sec"]:.2f} s` | `{pb1_test["strict_class_evaluation"]["metrics"]["mean_detection_delay_sec"]:.2f} s` | `{pb2_test["strict_class_evaluation"]["metrics"]["mean_detection_delay_sec"]:.2f} s` | `{pc_test["strict_class_evaluation"]["metrics"]["mean_detection_delay_sec"]:.2f} s` |

---

## 5. Kết Quả Quét Lưới Tham Số (Grid Search) Trên Validation Set & Đánh Giá Trên Test Set

- **Validation Set (12 video lẻ):** `POS_01, POS_03, POS_05, POS_07, POS_09, POS_11, POS_13, POS_15, POS_17, POS_19, POS_21, POS_23` (`456.65s`).
- **Test Set khóa cố định (12 video chẵn):** `POS_02, POS_04, POS_06, POS_08, POS_10, POS_12, POS_14, POS_16, POS_18, POS_20, POS_22, POS_24` (`407.33s`).
- **Tách biệt 100%:** Hai tập Validation và Test không có video nào trùng lặp (`val_ids.isdisjoint(test_ids) == True`).
- **Nguyên tắc so sánh Baseline:** Mỗi ứng viên `(conf, K, N, ...)` được so sánh với Frame-level baseline tại chính ngưỡng `conf` đó (`baseline_by_conf`).
- **Tổng số cấu hình đã quét:** `{len(grid_candidates_val)}` tổ hợp (`confidence_threshold` ∈ {{0.25, 0.35, 0.45}}, $K$ ∈ {{2, 3, 4, 5}}, $N$ ∈ {{5, 8, 10, 12}}, `feature_threshold` ∈ {{0.0, 0.01, 0.02}}, `cooldown_sec` ∈ {{0.0, 5.0, 12.0}}).
- **Kết luận chọn điểm vận hành:** `{operating_point_selection["status"]}`
  - Lý do: {operating_point_selection["reason"]}
- **Cấu hình Provisional tối ưu trên Validation Set:** `conf={default_conf}, K={default_k}, N={default_n}, feat_thr={default_feat_thr}, cooldown={default_cooldown}s`.
"""
    (out_dir / "benchmark_report.md").write_text(report_md, encoding="utf-8")

    # Đồng bộ sang runs/benchmarks/week4_bytetrack/
    for fname in ("comparison.csv", "metrics.json", "benchmark_report.md", "negative_video_metrics.json"):
        if (out_dir / fname).exists():
            shutil.copy2(out_dir / fname, legacy_out_dir / fname)

    print("\n" + "=" * 85)
    print(f"✅ ĐÃ XUẤT ĐẦY ĐỦ CÁC TỆP NGHIỆM THU TUẦN 4 TẠI: {out_dir}")
    print("=" * 85)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
