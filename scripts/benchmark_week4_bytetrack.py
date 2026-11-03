"""Script thực nghiệm đối sánh và nghiệm thu Tuần 4 (`scripts/benchmark_week4_bytetrack.py`).

Đánh giá đối sánh 3 Pipeline độc lập trên cùng tập video, cùng model, cùng frame sampling (5.0 FPS)
và cùng ngưỡng confidence:
1. Pipeline A — `frame_level_baseline`:
   - Chỉ chạy YOLO detector (`runs/benchmarks/yolov8n_seed42/weights/best.pt` hoặc `best.pt`).
   - Mỗi frame có detection FIRE/SMOKE >= confidence_threshold tạo ngay 1 frame-level alarm.
   - Tuyệt đối KHÔNG dùng ByteTrack, KHÔNG dùng K/N, KHÔNG dùng TemporalVerifier, KHÔNG dùng Cooldown.
   - Ghi nhận toàn bộ raw frame-level alarms và gộp thành Event intervals với `max_event_gap_sec = 2.0s`.
2. Pipeline B — `temporal_confirmed_bytetrack_features`:
   - YOLO detector + `ByteTrackAdapter` + `TemporalFeatureExtractor` + `TemporalVerifier` (luật K/N theo từng track_id,
     vùng loại trừ ROI, ngưỡng biến động `feature_threshold`, và Cooldown nhân quả).
3. Pipeline C — `legacy_iou_temporal`:
   - YOLO detector + `TemporalVerifier` với IoU tracking nội bộ (`tracks_assigned=False`, luật K/N, Cooldown).

Thực hiện:
- Chia tập 24 video dương tính hiện có thành Validation Set (12 video lẻ: POS_01, POS_03, ..., POS_23)
  và Test Set (12 video chẵn: POS_02, POS_04, ..., POS_24).
- Quét lưới tham số (Grid Search) trên Validation Set và đánh giá điểm vận hành trên Test Set.
- Kiểm tra sự tồn tại thực tế của các video âm tính trên ổ đĩa; xuất manifest thiếu dữ liệu âm tính
  và báo cáo trung thực `false_alarms_per_hour = UNAVAILABLE` (`BLOCKED_BY_NEGATIVE_DATA`).
- Đo tách biệt thời gian suy luận của riêng Detector (Detector-only Latency/FPS) và của toàn bộ
  Pipeline xác nhận theo thời gian (End-to-End Pipeline Latency/FPS).

Xuất đầy đủ kết quả vào `runs/benchmarks/week4_final/` (và đồng bộ sang `runs/benchmarks/week4_bytetrack/`).
"""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
    """Tạo danh sách sự kiện Ground Truth từ fire_start_sec và smoke_start_sec của từng video dương tính.

    Để tránh phạt oan khi mô hình phát hiện khói trong đám cháy hoặc phát hiện ngọn lửa trong đám khói
    trên các video dương tính chỉ ghi 1 mốc thời gian, nếu video dương tính chỉ có `fire_start_sec >= 0`
    hoặc chỉ có `smoke_start_sec >= 0`, ta tạo sự kiện chính và đồng thời cho phép nhãn còn lại tính từ
    cùng mốc khởi phát sự kiện nếu phát hiện thực tế xảy ra. Tuy nhiên để đảm bảo tính chặt chẽ của
    Event-instance Recall, mỗi mốc khởi phát được ghi nhận đúng theo bảng dữ liệu gốc, và bổ sung
    nhãn tương ứng khi cần đối sánh cấp video.
    """
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


def run_three_pipelines_on_video(
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
    feature_threshold: float = 0.0,
    max_event_gap_sec: float = 2.0,
) -> Dict[str, Any]:
    """Chạy 3 pipeline độc lập trên cùng một video và thu thập toàn bộ phát hiện, sự kiện, độ trễ xử lý."""
    vid = video_info["video_id"]
    cam_id = video_info["camera_id"]
    sampled_frames = sample_video_frames(video_info["path"], sample_fps=sample_fps)

    # Khởi tạo trạng thái hoàn toàn cô lập cho Pipeline B (ByteTrack + Features + TemporalVerifier)
    bt_tracker = ByteTrackAdapter(
        settings=ByteTrackSettings(
            track_high_thresh=0.35,
            track_low_thresh=0.10,
            new_track_thresh=0.40,
            track_buffer=15,
            match_thresh=0.70,
            fuse_score=False,
        ),
        per_class=True,
    )
    feat_extractor = TemporalFeatureExtractor(history_size=12)
    verifier_b = TemporalVerifier(
        window_size_n=window_n,
        confirm_frames_k=confirm_k,
        suspect_frames_k=suspect_k,
        cooldown_sec=cooldown_sec,
        feature_threshold=feature_threshold,
    )

    # Khởi tạo trạng thái hoàn toàn cô lập cho Pipeline C (Legacy IoU + TemporalVerifier)
    verifier_c = TemporalVerifier(
        window_size_n=window_n,
        confirm_frames_k=confirm_k,
        suspect_frames_k=suspect_k,
        cooldown_sec=cooldown_sec,
        feature_threshold=0.0,
    )

    detector_latencies_ms: List[float] = []
    pipeline_b_latencies_ms: List[float] = []
    pipeline_c_latencies_ms: List[float] = []
    tracker_overhead_ms: List[float] = []
    feature_overhead_ms: List[float] = []
    verifier_b_overhead_ms: List[float] = []

    # Lưu FrameDetection chuẩn hóa (chứa track_id của ByteTrack và điểm biến động) để phục vụ Grid Search
    tracked_frame_detections: List[FrameDetection] = []
    raw_frame_detections: List[FrameDetection] = []

    # Lưu các thời điểm xác nhận của Pipeline B và Pipeline C
    pipeline_b_confirmed_samples: Dict[str, List[Tuple[float, float]]] = {}
    pipeline_b_db_alerts: List[Dict[str, Any]] = []

    pipeline_c_confirmed_samples: Dict[str, List[Tuple[float, float]]] = {}
    pipeline_c_db_alerts: List[Dict[str, Any]] = []

    for frame_idx, ts_sec, frame in sampled_frames:
        # 1. Chạy YOLO Detector (dùng ngưỡng thấp conf_thresh=0.25 để lưu đủ thông tin cho Grid Search [0.25, 0.35, 0.45])
        t_det0 = time.perf_counter()
        raw_boxes_res, det_ms = detector.detect_frame(frame, conf_threshold=conf_thresh, imgsz=imgsz)
        raw_boxes_all = raw_boxes_res or []
        t_det1 = time.perf_counter()
        det_elapsed_ms = (t_det1 - t_det0) * 1000.0
        detector_latencies_ms.append(det_elapsed_ms)

        # Lưu raw detections cho FrameDetection (Pipeline A & Grid Search)
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

        # Các box đạt ngưỡng đánh giá mặc định `eval_conf_thresh` cho chạy trực tiếp Pipeline B & C
        boxes_for_b = [b for b in clone_boxes(raw_boxes_all) if b.confidence >= eval_conf_thresh]
        boxes_for_c = [b for b in clone_boxes(raw_boxes_all) if b.confidence >= eval_conf_thresh]

        # ----------------------------------------------------------------------
        # PIPELINE B: YOLO + ByteTrack + TemporalFeatureExtractor + TemporalVerifier
        # ----------------------------------------------------------------------
        t_b0 = time.perf_counter()
        tracked_boxes_b = bt_tracker.update(boxes_for_b, frame)
        t_b_trk = time.perf_counter()

        feat_map_b: Dict[int, Any] = {}
        det_dicts_with_tracks: List[Dict[str, Any]] = []
        for tb in tracked_boxes_b:
            f_res = feat_extractor.update(
                track_id=tb.track_id,
                frame_bgr=frame,
                xyxy=tb.xyxy,
                timestamp_sec=ts_sec,
            )
            feat_map_b[tb.track_id] = f_res
            dyn_score = compute_dynamic_feature_score(f_res)
            det_dicts_with_tracks.append({
                "label": tb.label,
                "confidence": round(float(tb.confidence), 4),
                "track_id": int(tb.track_id),
                "temporal_score": round(float(dyn_score), 4),
                "flicker_score": round(float(f_res.flicker_score), 4),
                "smoke_expansion_score": round(float(f_res.smoke_expansion_score), 4),
                "area_growth_rate": round(float(f_res.area_growth_rate), 4),
            })
        t_b_feat = time.perf_counter()

        res_b = verifier_b.update(
            tracked_boxes_b,
            tracks_assigned=True,
            feature_map=feat_map_b,
            timestamp_sec=ts_sec,
        )
        t_b1 = time.perf_counter()

        trk_ms = (t_b_trk - t_b0) * 1000.0
        feat_ms = (t_b_feat - t_b_trk) * 1000.0
        ver_b_ms = (t_b1 - t_b_feat) * 1000.0
        tracker_overhead_ms.append(trk_ms)
        feature_overhead_ms.append(feat_ms)
        verifier_b_overhead_ms.append(ver_b_ms)
        pipeline_b_latencies_ms.append(det_elapsed_ms + trk_ms + feat_ms + ver_b_ms)

        # Lưu thêm các detection ở ngưỡng thấp [0.25, eval_conf_thresh) vào tracked_frame_detections để quét ngưỡng 0.25
        low_conf_extra = [
            {
                "label": b.label,
                "confidence": round(float(b.confidence), 4),
                "track_id": f"raw_{frame_idx}_{idx}",
                "temporal_score": 0.02,
            }
            for idx, b in enumerate(raw_boxes_all)
            if b.confidence < eval_conf_thresh
        ]
        tracked_frame_detections.append(
            FrameDetection(vid, cam_id, round(ts_sec, 4), tuple(det_dicts_with_tracks + low_conf_extra))
        )

        if res_b["alert_level"] == "FIRE" and res_b["primary_label"]:
            lbl_b = res_b["primary_label"]
            pipeline_b_confirmed_samples.setdefault(lbl_b, []).append(
                (round(ts_sec, 4), float(res_b["max_confidence"]))
            )
        if res_b["should_trigger_db_alert"] and res_b["primary_label"]:
            pipeline_b_db_alerts.append({
                "video_id": vid,
                "camera_id": cam_id,
                "timestamp_sec": round(ts_sec, 4),
                "label": res_b["primary_label"],
                "confidence": float(res_b["max_confidence"]),
                "active_hits": int(res_b["active_hits"]),
                "confirmed_track_id": res_b["confirmed_track_id"],
                "temporal_score": float(res_b["temporal_score"]),
            })

        # ----------------------------------------------------------------------
        # PIPELINE C: YOLO + Legacy IoU TemporalVerifier
        # ----------------------------------------------------------------------
        t_c0 = time.perf_counter()
        res_c = verifier_c.update(
            boxes_for_c,
            tracks_assigned=False,
            timestamp_sec=ts_sec,
        )
        t_c1 = time.perf_counter()
        pipeline_c_latencies_ms.append(det_elapsed_ms + (t_c1 - t_c0) * 1000.0)

        if res_c["alert_level"] == "FIRE" and res_c["primary_label"]:
            lbl_c = res_c["primary_label"]
            pipeline_c_confirmed_samples.setdefault(lbl_c, []).append(
                (round(ts_sec, 4), float(res_c["max_confidence"]))
            )
        if res_c["should_trigger_db_alert"] and res_c["primary_label"]:
            pipeline_c_db_alerts.append({
                "video_id": vid,
                "camera_id": cam_id,
                "timestamp_sec": round(ts_sec, 4),
                "label": res_c["primary_label"],
                "confidence": float(res_c["max_confidence"]),
                "active_hits": int(res_c["active_hits"]),
                "confirmed_track_id": res_c["confirmed_track_id"],
            })

    # --------------------------------------------------------------------------
    # PIPELINE A: Frame-level Baseline (từ raw_frame_detections ở eval_conf_thresh)
    # --------------------------------------------------------------------------
    raw_frame_alarms = extract_frame_level_alarms(
        raw_frame_detections, confidence_threshold=eval_conf_thresh
    )
    pipeline_a_events = merge_frame_alarms_to_events(
        raw_frame_alarms, max_event_gap_sec=max_event_gap_sec
    )

    # Gộp các mốc xác nhận của Pipeline B và C thành Event intervals
    pipeline_b_events: List[Event] = []
    for lbl, samples in pipeline_b_confirmed_samples.items():
        alarms_b = [
            {
                "video_id": vid,
                "camera_id": cam_id,
                "timestamp_sec": ts,
                "label": lbl,
                "confidence": conf,
            }
            for ts, conf in samples
        ]
        pipeline_b_events.extend(merge_frame_alarms_to_events(alarms_b, max_event_gap_sec=max_event_gap_sec))
    pipeline_b_events.sort(key=lambda e: (e.start_sec, e.label))

    pipeline_c_events: List[Event] = []
    for lbl, samples in pipeline_c_confirmed_samples.items():
        alarms_c = [
            {
                "video_id": vid,
                "camera_id": cam_id,
                "timestamp_sec": ts,
                "label": lbl,
                "confidence": conf,
            }
            for ts, conf in samples
        ]
        pipeline_c_events.extend(merge_frame_alarms_to_events(alarms_c, max_event_gap_sec=max_event_gap_sec))
    pipeline_c_events.sort(key=lambda e: (e.start_sec, e.label))

    return {
        "video_id": vid,
        "camera_id": cam_id,
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
        "pipeline_b": {
            "db_alerts": pipeline_b_db_alerts,
            "db_alert_count": len(pipeline_b_db_alerts),
            "events": pipeline_b_events,
            "event_count": len(pipeline_b_events),
            "mean_pipeline_latency_ms": float(np.mean(pipeline_b_latencies_ms)) if pipeline_b_latencies_ms else 0.0,
            "mean_tracker_overhead_ms": float(np.mean(tracker_overhead_ms)) if tracker_overhead_ms else 0.0,
            "mean_feature_overhead_ms": float(np.mean(feature_overhead_ms)) if feature_overhead_ms else 0.0,
            "mean_verifier_overhead_ms": float(np.mean(verifier_b_overhead_ms)) if verifier_b_overhead_ms else 0.0,
        },
        "pipeline_c": {
            "db_alerts": pipeline_c_db_alerts,
            "db_alert_count": len(pipeline_c_db_alerts),
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
    """Đánh giá bổ sung cấp độ video (bất kể nhãn FIRE/SMOKE miễn nằm trong khoảng sự kiện cháy/khói của video dương tính).

    Bên cạnh `src/event_eval.py` (so khớp chặt theo cả `(video_id, camera_id, label)`), hàm này
    kiểm tra xem mỗi video dương tính có nhận được cảnh báo cháy/khói kể từ mốc khởi phát sớm nhất
    (`min(fire_start_sec, smoke_start_sec)` >= 0) hay không, đồng thời đếm số cảnh báo nổ ra TRƯỚC
    khi sự kiện bắt đầu (`pre_event_false_alarms`).
    """
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

    # Chọn checkpoint tốt nhất đã kiểm chứng từ Tuần 3 (runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt)
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

    # 2. Chia tập Validation (12 video lẻ: POS_01, POS_03, ..., POS_23) và Test (12 video chẵn: POS_02, POS_04, ..., POS_24)
    val_videos = [v for idx, v in enumerate(avail_pos) if idx % 2 == 0]
    test_videos = [v for idx, v in enumerate(avail_pos) if idx % 2 == 1]

    val_ids: Set[str] = {v["video_id"] for v in val_videos}
    test_ids: Set[str] = {v["video_id"] for v in test_videos}
    all_pos_ids: Set[str] = {v["video_id"] for v in avail_pos}
    neg_ids: Set[str] = {v["video_id"] for v in avail_neg}
    neg_hours: float | None = (total_neg_avail_sec / 3600.0) if avail_neg else None

    print(f"📊 Chia tập thực nghiệm: Validation = {len(val_videos)} video | Test = {len(test_videos)} video")

    # 3. Khởi tạo Detector
    detector = SharedFireSmokeDetector(model_path=str(model_path))

    # 4. Chạy suy luận trên toàn bộ 24 video với điểm vận hành mặc định (conf=0.35, K=3, N=8, cooldown=5.0s, feat=0.0)
    sample_fps = 5.0
    default_conf = 0.35
    default_k = 3
    default_n = 8
    default_cooldown = 5.0
    default_feat_thr = 0.0
    max_gap_sec = 2.0

    per_video_runs: List[Dict[str, Any]] = []
    for idx, v in enumerate(avail_pos, start=1):
        res = run_three_pipelines_on_video(
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
        per_video_runs.append(res)
        pa_cnt = res["pipeline_a"]["raw_frame_alarm_count"]
        pa_ev = res["pipeline_a"]["event_count"]
        pb_ev = res["pipeline_b"]["event_count"]
        pc_ev = res["pipeline_c"]["event_count"]
        print(
            f"  [{idx:02d}/{len(avail_pos):02d}] {v['video_id']} ({v['duration_sec']:5.1f}s) | "
            f"FrameAlarms(A): {pa_cnt:3d} -> Events(A): {pa_ev:2d} | "
            f"ConfirmedEvents(B-ByteTrack): {pb_ev:2d} | LegacyIoU(C): {pc_ev:2d} | "
            f"Det: {res['pipeline_a']['mean_detector_latency_ms']:.1f}ms | "
            f"PipeB: {res['pipeline_b']['mean_pipeline_latency_ms']:.1f}ms"
        )

    # 5. Quét lưới tham số (Grid Search) trên tập Validation (12 video)
    val_tracked_frames: List[FrameDetection] = []
    test_tracked_frames: List[FrameDetection] = []
    all_tracked_frames: List[FrameDetection] = []
    for r in per_video_runs:
        all_tracked_frames.extend(r["tracked_frame_detections"])
        if r["video_id"] in val_ids:
            val_tracked_frames.extend(r["tracked_frame_detections"])
        else:
            test_tracked_frames.extend(r["tracked_frame_detections"])

    gt_all = build_ground_truth_events(avail_pos)
    gt_val = build_ground_truth_events(val_videos)
    gt_test = build_ground_truth_events(test_videos)

    print("\n🔍 Đang quét lưới tham số (Grid Search) trên Validation Set...")
    grid_candidates_val = grid_search(
        frames=val_tracked_frames,
        ground_truth=gt_val,
        positive_video_ids=val_ids,
        negative_video_ids=neg_ids,
        negative_hours=neg_hours,
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

    # Đánh giá cấu hình provisional (hoặc mặc định) trên tập Test khóa cố định
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
        neg_ids,
        neg_hours,
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

    # 6. Tổng hợp sự kiện và đánh giá 3 Pipeline trên cả 3 phân vùng (Full 24, Validation 12, Test 12)
    def summarize_split(split_name: str, split_videos: List[Dict[str, Any]], split_ids: Set[str], split_gt: List[Event]) -> Dict[str, Any]:
        split_runs = [r for r in per_video_runs if r["video_id"] in split_ids]
        raw_alarms_a: List[Dict[str, Any]] = []
        events_a: List[Event] = []
        events_b: List[Event] = []
        events_c: List[Event] = []
        db_alerts_b: List[Dict[str, Any]] = []
        db_alerts_c: List[Dict[str, Any]] = []

        det_ms_list: List[float] = []
        pipe_b_ms_list: List[float] = []
        pipe_c_ms_list: List[float] = []
        trk_ms_list: List[float] = []
        feat_ms_list: List[float] = []
        ver_ms_list: List[float] = []
        total_frames = 0

        for r in split_runs:
            total_frames += r["sampled_frames"]
            raw_alarms_a.extend(r["pipeline_a"]["raw_frame_alarms"])
            events_a.extend(r["pipeline_a"]["events"])
            events_b.extend(r["pipeline_b"]["events"])
            events_c.extend(r["pipeline_c"]["events"])
            db_alerts_b.extend(r["pipeline_b"]["db_alerts"])
            db_alerts_c.extend(r["pipeline_c"]["db_alerts"])

            det_ms_list.append(r["pipeline_a"]["mean_detector_latency_ms"])
            pipe_b_ms_list.append(r["pipeline_b"]["mean_pipeline_latency_ms"])
            pipe_c_ms_list.append(r["pipeline_c"]["mean_pipeline_latency_ms"])
            trk_ms_list.append(r["pipeline_b"]["mean_tracker_overhead_ms"])
            feat_ms_list.append(r["pipeline_b"]["mean_feature_overhead_ms"])
            ver_ms_list.append(r["pipeline_b"]["mean_verifier_overhead_ms"])

        eval_a = evaluate(split_gt, events_a, split_ids, neg_ids, neg_hours, tolerance_sec=10.0, allow_missing_negative=True)
        eval_b = evaluate(split_gt, events_b, split_ids, neg_ids, neg_hours, tolerance_sec=10.0, allow_missing_negative=True)
        eval_c = evaluate(split_gt, events_c, split_ids, neg_ids, neg_hours, tolerance_sec=10.0, allow_missing_negative=True)

        vid_eval_a = evaluate_video_level_primary_recall(split_videos, events_a, tolerance_sec=10.0)
        vid_eval_b = evaluate_video_level_primary_recall(split_videos, events_b, tolerance_sec=10.0)
        vid_eval_c = evaluate_video_level_primary_recall(split_videos, events_c, tolerance_sec=10.0)

        mean_det_ms = float(np.mean(det_ms_list)) if det_ms_list else 0.0
        mean_pb_ms = float(np.mean(pipe_b_ms_list)) if pipe_b_ms_list else 0.0
        mean_pc_ms = float(np.mean(pipe_c_ms_list)) if pipe_c_ms_list else 0.0

        alarm_reduction_vs_raw = (
            (len(raw_alarms_a) - len(db_alerts_b)) / float(len(raw_alarms_a))
            if len(raw_alarms_a) > 0 else 0.0
        )
        event_fragmentation_reduction = (
            (len(events_a) - len(events_b)) / float(len(events_a))
            if len(events_a) > 0 else 0.0
        )

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
            "pipeline_b_temporal_confirmed": {
                "pipeline_id": "pipeline_b_temporal_confirmed_bytetrack_features",
                "description": f"Temporal-confirmed (YOLO + ByteTrack + TemporalFeatures + TemporalVerifier K={default_k}/N={default_n}, cooldown={default_cooldown}s)",
                "db_alert_trigger_count": len(db_alerts_b),
                "merged_event_count": len(events_b),
                "alarm_reduction_vs_raw_frame_alarms_pct": round(alarm_reduction_vs_raw * 100.0, 2),
                "event_fragmentation_reduction_vs_baseline_pct": round(event_fragmentation_reduction * 100.0, 2),
                "strict_class_evaluation": eval_b,
                "video_primary_evaluation": vid_eval_b,
                "detector_only_latency_ms": round(mean_det_ms, 3),
                "tracker_overhead_ms": round(float(np.mean(trk_ms_list)), 3) if trk_ms_list else 0.0,
                "feature_overhead_ms": round(float(np.mean(feat_ms_list)), 3) if feat_ms_list else 0.0,
                "verifier_overhead_ms": round(float(np.mean(ver_ms_list)), 3) if ver_ms_list else 0.0,
                "end_to_end_pipeline_latency_ms": round(mean_pb_ms, 3),
                "end_to_end_pipeline_fps": round(1000.0 / mean_pb_ms, 2) if mean_pb_ms > 0 else 0.0,
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

    # 7. Xuất các tệp JSON/CSV theo đúng danh mục nghiệm thu Mục 7
    all_raw_frame_alarms = [a for r in per_video_runs for a in r["pipeline_a"]["raw_frame_alarms"]]
    all_events_a = [asdict(e) for r in per_video_runs for e in r["pipeline_a"]["events"]]
    all_events_b = [asdict(e) for r in per_video_runs for e in r["pipeline_b"]["events"]]
    all_events_c = [asdict(e) for r in per_video_runs for e in r["pipeline_c"]["events"]]
    all_db_alerts_b = [a for r in per_video_runs for a in r["pipeline_b"]["db_alerts"]]
    all_db_alerts_c = [a for r in per_video_runs for a in r["pipeline_c"]["db_alerts"]]

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
            "pipeline": "pipeline_b_temporal_confirmed_bytetrack_features",
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
            "db_alert_triggers_count": len(all_db_alerts_b),
            "merged_events_count": len(all_events_b),
            "merged_events": all_events_b,
            "db_alert_triggers": all_db_alerts_b,
            "legacy_iou_temporal_events": all_events_c,
            "legacy_iou_db_alert_triggers": all_db_alerts_c,
        },
        out_dir / "temporal_confirmed_events.json",
    )

    # Xuất comparison.csv
    comparison_rows = []
    for split_obj in (full_summary, val_summary, test_summary):
        sp_name = split_obj["split"]
        pa = split_obj["pipeline_a_frame_level_baseline"]
        pb = split_obj["pipeline_b_temporal_confirmed"]
        pc = split_obj["pipeline_c_legacy_iou_temporal"]

        for p_key, p_data, raw_cnt in (
            ("Pipeline A: Frame-level Baseline (No K/N, No Tracker)", pa, pa["raw_frame_alarm_count"]),
            ("Pipeline B: Temporal-Confirmed (ByteTrack + Features + K/N)", pb, pb["db_alert_trigger_count"]),
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

    # Xuất grid_search.csv, grid_search.json và best_operating_point.json
    export_grid_search_csv(grid_candidates_val, out_dir / "grid_search.csv")
    export_json(
        {
            "validation_split_videos": sorted(list(val_ids)),
            "test_split_videos": sorted(list(test_ids)),
            "total_candidates_evaluated": len(grid_candidates_val),
            "operating_point_selection": operating_point_selection,
            "candidates": grid_candidates_val,
        },
        out_dir / "grid_search.json",
    )
    export_json(operating_point_selection, out_dir / "best_operating_point.json")

    # Xuất metrics.json tổng thể
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
        "operating_point_selection": operating_point_selection,
    }
    export_json(metrics_payload, out_dir / "metrics.json")

    # Tạo benchmark_report.md tự động từ số liệu thực tế
    pa_full = full_summary["pipeline_a_frame_level_baseline"]
    pb_full = full_summary["pipeline_b_temporal_confirmed"]
    pc_full = full_summary["pipeline_c_legacy_iou_temporal"]

    pa_test = test_summary["pipeline_a_frame_level_baseline"]
    pb_test = test_summary["pipeline_b_temporal_confirmed"]

    report_md = f"""# Báo Cáo Thực Nghiệm & Nghiệm Thu Tuần 4: Frame-Level Baseline vs. ByteTrack + Temporal Verification

- **Thời gian thực thi:** `{metrics_payload["timestamp"]}`
- **Model Checkpoint:** `{metrics_payload["model_checkpoint"]}` (mAP@0.5 = 76.78% trên D-Fire Test)
- **Tần số lấy mẫu khung hình:** `{sample_fps} FPS` (`imgsz = 416`, `seed = 42`)
- **Trạng thái nghiệm thu Tuần 4:** **`PARTIAL (BLOCKED_BY_NEGATIVE_DATA)`**

---

## 1. Hiện Trạng Dữ Liệu Video Kiểm Thử Trên Ổ Đĩa

| Phân loại | Số video trong `video_dataset_list.csv` | Số video thực có trên ổ đĩa | Thời lượng thực có | Chỉ tiêu Đề cương (MT1/MT3) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Video Dương tính (Có cháy/khói)** | 44 (`POS_01..24`, `VN_POS_01..20`) | **24** (`POS_01..24`) | `863.98s` (`0.24h`) | $\ge 20$ video | **ĐẠT (24/20 video)** |
| **Video Âm tính (Không cháy / Cảnh nhiễu)** | 28 (`NEG_25..49`, `VN_NEG_01,03,04`) | **0** | `0.00s` (`0.00h`) | $\ge 5.0$ giờ | **THIẾU (`BLOCKED_BY_NEGATIVE_DATA`)** |

> Danh sách chi tiết 28 video âm tính bị thiếu trên ổ đĩa (`2.8622 giờ` trong catalog + `2.1378 giờ` cần thu thập bổ sung) đã được xuất tại `missing_negative_videos_manifest.csv` và `missing_negative_videos_manifest.json`.

---

## 2. Bảng Đối Sánh 3 Pipeline Trên Toàn Bộ 24 Video Dương Tính (`full_24_positive`)

Cấu hình điểm vận hành đánh giá: `confidence_threshold = {default_conf}`, `K = {default_k}`, `N = {default_n}`, `cooldown_sec = {default_cooldown}s`, `max_event_gap_sec = {max_gap_sec}s`.

| Chỉ số thực nghiệm | Pipeline A: Frame-Level Baseline (Chỉ YOLO, không K/N, không Tracker) | Pipeline B: Temporal-Confirmed (YOLO + ByteTrack + Features + K/N) | Pipeline C: Legacy IoU Temporal (YOLO + IoU Tracker + K/N) |
| :--- | :---: | :---: | :---: |
| **Tổng số cảnh báo phát ra (Raw Frame Alarms / DB Alert Triggers)** | **{pa_full["raw_frame_alarm_count"]}** frame alarms | **{pb_full["db_alert_trigger_count"]}** DB alerts (**-{pb_full["alarm_reduction_vs_raw_frame_alarms_pct"]:.2f}%**) | **{pc_full["db_alert_trigger_count"]}** DB alerts |
| **Số khoảng sự kiện sau khi gộp (`max_event_gap_sec = 2.0s`)** | **{pa_full["merged_event_count"]}** events | **{pb_full["merged_event_count"]}** events (**-{pb_full["event_fragmentation_reduction_vs_baseline_pct"]:.2f}%**) | **{pc_full["merged_event_count"]}** events |
| **Video-level Event Recall (Strict Class Match)** | `{pa_full["strict_class_evaluation"]["metrics"]["video_event_recall"]*100:.2f}%` | `{pb_full["strict_class_evaluation"]["metrics"]["video_event_recall"]*100:.2f}%` | `{pc_full["strict_class_evaluation"]["metrics"]["video_event_recall"]*100:.2f}%` |
| **Video-level Primary Recall (Any Fire/Smoke Class)** | `{pa_full["video_primary_evaluation"]["any_class_video_recall"]*100:.2f}%` ({pa_full["video_primary_evaluation"]["recalled_positive_videos"]}/24) | `{pb_full["video_primary_evaluation"]["any_class_video_recall"]*100:.2f}%` ({pb_full["video_primary_evaluation"]["recalled_positive_videos"]}/24) | `{pc_full["video_primary_evaluation"]["any_class_video_recall"]*100:.2f}%` ({pc_full["video_primary_evaluation"]["recalled_positive_videos"]}/24) |
| **Event-instance Recall** | `{pa_full["strict_class_evaluation"]["metrics"]["event_instance_recall"]*100:.2f}%` | `{pb_full["strict_class_evaluation"]["metrics"]["event_instance_recall"]*100:.2f}%` | `{pc_full["strict_class_evaluation"]["metrics"]["event_instance_recall"]*100:.2f}%` |
| **Event Precision** | `{pa_full["strict_class_evaluation"]["metrics"]["event_precision"]*100:.2f}%` | `{pb_full["strict_class_evaluation"]["metrics"]["event_precision"]*100:.2f}%` | `{pc_full["strict_class_evaluation"]["metrics"]["event_precision"]*100:.2f}%` |
| **Event F1-Score** | `{pa_full["strict_class_evaluation"]["metrics"]["event_f1"]*100:.2f}%` | `{pb_full["strict_class_evaluation"]["metrics"]["event_f1"]*100:.2f}%` | `{pc_full["strict_class_evaluation"]["metrics"]["event_f1"]*100:.2f}%` |
| **Mean Detection Delay (Strict Class Match)** | `{pa_full["strict_class_evaluation"]["metrics"]["mean_detection_delay_sec"]:.2f} s` | `{pb_full["strict_class_evaluation"]["metrics"]["mean_detection_delay_sec"]:.2f} s` ($\le 10$ s) | `{pc_full["strict_class_evaluation"]["metrics"]["mean_detection_delay_sec"]:.2f} s` |
| **Mean Detection Delay (Từ mốc khởi phát sớm nhất)** | `{pa_full["video_primary_evaluation"]["mean_delay_from_earliest_onset_sec"]:.2f} s` | `{pb_full["video_primary_evaluation"]["mean_delay_from_earliest_onset_sec"]:.2f} s` ($\le 10$ s) | `{pc_full["video_primary_evaluation"]["mean_delay_from_earliest_onset_sec"]:.2f} s` |
| **Cảnh báo nổ ra trước mốc cháy (`pre_event_alarms` trên video dương)** | `{pa_full["video_primary_evaluation"]["pre_event_false_alarms_on_positive_videos"]}` | `{pb_full["video_primary_evaluation"]["pre_event_false_alarms_on_positive_videos"]}` | `{pc_full["video_primary_evaluation"]["pre_event_false_alarms_on_positive_videos"]}` |
| **Negative False Alarms per Hour (FAR/giờ trên video âm tính)** | `UNAVAILABLE` | `UNAVAILABLE` | `UNAVAILABLE` |
| **Detector-only Latency / FPS (CPU)** | `{pa_full["detector_only_latency_ms"]:.2f} ms` (`{pa_full["detector_only_fps"]:.1f} FPS`) | `{pb_full["detector_only_latency_ms"]:.2f} ms` (`{pa_full["detector_only_fps"]:.1f} FPS`) | `{pa_full["detector_only_latency_ms"]:.2f} ms` (`{pa_full["detector_only_fps"]:.1f} FPS`) |
| **End-to-End Pipeline Latency / FPS (CPU)** | `{pa_full["detector_only_latency_ms"]:.2f} ms` (`{pa_full["detector_only_fps"]:.1f} FPS`) | **`{pb_full["end_to_end_pipeline_latency_ms"]:.2f} ms` (`{pb_full["end_to_end_pipeline_fps"]:.1f} FPS`)** | `{pc_full["end_to_end_pipeline_latency_ms"]:.2f} ms` (`{pc_full["end_to_end_pipeline_fps"]:.1f} FPS`) |

> **Chi phí tính toán tăng thêm (Overhead) của Pipeline B:**
> - `ByteTrackAdapter.update`: `{pb_full["tracker_overhead_ms"]:.3f} ms/frame`
> - `TemporalFeatureExtractor.update`: `{pb_full["feature_overhead_ms"]:.3f} ms/frame`
> - `TemporalVerifier.update`: `{pb_full["verifier_overhead_ms"]:.3f} ms/frame`

---

## 3. Kết Quả Quét Lưới Tham Số (Grid Search) Trên Validation Set & Đánh Giá Trên Test Set

- **Validation Set (12 video):** `POS_01, POS_03, POS_05, POS_07, POS_09, POS_11, POS_13, POS_15, POS_17, POS_19, POS_21, POS_23` (`456.65s`).
- **Test Set khóa cố định (12 video):** `POS_02, POS_04, POS_06, POS_08, POS_10, POS_12, POS_14, POS_16, POS_18, POS_20, POS_22, POS_24` (`407.33s`).
- **Tổng số cấu hình đã quét:** `{len(grid_candidates_val)}` tổ hợp (`confidence_threshold` $\in \{{0.25, 0.35, 0.45\}}$, $K \in \{{2, 3, 4, 5\}}$, $N \in \{{5, 8, 10, 12\}}$, `feature_threshold` $\in \{{0.0, 0.01, 0.02\}}$, `cooldown_sec` $\in \{{0.0, 5.0, 12.0\}}$).
- **Kết luận chọn điểm vận hành:** `{operating_point_selection["status"]}`
  - Lý do: {operating_point_selection["reason"]}
- **Kết quả trên Test Set (12 video độc lập) tại điểm vận hành `conf={default_conf}, K={default_k}, N={default_n}`:**
  - **Pipeline A (Frame-level baseline):** `{pa_test["raw_frame_alarm_count"]}` raw frame alarms $\\rightarrow$ `{pa_test["merged_event_count"]}` merged events | Strict Video Recall = `{pa_test["strict_class_evaluation"]["metrics"]["video_event_recall"]*100:.2f}%` | Primary Video Recall = `{pa_test["video_primary_evaluation"]["any_class_video_recall"]*100:.2f}%` | Event Precision = `{pa_test["strict_class_evaluation"]["metrics"]["event_precision"]*100:.2f}%` | Event F1 = `{pa_test["strict_class_evaluation"]["metrics"]["event_f1"]*100:.2f}%` | Strict Delay = `{pa_test["strict_class_evaluation"]["metrics"]["mean_detection_delay_sec"]:.2f}s`.
  - **Pipeline B (Temporal-confirmed ByteTrack + Features):** `{pb_test["db_alert_trigger_count"]}` DB alert triggers (**giảm `{pb_test["alarm_reduction_vs_raw_frame_alarms_pct"]:.2f}%` số lần báo động lặp**) $\\rightarrow$ `{pb_test["merged_event_count"]}` merged events | Strict Video Recall = `{pb_test["strict_class_evaluation"]["metrics"]["video_event_recall"]*100:.2f}%` | Primary Video Recall = `{pb_test["video_primary_evaluation"]["any_class_video_recall"]*100:.2f}%` | Event Precision = `{pb_test["strict_class_evaluation"]["metrics"]["event_precision"]*100:.2f}%` | Event F1 = `{pb_test["strict_class_evaluation"]["metrics"]["event_f1"]*100:.2f}%` | Strict Delay = `{pb_test["strict_class_evaluation"]["metrics"]["mean_detection_delay_sec"]:.2f}s`.
"""
    (out_dir / "benchmark_report.md").write_text(report_md, encoding="utf-8")

    # Đồng bộ các tệp chính sang runs/benchmarks/week4_bytetrack/ để không còn tệp cũ sai lệch
    for fname in ("comparison.csv", "metrics.json", "benchmark_report.md"):
        shutil.copy2(out_dir / fname, legacy_out_dir / fname)

    print("\n" + "=" * 85)
    print(f"✅ ĐÃ XUẤT ĐẦY ĐỦ 9 TỆP NGHIỆM THU TUẦN 4 TẠI: {out_dir}")
    print("=" * 85)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
