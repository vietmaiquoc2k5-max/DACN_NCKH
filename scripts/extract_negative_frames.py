import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bytetrack_adapter import ByteTrackAdapter
from src.config import ByteTrackSettings
from src.detector import DetectionBox, SharedFireSmokeDetector
from src.temporal_features import TemporalFeatureExtractor
from src.temporal_verifier import compute_dynamic_feature_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("extract_negative_frames")


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


def run_negative_extraction_streaming(sample_interval_sec: float = 2.0):
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    manifest_path = PROJECT_ROOT / "data" / "negative_videos_manifest.json"
    if not manifest_path.exists():
        LOGGER.error("Manifest không tồn tại: %s", manifest_path)
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    videos = manifest.get("videos", [])
    LOGGER.info("Bắt đầu trích xuất trực tuyến (streaming) cho %d video âm tính.", len(videos))

    model_path = PROJECT_ROOT / "runs" / "detect" / "runs" / "benchmarks" / "yolov8n_seed42" / "weights" / "best.pt"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "best.pt"
    LOGGER.info("Sử dụng model: %s", model_path)

    detector = SharedFireSmokeDetector(model_path=str(model_path), conf_threshold=0.25)

    out_dir = PROJECT_ROOT / "runs" / "benchmarks" / "week4_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    neg_raw_path = out_dir / "negative_raw_detections.jsonl"
    neg_tracked_path = out_dir / "negative_tracked_detections.jsonl"

    total_sampled = 0
    total_raw_dets = 0
    total_tracked_dets = 0

    t_start = time.time()

    with open(neg_raw_path, "w", encoding="utf-8") as raw_f, open(neg_tracked_path, "w", encoding="utf-8") as trk_f:
        for v in videos:
            vid = v["video_id"]
            cam_id = f"BENCH_{vid}"
            rel_p = Path(v["relative_path"])
            full_p = PROJECT_ROOT / rel_p if not rel_p.is_absolute() else rel_p

            if not full_p.exists():
                LOGGER.error("File không tồn tại: %s", full_p)
                continue

            cap = cv2.VideoCapture(str(full_p))
            if not cap.isOpened():
                LOGGER.error("Không thể mở video bằng OpenCV: %s", full_p)
                continue

            native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            dur_sec = total_frames / native_fps if native_fps > 0 else 0

            # Đối với video ngắn (<= 60s), lấy mẫu 0.5s/frame = 2 FPS
            # Đối với video dài (> 60s), lấy mẫu theo sample_interval_sec (2.0s = 0.5 FPS)
            target_interval = 0.5 if dur_sec <= 60.0 else sample_interval_sec
            step = max(1, int(round(native_fps * target_interval)))

            LOGGER.info(
                "Đang xử lý %s (%s, %.1fs, %d frames) | step=%d (~%.1fs/frame)...",
                vid, v["file_name"], dur_sec, total_frames, step, target_interval
            )

            bt_tracker = ByteTrackAdapter(
                settings=ByteTrackSettings(
                    track_high_thresh=0.25,
                    track_low_thresh=0.10,
                    new_track_thresh=0.25,
                    track_buffer=15,
                ),
                per_class=True,
            )
            feat_extractor = TemporalFeatureExtractor(history_size=12)

            frame_idx = 0
            v_sampled = 0
            t_vid_start = time.time()

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % step == 0:
                    ts_sec = round(frame_idx / native_fps, 4)
                    v_sampled += 1
                    total_sampled += 1

                    raw_boxes_res, _ = detector.detect_frame(frame, conf_threshold=0.25, imgsz=416, non_blocking=False)
                    raw_boxes = raw_boxes_res or []

                    raw_det_list: List[Dict[str, Any]] = []
                    for rb in raw_boxes:
                        raw_det_list.append({
                            "label": rb.label,
                            "confidence": round(float(rb.confidence), 4),
                            "track_id": None,
                            "xyxy": [int(coord) for coord in rb.xyxy],
                        })
                    total_raw_dets += len(raw_det_list)

                    raw_record = {
                        "video_id": vid,
                        "camera_id": cam_id,
                        "timestamp_sec": float(ts_sec),
                        "detections": raw_det_list,
                    }
                    raw_f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")

                    # Track qua ByteTrack
                    tracked_boxes = bt_tracker.update(clone_boxes(raw_boxes), frame)
                    tracked_det_list: List[Dict[str, Any]] = []

                    for tb in tracked_boxes:
                        if tb.track_id <= 0:
                            continue
                        try:
                            f_res = feat_extractor.update(
                                track_id=tb.track_id,
                                frame_bgr=frame,
                                xyxy=tb.xyxy,
                                timestamp_sec=ts_sec,
                            )
                            dyn_score = compute_dynamic_feature_score(f_res)
                        except Exception as err:
                            dyn_score = 0.0
                            f_res = None

                        tracked_det_list.append({
                            "label": tb.label,
                            "confidence": round(float(tb.confidence), 4),
                            "track_id": int(tb.track_id),
                            "temporal_score": round(float(dyn_score), 4),
                            "flicker_score": round(float(f_res.flicker_score), 4) if f_res else 0.0,
                            "smoke_expansion_score": round(float(f_res.smoke_expansion_score), 4) if f_res else 0.0,
                            "area_growth_rate": round(float(f_res.area_growth_rate), 4) if f_res else 0.0,
                            "xyxy": [int(coord) for coord in tb.xyxy],
                        })
                    total_tracked_dets += len(tracked_det_list)

                    tracked_record = {
                        "video_id": vid,
                        "camera_id": cam_id,
                        "timestamp_sec": float(ts_sec),
                        "detections": tracked_det_list,
                    }
                    trk_f.write(json.dumps(tracked_record, ensure_ascii=False) + "\n")

                    if v_sampled % 250 == 0:
                        raw_f.flush()
                        trk_f.flush()
                        v_fps = v_sampled / (time.time() - t_vid_start)
                        LOGGER.info("   [Progress %s] %d frames (ts=%.1fs) | %.1f samples/sec", vid, v_sampled, ts_sec, v_fps)

                frame_idx += 1

            cap.release()
            raw_f.flush()
            trk_f.flush()
            LOGGER.info("   -> Xong %s: %d frames lấy mẫu.", vid, v_sampled)

    elapsed = time.time() - t_start
    LOGGER.info(
        "Hoàn tất trích xuất negative frames: %d frames trong %.1fs (%.1f FPS)",
        total_sampled,
        elapsed,
        total_sampled / elapsed if elapsed > 0 else 0,
    )
    LOGGER.info("  Raw detections: %d, Tracked detections: %d", total_raw_dets, total_tracked_dets)
    LOGGER.info("  Lưu tại: %s và %s", neg_raw_path, neg_tracked_path)


if __name__ == "__main__":
    run_negative_extraction_streaming(sample_interval_sec=2.0)
