"""Script xuất tập dữ liệu phát hiện cấp khung hình Frame-level JSONL (`scripts/export_week4_frame_jsonl.py`).

Xuất 2 tệp JSONL phục vụ đánh giá và thẩm định độc lập:
1. `frame_level_detections.jsonl`: Kết quả phát hiện thô từ YOLOv8n (`conf_thresh >= 0.25`),
   không gán track_id giả lập (`track_id: null`).
2. `tracked_frame_detections.jsonl`: Kết quả đã qua liên kết ByteTrack chính thức 2 giai đoạn,
   giữ nguyên track_id thực tế và điểm biến động đặc trưng `temporal_score` trích xuất thực nghiệm.

Quy tắc bắt buộc:
- Tái sử dụng chính xác pipeline suy luận hiện tại (`SharedFireSmokeDetector`, `ByteTrackAdapter`, `TemporalFeatureExtractor`).
- Giữ đầy đủ tất cả khung hình được lấy mẫu (kể cả khung hình không có detection: `detections: []`).
- Timestamp tính theo thời gian thực (giây) tăng đơn điệu.
- Tần số lấy mẫu: 5.0 FPS, ngưỡng confidence khởi tạo detector: 0.25.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
LOGGER = logging.getLogger("export_week4_frame_jsonl")


def set_deterministic_seed(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên nhằm đảm bảo tính tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_catalog(csv_path: Path) -> List[Dict[str, Any]]:
    """Đọc và xác minh danh mục video từ data/video_dataset_list.csv."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp danh mục: {csv_path}")

    videos: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            vid = row["video_id"].strip()
            file_name = row.get("file_name", "").strip()
            rel_path = row.get("relative_path", row.get("file_path", "")).strip()
            cat = row.get("category", "").strip().lower()

            primary_path = PROJECT_ROOT / rel_path
            fallback_path = PROJECT_ROOT / "FireSmokeLabeler" / "videos" / file_name

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
            videos.append({
                "video_id": vid,
                "camera_id": f"BENCH_{vid}",
                "file_name": file_name,
                "path": full_path,
                "exists_on_disk": exists,
                "is_positive": is_pos,
                "category": cat,
                "duration_sec": float(row.get("duration_sec", 0.0)),
                "fps": float(row.get("fps", 25.0)),
            })
    return videos


def sample_frames(video_path: Path, sample_fps: float = 5.0) -> List[Tuple[int, float, np.ndarray]]:
    """Lấy mẫu các khung hình từ video theo tần số sample_fps cố định."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.error("Không thể mở video bằng OpenCV: %s", video_path)
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(native_fps / sample_fps)))

    sampled: List[Tuple[int, float, np.ndarray]] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            ts_sec = frame_idx / native_fps
            sampled.append((frame_idx, round(ts_sec, 4), frame))
        frame_idx += 1
    cap.release()
    return sampled


def clone_boxes(boxes: List[DetectionBox]) -> List[DetectionBox]:
    """Tạo bản sao sâu của danh sách DetectionBox."""
    return [
        DetectionBox(
            cls_id=b.cls_id,
            label=b.label,
            confidence=float(b.confidence),
            xyxy=tuple(b.xyxy),  # type: ignore[arg-type]
            norm_xywh=tuple(b.norm_xywh),  # type: ignore[arg-type]
            track_id=int(b.track_id),
        )
        for b in boxes
    ]


def export_frame_jsonl(
    catalog_path: Path,
    model_path: Path,
    output_dir: Path,
    sample_fps: float = 5.0,
    conf_threshold: float = 0.25,
    imgsz: int = 416,
) -> Tuple[Path, Path]:
    """Thực thi suy luận và xuất hai tệp JSONL cấp khung hình.

    Args:
        catalog_path: Đường dẫn tệp video_dataset_list.csv.
        model_path: Đường dẫn checkpoint YOLO (best.pt).
        output_dir: Thư mục xuất dữ liệu.
        sample_fps: Tần số lấy mẫu khung hình (mặc định 5.0 FPS).
        conf_threshold: Ngưỡng confidence suy luận (mặc định 0.25).
        imgsz: Kích thước ảnh nạp vào mô hình (mặc định 416).

    Returns:
        Tuple gồm (đường_dẫn_raw_jsonl, đường_dẫn_tracked_jsonl).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl_path = output_dir / "frame_level_detections.jsonl"
    tracked_jsonl_path = output_dir / "tracked_frame_detections.jsonl"

    videos = load_catalog(catalog_path)
    available_videos = [v for v in videos if v["exists_on_disk"]]
    LOGGER.info("Tìm thấy %d/%d video khả dụng trên ổ đĩa.", len(available_videos), len(videos))

    LOGGER.info("Đang khởi tạo detector từ: %s (conf=%.2f, device=%s)", model_path, conf_threshold, "GPU" if torch.cuda.is_available() else "CPU")
    detector = SharedFireSmokeDetector(model_path=str(model_path), conf_threshold=conf_threshold)

    total_sampled_frames = 0
    total_raw_detections = 0
    total_tracked_detections = 0
    total_empty_frames = 0

    with raw_jsonl_path.open("w", encoding="utf-8") as raw_f, tracked_jsonl_path.open("w", encoding="utf-8") as trk_f:
        for idx, v in enumerate(available_videos, 1):
            vid = v["video_id"]
            cam_id = v["camera_id"]
            v_path = v["path"]
            LOGGER.info("[%02d/%02d] Đang xử lý video %s (%s)...", idx, len(available_videos), vid, v_path.name)

            frames = sample_frames(v_path, sample_fps=sample_fps)
            if not frames:
                LOGGER.warning("Video %s không trích xuất được frame nào!", vid)
                continue

            bt_tracker = ByteTrackAdapter(
                settings=ByteTrackSettings(
                    track_high_thresh=conf_threshold,
                    track_low_thresh=0.10,
                    new_track_thresh=conf_threshold,
                    track_buffer=15,
                ),
                per_class=True,
            )
            feat_extractor = TemporalFeatureExtractor(history_size=12)

            for f_idx, ts_sec, frame in frames:
                total_sampled_frames += 1
                raw_boxes_res, _ = detector.detect_frame(frame, conf_threshold=conf_threshold, imgsz=imgsz, non_blocking=False)
                raw_boxes = raw_boxes_res or []

                # 1. Định dạng bản ghi raw detection (không gán track_id giả lập)
                raw_det_list: List[Dict[str, Any]] = []
                for rb in raw_boxes:
                    raw_det_list.append({
                        "label": rb.label,
                        "confidence": round(float(rb.confidence), 4),
                        "track_id": None,
                        "xyxy": [int(coord) for coord in rb.xyxy],
                    })
                total_raw_detections += len(raw_det_list)

                raw_record = {
                    "video_id": vid,
                    "camera_id": cam_id,
                    "timestamp_sec": float(ts_sec),
                    "detections": raw_det_list,
                }
                raw_f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")

                # 2. Định dạng bản ghi tracked detection (ByteTrack + Temporal Features)
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
                        LOGGER.warning("Lỗi trích xuất feature track %d @ %.2fs video %s: %s", tb.track_id, ts_sec, vid, err)
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
                total_tracked_detections += len(tracked_det_list)
                if len(tracked_det_list) == 0:
                    total_empty_frames += 1

                tracked_record = {
                    "video_id": vid,
                    "camera_id": cam_id,
                    "timestamp_sec": float(ts_sec),
                    "detections": tracked_det_list,
                }
                trk_f.write(json.dumps(tracked_record, ensure_ascii=False) + "\n")

    LOGGER.info("Hoàn tất xuất dữ liệu khung hình:")
    LOGGER.info("  - Tổng số frame được lấy mẫu: %d", total_sampled_frames)
    LOGGER.info("  - Số frame rỗng (không có vật thể): %d (%.2f%%)", total_empty_frames, (total_empty_frames / total_sampled_frames * 100) if total_sampled_frames > 0 else 0.0)
    LOGGER.info("  - Tổng phát hiện thô (Raw Detections): %d", total_raw_detections)
    LOGGER.info("  - Tổng phát hiện có theo dõi (Tracked Detections): %d", total_tracked_detections)
    LOGGER.info("  - Raw JSONL: %s", raw_jsonl_path)
    LOGGER.info("  - Tracked JSONL: %s", tracked_jsonl_path)

    return raw_jsonl_path, tracked_jsonl_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Xuất tập dữ liệu phát hiện cấp khung hình Frame-level JSONL Tuần 4.")
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data" / "video_dataset_list.csv", help="Đường dẫn file video_dataset_list.csv")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "best.pt", help="Đường dẫn YOLO model checkpoint")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "runs" / "benchmarks" / "week4_final", help="Thư mục xuất JSONL")
    parser.add_argument("--fps", type=float, default=5.0, help="Tần số lấy mẫu khung hình")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng confidence suy luận")
    parser.add_argument("--imgsz", type=int, default=416, help="Kích thước ảnh suy luận YOLO")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    args = parser.parse_args()

    set_deterministic_seed(args.seed)

    cand_model = PROJECT_ROOT / "runs" / "detect" / "runs" / "benchmarks" / "yolov8n_seed42" / "weights" / "best.pt"
    resolved_model = cand_model if cand_model.exists() else args.model

    export_frame_jsonl(
        catalog_path=args.catalog,
        model_path=resolved_model,
        output_dir=args.out_dir,
        sample_fps=args.fps,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
