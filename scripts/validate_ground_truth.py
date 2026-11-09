"""Script kiểm tra và xác thực tính hợp lệ của tệp Ground Truth sự kiện (`scripts/validate_ground_truth.py`).

Kiểm tra:
1. Tính toàn vẹn cú pháp JSONL.
2. Đầy đủ các trường bắt buộc: `video_id`, `camera_id`, `label`, `start_sec`, `end_sec`, `confidence`, `annotator_id`.
3. Phân tách rõ hai nhãn `FIRE` và `SMOKE`.
4. Ràng buộc thời gian: 0 <= start_sec < end_sec.
5. Khoảng thời gian nằm trong thời lượng video (đối chiếu với `data/video_dataset_list.csv`).
6. Điểm tin cậy Ground Truth hợp lệ (0 <= confidence <= 1.0).
7. Không có sự kiện trùng lặp bất thường.
8. Thống kê rõ ràng số lượng nhãn 'verified' và 'pending_verification'.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("validate_ground_truth")


def load_video_durations(catalog_path: Path) -> Dict[str, float]:
    """Đọc thời lượng video từ tệp danh mục catalog."""
    if not catalog_path.exists():
        return {}
    durations: Dict[str, float] = {}
    with catalog_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"].strip()
            try:
                durations[vid] = float(row.get("duration_sec", 0.0))
            except ValueError:
                pass
    return durations


def validate_ground_truth(
    gt_path: Path,
    catalog_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Kiểm tra toàn diện tệp Ground Truth sự kiện.

    Args:
        gt_path: Đường dẫn tệp events.jsonl.
        catalog_path: Đường dẫn tệp video_dataset_list.csv để đối chiếu thời lượng.

    Returns:
        Dictionary chứa chi tiết kết quả kiểm tra và thống kê.
    """
    if not gt_path.exists():
        LOGGER.error("Không tìm thấy tệp Ground Truth: %s", gt_path)
        return {"status": "FAIL", "error": f"Không tìm thấy tệp: {gt_path}"}

    video_durations = load_video_durations(catalog_path) if catalog_path else {}

    total_events = 0
    fire_count = 0
    smoke_count = 0
    verified_count = 0
    pending_count = 0
    annotator_set: Set[str] = set()
    videos_with_gt: Set[str] = set()

    seen_keys: Set[tuple] = set()
    errors: List[str] = []

    with gt_path.open("r", encoding="utf-8") as stream:
        for line_no, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            total_events += 1

            try:
                record = json.loads(line)
            except Exception as exc:
                errors.append(f"Dòng {line_no}: Lỗi giải mã JSON: {exc}")
                continue

            if not isinstance(record, dict):
                errors.append(f"Dòng {line_no}: Bản ghi không phải JSON object")
                continue

            # Kiểm tra trường bắt buộc
            for req_field in ("video_id", "camera_id", "label", "start_sec", "end_sec", "confidence", "annotator_id", "verification_status"):
                if req_field not in record:
                    errors.append(f"Dòng {line_no}: Thiếu trường bắt buộc '{req_field}'")

            vid = str(record.get("video_id", "")).strip()
            cam_id = str(record.get("camera_id", "")).strip()
            label = str(record.get("label", "")).upper()
            annotator = str(record.get("annotator_id", "")).strip()
            status = str(record.get("verification_status", "")).lower().strip()

            if not vid or not cam_id:
                errors.append(f"Dòng {line_no}: video_id hoặc camera_id rỗng")
                continue

            videos_with_gt.add(vid)
            if annotator:
                annotator_set.add(annotator)

            # Kiểm tra nhãn
            if label not in {"FIRE", "SMOKE"}:
                errors.append(f"Dòng {line_no}: Nhãn '{label}' không hợp lệ (phải là FIRE hoặc SMOKE)")
            elif label == "FIRE":
                fire_count += 1
            elif label == "SMOKE":
                smoke_count += 1

            # Kiểm tra trạng thái xác minh và annotator_id
            if status not in {"verified", "pending_verification"}:
                errors.append(
                    f"Dòng {line_no} video {vid}: verification_status '{status}' không hợp lệ "
                    "(chỉ chấp nhận 'verified' hoặc 'pending_verification')"
                )
            elif status == "verified":
                verified_count += 1
                if not annotator or annotator.lower() in {"unspecified", "annotator_pending", "none"}:
                    errors.append(
                        f"Dòng {line_no} video {vid}: Bản ghi verified bắt buộc phải có annotator_id hợp lệ "
                        f"(tìm thấy: '{annotator}')"
                    )
            else:
                pending_count += 1

            # Kiểm tra thời gian
            try:
                start_sec = float(record.get("start_sec", -1.0))
                end_sec = float(record.get("end_sec", -1.0))
            except (TypeError, ValueError):
                errors.append(f"Dòng {line_no}: start_sec hoặc end_sec không phải là số hợp lệ")
                continue

            if start_sec < 0.0:
                errors.append(f"Dòng {line_no} video {vid}: start_sec ({start_sec}) phải >= 0")
            if end_sec <= start_sec:
                errors.append(f"Dòng {line_no} video {vid}: end_sec ({end_sec}) phải lớn hơn start_sec ({start_sec})")

            # Đối chiếu với thời lượng catalog (cho phép sai số 1.0 giây làm tròn)
            if vid in video_durations:
                max_dur = video_durations[vid]
                if end_sec > max_dur + 1.0:
                    errors.append(
                        f"Dòng {line_no} video {vid}: end_sec ({end_sec:.2f}s) vượt quá thời lượng video ({max_dur:.2f}s)"
                    )

            # Kiểm tra trùng lặp
            event_key = (vid, label, round(start_sec, 2), round(end_sec, 2))
            if event_key in seen_keys:
                errors.append(f"Dòng {line_no} video {vid}: Sự kiện trùng lặp {event_key}")
            seen_keys.add(event_key)

            # Kiểm tra confidence
            try:
                conf = float(record.get("confidence", 1.0))
                if not 0.0 <= conf <= 1.0:
                    errors.append(f"Dòng {line_no} video {vid}: confidence ({conf}) ngoài khoảng [0, 1]")
            except (TypeError, ValueError):
                errors.append(f"Dòng {line_no} video {vid}: confidence không phải số hợp lệ")

    schema_valid = len(errors) == 0 and total_events > 0
    annotation_complete = schema_valid and (pending_count == 0)

    report = {
        "status": "PASS" if schema_valid else "FAIL",
        "schema_valid": schema_valid,
        "annotation_complete": annotation_complete,
        "annotation_status": "COMPLETE" if annotation_complete else "INCOMPLETE_PENDING_ANNOTATIONS",
        "ground_truth_file": str(gt_path),
        "metrics": {
            "total_events": total_events,
            "total_videos_with_ground_truth": len(videos_with_gt),
            "fire_events_count": fire_count,
            "smoke_events_count": smoke_count,
            "verified_events_count": verified_count,
            "pending_verification_events_count": pending_count,
            "annotators": sorted(list(annotator_set)),
        },
        "errors_count": len(errors),
        "errors": errors[:20],
    }

    LOGGER.info("Kết quả xác thực Ground Truth: %s (Schema Valid: %s, Complete: %s)", report["status"], schema_valid, annotation_complete)
    LOGGER.info(
        "Tổng cộng %d events trên %d videos (FIRE: %d, SMOKE: %d | Verified: %d, Pending: %d)",
        total_events, len(videos_with_gt), fire_count, smoke_count, verified_count, pending_count
    )
    if errors:
        LOGGER.error("Phát hiện %d lỗi trong tệp Ground Truth:", len(errors))
        for err in errors[:5]:
            LOGGER.error("  - %s", err)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra và xác thực tính hợp lệ của tệp Ground Truth sự kiện.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "ground_truth" / "events.jsonl", help="Đường dẫn file events.jsonl")
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data" / "video_dataset_list.csv", help="Đường dẫn file video_dataset_list.csv")
    args = parser.parse_args()

    report = validate_ground_truth(gt_path=args.input, catalog_path=args.catalog)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

