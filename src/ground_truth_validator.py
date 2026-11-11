"""Quy trình chuẩn hóa và xác thực dữ liệu Ground Truth đa thẩm định (Multi-Annotator Cross-Validation).

Theo đề cương NCKH và Task 1:
- Loại bỏ hoàn toàn việc tự động nâng pending lên verified.
- Tách bạch 4 trạng thái:
  1. pending_verification: Chưa có đủ thẩm định viên đối soát.
  2. annotator_reviewed: Đã có thẩm định viên 1 (annotator_01) xem xét.
  3. cross_verified: Đã có thẩm định viên 2 (annotator_02) kiểm định chéo độc lập,
     sai số |start1 - start2| <= 0.5s và |end1 - end2| <= 0.5s.
  4. disputed: Có sự bất đồng giữa 2 thẩm định viên (sai số > 0.5s) hoặc tranh chấp nhãn.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("ground_truth_cross_validator")

VALID_STATUSES = {
    "pending_verification",
    "annotator_reviewed",
    "cross_verified",
    "verified",  # Tương đương cross_verified cho tính tương thích ngược
    "disputed",
}


def load_video_durations(catalog_path: Path) -> Dict[str, float]:
    """Đọc thời lượng video từ tệp danh mục catalog CSV."""
    durations: Dict[str, float] = {}
    if not catalog_path.exists():
        return durations
    with catalog_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id", "").strip()
            try:
                durations[vid] = float(row.get("duration_sec", 0.0))
            except (ValueError, TypeError):
                pass
    return durations


def validate_single_event(
    record: Dict[str, Any],
    line_no: int,
    video_durations: Dict[str, float],
) -> Tuple[bool, List[str], str]:
    """Kiểm tra tính hợp lệ của một bản ghi sự kiện theo chuẩn khoa học.

    Returns:
        Tuple (is_valid, errors, resolved_status)
    """
    errors: List[str] = []

    # 1. Các trường cơ bản bắt buộc
    for req in ("video_id", "camera_id", "label", "start_sec", "end_sec", "verification_status"):
        if req not in record or record[req] is None:
            errors.append(f"Dòng {line_no}: Thiếu trường bắt buộc '{req}'")

    vid = str(record.get("video_id", "")).strip()
    label = str(record.get("label", "")).upper().strip()
    status = str(record.get("verification_status", "")).lower().strip()

    if label not in {"FIRE", "SMOKE"}:
        errors.append(f"Dòng {line_no} video {vid}: Nhãn '{label}' không hợp lệ (phải là FIRE hoặc SMOKE)")

    if status not in VALID_STATUSES:
        errors.append(f"Dòng {line_no} video {vid}: verification_status '{status}' không hợp lệ")

    # 2. Kiểm tra thời gian
    try:
        start_sec = float(record.get("start_sec", -1.0))
        end_sec = float(record.get("end_sec", -1.0))
    except (ValueError, TypeError):
        errors.append(f"Dòng {line_no} video {vid}: start_sec hoặc end_sec không phải là số hợp lệ")
        return False, errors, status

    if start_sec < 0.0:
        errors.append(f"Dòng {line_no} video {vid}: start_sec ({start_sec}) phải >= 0")
    if end_sec <= start_sec:
        errors.append(f"Dòng {line_no} video {vid}: end_sec ({end_sec}) phải lớn hơn start_sec ({start_sec})")

    # Đối chiếu thời lượng video
    if vid in video_durations:
        max_dur = video_durations[vid]
        if end_sec > max_dur + 1.0:
            errors.append(f"Dòng {line_no} video {vid}: end_sec ({end_sec}s) vượt quá thời lượng video ({max_dur}s)")

    # 3. Kiểm định chéo (Cross-Verification logic)
    ann1 = record.get("annotator_01")
    ann2 = record.get("annotator_02")
    s1 = record.get("start_sec_annotator_01")
    e1 = record.get("end_sec_annotator_01")
    s2 = record.get("start_sec_annotator_02")
    e2 = record.get("end_sec_annotator_02")

    resolved_status = status

    # Kiểm tra sai lệch nếu có cả 2 annotator
    if ann1 and ann2 and s1 is not None and e1 is not None and s2 is not None and e2 is not None:
        try:
            delta_start = abs(float(s1) - float(s2))
            delta_end = abs(float(e1) - float(e2))
            if delta_start > 0.5 or delta_end > 0.5:
                resolved_status = "disputed"
                if status in {"cross_verified", "verified"}:
                    errors.append(
                        f"Dòng {line_no} video {vid}: Trạng thái '{status}' không hợp lệ vì sai lệch giữa 2 annotator "
                        f"(delta_start={delta_start:.2f}s, delta_end={delta_end:.2f}s) vượt quá ngưỡng 0.5s. Phải chuyển thành 'disputed'."
                    )
            else:
                if status in {"cross_verified", "verified"}:
                    resolved_status = "cross_verified"
        except (ValueError, TypeError):
            errors.append(f"Dòng {line_no} video {vid}: Timestamp của annotator không phải số hợp lệ")

    # Quy tắc bắt buộc: Không thể là cross_verified/verified nếu thiếu annotator_02
    if status in {"cross_verified", "verified"}:
        if not ann2 or str(ann2).lower() in {"none", "null", "", "annotator_pending"}:
            errors.append(
                f"Dòng {line_no} video {vid}: Không thể đạt trạng thái '{status}' khi thiếu annotator_02 kiểm định độc lập."
            )

    return len(errors) == 0, errors, resolved_status


def compute_file_sha256(filepath: Path) -> str:
    """Tính mã băm SHA256 của một tệp."""
    sha = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

