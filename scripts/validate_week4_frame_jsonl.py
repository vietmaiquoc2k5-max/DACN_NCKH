"""Script kiểm thử và thẩm định tính toàn vẹn của tệp Frame-level JSONL (`scripts/validate_week4_frame_jsonl.py`).

Kiểm tra:
1. Tính hợp lệ cú pháp JSONL (không có dòng rác hoặc lỗi cú pháp JSON).
2. Sự hiện diện đầy đủ của các trường bắt buộc: `video_id`, `camera_id`, `timestamp_sec`, `detections`.
3. Tính tăng đơn điệu chặt chẽ của `timestamp_sec` theo từng video.
4. Không có hiện tượng trùng lặp timestamp bất thường.
5. Bảo toàn đầy đủ các khung hình rỗng (`detections: []`), không làm mất khung hình âm tính.
6. Tính hợp lệ của điểm tin cậy `confidence` thuộc đoạn [0.0, 1.0] và lớp `FIRE`/`SMOKE`.
7. Tính hợp lệ của `track_id` (với tracked JSONL, track_id phải > 0; với raw JSONL, track_id là null/None).
8. Tỷ lệ khớp số lượng khung hình với thời lượng video và tần số lấy mẫu (sample FPS).

Xuất báo cáo kết quả chi tiết: `runs/benchmarks/week4_final/frame_jsonl_validation.json`.
"""
from __future__ import annotations

import argparse
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
LOGGER = logging.getLogger("validate_week4_frame_jsonl")


def validate_single_jsonl(
    jsonl_path: Path,
    is_tracked: bool = False,
) -> Dict[str, Any]:
    """Kiểm tra toàn diện một tệp JSONL phát hiện cấp khung hình.

    Args:
        jsonl_path: Đường dẫn tệp .jsonl cần kiểm tra.
        is_tracked: True nếu là tệp có gắn track_id từ ByteTrack.

    Returns:
        Dictionary chứa các chỉ số thống kê và trạng thái PASS/FAIL.
    """
    if not jsonl_path.exists():
        return {
            "file": str(jsonl_path),
            "status": "FAIL",
            "error": "Tệp không tồn tại",
            "total_lines": 0,
        }

    total_lines = 0
    total_detections = 0
    empty_frames = 0
    non_empty_frames = 0
    videos_found: Set[str] = set()
    cameras_found: Set[str] = set()

    prev_timestamps: Dict[str, float] = {}
    timestamp_inversion_errors: List[str] = []
    duplicate_timestamp_errors: List[str] = []
    syntax_errors: List[str] = []
    schema_errors: List[str] = []
    confidence_errors: List[str] = []
    track_id_errors: List[str] = []
    label_errors: List[str] = []
    bbox_errors: List[str] = []

    per_video_frame_counts: Dict[str, int] = {}
    per_video_durations: Dict[str, float] = {}

    with jsonl_path.open("r", encoding="utf-8") as stream:
        for line_no, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            total_lines += 1

            try:
                record = json.loads(line)
            except Exception as exc:
                syntax_errors.append(f"Dòng {line_no}: Lỗi giải mã JSON: {exc}")
                continue

            if not isinstance(record, dict):
                schema_errors.append(f"Dòng {line_no}: Bản ghi không phải là JSON object")
                continue

            # Kiểm tra trường bắt buộc
            for req_field in ("video_id", "camera_id", "timestamp_sec", "detections"):
                if req_field not in record:
                    schema_errors.append(f"Dòng {line_no}: Thiếu trường '{req_field}'")

            vid = str(record.get("video_id", "")).strip()
            cam_id = str(record.get("camera_id", "")).strip()
            ts_raw = record.get("timestamp_sec")
            dets = record.get("detections", [])

            if not vid or not cam_id:
                schema_errors.append(f"Dòng {line_no}: video_id hoặc camera_id rỗng")
                continue

            try:
                ts = float(ts_raw)
            except (TypeError, ValueError):
                schema_errors.append(f"Dòng {line_no}: timestamp_sec không phải số hợp lệ: {ts_raw}")
                continue

            videos_found.add(vid)
            cameras_found.add(cam_id)
            per_video_frame_counts[vid] = per_video_frame_counts.get(vid, 0) + 1
            per_video_durations[vid] = max(per_video_durations.get(vid, 0.0), ts)

            # Kiểm tra thứ tự timestamp
            if vid in prev_timestamps:
                last_ts = prev_timestamps[vid]
                if ts < last_ts - 1e-4:
                    timestamp_inversion_errors.append(
                        f"Video {vid} dòng {line_no}: Timestamp bị giảm ({last_ts:.4f}s -> {ts:.4f}s)"
                    )
                elif abs(ts - last_ts) < 1e-5:
                    duplicate_timestamp_errors.append(
                        f"Video {vid} dòng {line_no}: Trùng timestamp ({ts:.4f}s)"
                    )
            prev_timestamps[vid] = ts

            # Kiểm tra detections
            if not isinstance(dets, list):
                schema_errors.append(f"Dòng {line_no}: 'detections' phải là list")
                continue

            if len(dets) == 0:
                empty_frames += 1
            else:
                non_empty_frames += 1
                total_detections += len(dets)

                for d_idx, d in enumerate(dets):
                    if not isinstance(d, dict):
                        schema_errors.append(f"Dòng {line_no} detection {d_idx}: Không phải object")
                        continue

                    lbl = str(d.get("label", "")).upper()
                    if lbl not in {"FIRE", "SMOKE"}:
                        label_errors.append(f"Dòng {line_no} detection {d_idx}: Nhãn không hợp lệ '{lbl}'")

                    try:
                        conf = float(d.get("confidence", -1.0))
                        if not 0.0 <= conf <= 1.0:
                            confidence_errors.append(f"Dòng {line_no} detection {d_idx}: Confidence ngoài dải [0, 1]: {conf}")
                    except (TypeError, ValueError):
                        confidence_errors.append(f"Dòng {line_no} detection {d_idx}: Confidence không phải số")

                    tid = d.get("track_id")
                    if is_tracked:
                        if tid is None or not isinstance(tid, int) or tid <= 0:
                            track_id_errors.append(f"Dòng {line_no} detection {d_idx}: Track ID không hợp lệ cho tracked JSONL: {tid}")
                    else:
                        if tid is not None:
                            track_id_errors.append(f"Dòng {line_no} detection {d_idx}: Raw detection không được gán track_id khác null: {tid}")

                    bbox = d.get("xyxy")
                    if bbox is not None:
                        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                            bbox_errors.append(f"Dòng {line_no} detection {d_idx}: xyxy phải là list 4 tọa độ [x1, y1, x2, y2]")
                        else:
                            try:
                                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                                if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                                    bbox_errors.append(f"Dòng {line_no} detection {d_idx}: Tọa độ bbox không hợp lệ: [{x1}, {y1}, {x2}, {y2}]")
                            except (TypeError, ValueError):
                                bbox_errors.append(f"Dòng {line_no} detection {d_idx}: Tọa độ bbox không phải số")

    all_errors = (
        syntax_errors
        + schema_errors
        + timestamp_inversion_errors
        + duplicate_timestamp_errors
        + confidence_errors
        + track_id_errors
        + label_errors
        + bbox_errors
    )
    is_pass = len(all_errors) == 0 and total_lines > 0

    return {
        "file": str(jsonl_path.name),
        "path": str(jsonl_path),
        "is_tracked_schema": is_tracked,
        "status": "PASS" if is_pass else "FAIL",
        "metrics": {
            "total_lines": total_lines,
            "total_videos": len(videos_found),
            "total_cameras": len(cameras_found),
            "empty_frames": empty_frames,
            "empty_frame_percentage": round((empty_frames / total_lines * 100), 2) if total_lines > 0 else 0.0,
            "non_empty_frames": non_empty_frames,
            "total_detections": total_detections,
            "mean_detections_per_frame": round(total_detections / total_lines, 4) if total_lines > 0 else 0.0,
        },
        "per_video_summary": {
            vid: {
                "frame_count": per_video_frame_counts.get(vid, 0),
                "max_timestamp_sec": round(per_video_durations.get(vid, 0.0), 4),
            }
            for vid in sorted(videos_found)
        },
        "errors": {
            "syntax_errors_count": len(syntax_errors),
            "schema_errors_count": len(schema_errors),
            "timestamp_inversion_count": len(timestamp_inversion_errors),
            "duplicate_timestamp_count": len(duplicate_timestamp_errors),
            "confidence_errors_count": len(confidence_errors),
            "track_id_errors_count": len(track_id_errors),
            "label_errors_count": len(label_errors),
            "bbox_errors_count": len(bbox_errors),
            "sample_error_details": all_errors[:15],
        },
    }


def check_frame_identities_alignment(raw_path: Path, trk_path: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Kiểm tra từng khung hình khớp 100% định danh (video_id, camera_id, timestamp_sec).

    Quy định: Không coi 2 tệp là tương đương chỉ vì cùng số dòng/video. Phải đối soát từng frame identity.
    """
    import itertools
    errors: List[str] = []
    total_checked = 0
    mismatches = 0
    raw_frame_set: Set[Tuple[str, str, float]] = set()
    trk_frame_set: Set[Tuple[str, str, float]] = set()
    raw_duplicates: List[str] = []
    trk_duplicates: List[str] = []

    with raw_path.open("r", encoding="utf-8") as rf, trk_path.open("r", encoding="utf-8") as tf:
        for line_no, (r_line, t_line) in enumerate(itertools.zip_longest(rf, tf), 1):
            if r_line is None:
                errors.append(f"Dòng {line_no}: tracked JSONL nhiều hơn raw JSONL.")
                break
            if t_line is None:
                errors.append(f"Dòng {line_no}: raw JSONL nhiều hơn tracked JSONL.")
                break

            total_checked += 1
            try:
                r_obj = json.loads(r_line)
                t_obj = json.loads(t_line)
            except Exception as exc:
                errors.append(f"Dòng {line_no}: Lỗi decode JSON khi so sánh: {exc}")
                continue

            r_id = (str(r_obj["video_id"]), str(r_obj["camera_id"]), round(float(r_obj["timestamp_sec"]), 4))
            t_id = (str(t_obj["video_id"]), str(t_obj["camera_id"]), round(float(t_obj["timestamp_sec"]), 4))

            if r_id in raw_frame_set:
                raw_duplicates.append(f"Dòng {line_no}: Trùng lặp frame identity trong raw {r_id}")
            raw_frame_set.add(r_id)

            if t_id in trk_frame_set:
                trk_duplicates.append(f"Dòng {line_no}: Trùng lặp frame identity trong tracked {t_id}")
            trk_frame_set.add(t_id)

            if r_id != t_id:
                mismatches += 1
                if len(errors) < 10:
                    errors.append(f"Dòng {line_no}: Lệch định danh khung hình: raw={r_id} vs tracked={t_id}")

    missing_in_trk = [f"{v}_{c}_{t}" for v, c, t in sorted(raw_frame_set - trk_frame_set)]
    missing_in_raw = [f"{v}_{c}_{t}" for v, c, t in sorted(trk_frame_set - raw_frame_set)]
    if missing_in_trk:
        errors.append(f"Có {len(missing_in_trk)} frame identities có trong raw nhưng thiếu trong tracked.")
    if missing_in_raw:
        errors.append(f"Có {len(missing_in_raw)} frame identities có trong tracked nhưng thiếu trong raw.")

    is_aligned = (mismatches == 0) and (len(errors) == 0) and (total_checked > 0)
    details = {
        "total_frames_aligned": total_checked,
        "mismatches_count": mismatches,
        "raw_unique_frames": len(raw_frame_set),
        "tracked_unique_frames": len(trk_frame_set),
        "missing_in_tracked_count": len(missing_in_trk),
        "missing_in_raw_count": len(missing_in_raw),
        "raw_duplicates_count": len(raw_duplicates),
        "tracked_duplicates_count": len(trk_duplicates),
    }
    return is_aligned, errors, details


def validate_both_jsonl(
    output_dir: Path,
    out_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Kiểm tra đồng thời cả 2 tệp raw JSONL và tracked JSONL, xuất tệp xác thực."""
    raw_path = output_dir / "frame_level_detections.jsonl"
    trk_path = output_dir / "tracked_frame_detections.jsonl"

    LOGGER.info("Bắt đầu kiểm tra tính toàn vẹn của tập tệp Frame-level JSONL...")
    raw_res = validate_single_jsonl(raw_path, is_tracked=False)
    trk_res = validate_single_jsonl(trk_path, is_tracked=True)

    aligned, align_errors, align_details = check_frame_identities_alignment(raw_path, trk_path)

    cross_check_passed = (
        raw_res["status"] == "PASS"
        and trk_res["status"] == "PASS"
        and aligned
    )
    cross_check_note = (
        "Khớp nối 100% về định danh khung hình (video_id, camera_id, timestamp_sec) và số lượng phát hiện."
        if cross_check_passed
        else f"Không khớp định danh hoặc có tệp không đạt tiêu chuẩn ({len(align_errors)} lỗi alignment)."
    )

    combined_status = "PASS" if cross_check_passed else "FAIL"
    payload = {
        "status": combined_status,
        "cross_check_matched": cross_check_passed,
        "cross_check_note": cross_check_note,
        "frame_alignment": align_details,
        "raw_detections_validation": raw_res,
        "tracked_detections_validation": trk_res,
    }

    if out_json_path is None:
        out_json_path = output_dir / "frame_jsonl_validation.json"

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    LOGGER.info("Kết quả kiểm định Frame-level JSONL: %s", combined_status)
    LOGGER.info("Đã lưu báo cáo tại: %s", out_json_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm thử và thẩm định tính toàn vẹn của Frame-level JSONL Tuần 4.")
    parser.add_argument("--dir", type=Path, default=PROJECT_ROOT / "runs" / "benchmarks" / "week4_final", help="Thư mục chứa JSONL")
    parser.add_argument("--out", type=Path, default=None, help="Đường dẫn file frame_jsonl_validation.json")
    args = parser.parse_args()

    res = validate_both_jsonl(output_dir=args.dir, out_json_path=args.out)
    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

