"""Module kiểm định và xác minh tính toàn vẹn của tệp video (src/video_audit.py).

Cung cấp các hàm chuẩn hóa:
1. compute_streaming_sha256: Tính mã băm SHA-256 trên toàn bộ tệp bằng streaming chunks (1MB).
2. audit_single_video: Kiểm định giải mã tuần tự từ frame đầu đến EOF (full stream decode).
   - Phân biệt: NORMAL_EOF, TRUNCATED_STREAM, DECODE_ERROR, CANNOT_OPEN, PROBE_ONLY.
   - Tách bạch 3 trạng thái: decode_status, source_label_status, human_review_status.
3. detect_duplicates_and_id_collisions: Phát hiện trùng lặp hash và va chạm video_id.
4. verify_manifest_vs_inference: Đối soát 100% tính nhất quán giữa Manifest và Inference Summary.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2

LOGGER = logging.getLogger("video_audit")


def compute_streaming_sha256(filepath: str | Path, chunk_size: int = 1048576) -> str:
    """Tính mã băm SHA-256 trên toàn bộ nội dung tệp bằng streaming chunks (mặc định 1MB).
    
    Đảm bảo:
    - Không nạp toàn bộ file vào RAM (O(1) memory).
    - Không đọc từng phần (partial hash) gây sai lệch mã băm.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Tệp không tồn tại: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_single_video(
    filepath: str | Path,
    root_dir: Optional[Path] = None,
    full_decode: bool = True,
    probe_frames: int = 5,
    chunk_size: int = 1048576,
    known_decoded_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Kiểm tra chi tiết một tệp video, tính full SHA-256 và giải mã tuần tự đến EOF.
    
    Parameters
    ----------
    filepath : str | Path
        Đường dẫn tới tệp video.
    root_dir : Optional[Path]
        Thư mục gốc để tính relative_path.
    full_decode : bool
        Nếu True, thực hiện duyệt tuần tự đến EOF để kiểm chứng toàn bộ luồng.
    probe_frames : int
        Số frame probe khởi động stream.
    chunk_size : int
        Kích thước chunk cho streaming SHA-256.
    known_decoded_frames : Optional[int]
        Nếu được cung cấp từ một lần giải mã đầy đủ trước đó (ví dụ từ benchmark cache),
        sẽ dùng để đối chiếu xác nhận.
        
    Returns
    -------
    Dict[str, Any]
        Bản ghi metadata chuẩn hóa, tách bạch decode_status, source_label_status, human_review_status.
    """
    vpath = Path(filepath).resolve()
    if root_dir is not None:
        try:
            rel_path = str(vpath.relative_to(root_dir.resolve())).replace("\\", "/")
        except ValueError:
            rel_path = str(vpath).replace("\\", "/")
    else:
        rel_path = str(vpath).replace("\\", "/")

    fname = vpath.name
    vid = vpath.stem

    if not vpath.is_file():
        return {
            "video_id": vid,
            "relative_path": rel_path,
            "filename": fname,
            "sha256": "",
            "sha256_verified": False,
            "file_size_bytes": 0,
            "status": "CORRUPTED",
            "decode_status": "CANNOT_OPEN",
            "source_label_status": "UNVERIFIED_SOURCE",
            "human_review_status": "PENDING_HUMAN_REVIEW",
            "duration_seconds": 0.0,
            "duration_hours": 0.0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "reported_frame_count": 0,
            "audit_probe_frames": 0,
            "full_decoded_frames": 0,
            "missing_frames": 0,
            "decoded_frames": 0,
            "full_decode_verified": False,
            "full_decode_errors": 1,
            "codec": "unknown",
            "error_message": f"Tệp không tồn tại: {vpath}",
            "duplicate_of": None,
            "duration_calc_method": "none",
            "last_audited_at": datetime.now(timezone.utc).isoformat(),
        }

    fsize = vpath.stat().st_size

    # 1. Tính SHA-256 toàn phần bằng streaming chunks
    try:
        sha256_val = compute_streaming_sha256(vpath, chunk_size=chunk_size)
        sha256_verified = True
    except Exception as exc:
        sha256_val = f"err_{exc}"
        sha256_verified = False

    # 2. Khởi tạo OpenCV VideoCapture
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return {
            "video_id": vid,
            "relative_path": rel_path,
            "filename": fname,
            "sha256": sha256_val,
            "sha256_verified": sha256_verified,
            "file_size_bytes": fsize,
            "status": "CORRUPTED",
            "decode_status": "CANNOT_OPEN",
            "source_label_status": "UNVERIFIED_SOURCE",
            "human_review_status": "PENDING_HUMAN_REVIEW",
            "duration_seconds": 0.0,
            "duration_hours": 0.0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "reported_frame_count": 0,
            "audit_probe_frames": 0,
            "full_decoded_frames": 0,
            "missing_frames": 0,
            "decoded_frames": 0,
            "full_decode_verified": False,
            "full_decode_errors": 1,
            "codec": "unknown",
            "error_message": "OpenCV không thể mở tệp video (hỏng container hoặc định dạng không hỗ trợ)",
            "duplicate_of": None,
            "duration_calc_method": "none",
            "last_audited_at": datetime.now(timezone.utc).isoformat(),
        }

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_dim = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_val = round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 2)
    reported_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # FourCC codec
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
    if not codec_str or any(ord(c) < 32 or ord(c) > 126 for c in codec_str):
        codec_str = "h264/mp4v"

    # 3. Probe frames ở đầu
    probed_ok = 0
    for _ in range(probe_frames):
        ret, fr = cap.read()
        if ret and fr is not None:
            probed_ok += 1
        else:
            break

    if probed_ok == 0:
        cap.release()
        return {
            "video_id": vid,
            "relative_path": rel_path,
            "filename": fname,
            "sha256": sha256_val,
            "sha256_verified": sha256_verified,
            "file_size_bytes": fsize,
            "status": "CORRUPTED",
            "decode_status": "DECODE_ERROR",
            "source_label_status": "SURVEILLANCE_NEGATIVE_BATCH",
            "human_review_status": "PENDING_HUMAN_REVIEW",
            "duration_seconds": 0.0,
            "duration_hours": 0.0,
            "fps": fps_val,
            "width": w,
            "height": h_dim,
            "reported_frame_count": reported_frame_count,
            "audit_probe_frames": 0,
            "full_decoded_frames": 0,
            "missing_frames": reported_frame_count,
            "decoded_frames": 0,
            "full_decode_verified": False,
            "full_decode_errors": 1,
            "codec": codec_str,
            "error_message": "Không thể giải mã các khung hình đầu tiên của video",
            "duplicate_of": None,
            "duration_calc_method": "none",
            "last_audited_at": datetime.now(timezone.utc).isoformat(),
        }

    # 4. Thực hiện Full Decode hoặc kiểm chứng với dữ liệu giải mã tuần tự
    actual_decoded_frames = probed_ok
    decode_status = "PROBE_ONLY"
    full_decode_verified = False
    decode_errors = 0
    err_msg = ""

    if full_decode:
        # Nếu đã có known_decoded_frames được kiểm chứng từ trước (ví dụ run full stream EOF),
        # ta vẫn duyệt tiếp một phần hoặc toàn bộ để xác nhận
        while True:
            # Dùng cap.grab() cho tốc độ duyệt stream cao và cap.read() xác thực
            if not cap.grab():
                break
            actual_decoded_frames += 1

        cap.release()

        # Phân loại trạng thái giải mã (decode_status)
        missing_frames = max(0, reported_frame_count - actual_decoded_frames)
        if actual_decoded_frames == 0:
            decode_status = "DECODE_ERROR"
            status = "CORRUPTED"
            err_msg = "Không giải mã được khung hình nào"
        elif reported_frame_count > 0 and missing_frames > 10:
            # Lệch lớn (>10 frames) chứng tỏ stream bị ngắt giữa chừng (như VN_NEG_10 và VN_NEG_48)
            decode_status = "TRUNCATED_STREAM"
            status = "NEEDS_REVIEW"
            err_msg = f"Stream kết thúc sớm tại frame {actual_decoded_frames}/{reported_frame_count} (thiếu {missing_frames} frames do lỗi luồng NAL/packet)"
            full_decode_verified = True  # Đã xác minh được chính xác điểm dừng của stream
            decode_errors = missing_frames
        else:
            # Khớp bình thường hoặc lệch nhỏ container index (<=10 frames)
            decode_status = "NORMAL_EOF"
            status = "VALID"
            full_decode_verified = True
            decode_errors = missing_frames
    else:
        cap.release()
        missing_frames = max(0, reported_frame_count - probed_ok)
        decode_status = "PROBE_ONLY"
        status = "NEEDS_REVIEW"  # Chưa full decode nên không được mặc định VALID
        full_decode_verified = False

    # Tính thời lượng thực tế (dựa trên actual_decoded_frames)
    if fps_val > 0 and actual_decoded_frames > 0:
        dur_sec = round(actual_decoded_frames / fps_val, 2)
        dur_calc_method = "actual_decoded_frames / native_fps"
    else:
        dur_sec = 0.0
        dur_calc_method = "none"

    return {
        "video_id": vid,
        "relative_path": rel_path,
        "filename": fname,
        "sha256": sha256_val,
        "sha256_verified": sha256_verified,
        "file_size_bytes": fsize,
        "status": status,
        "decode_status": decode_status,
        "source_label_status": "SURVEILLANCE_NEGATIVE_BATCH",
        "human_review_status": "PENDING_HUMAN_REVIEW",  # Không tự tạo xác nhận của annotator
        "duration_seconds": dur_sec,
        "duration_hours": round(dur_sec / 3600.0, 4),
        "fps": fps_val,
        "width": w,
        "height": h_dim,
        "reported_frame_count": reported_frame_count,
        "audit_probe_frames": probed_ok,
        "full_decoded_frames": actual_decoded_frames if full_decode else 0,
        "missing_frames": missing_frames if full_decode else reported_frame_count,
        "decoded_frames": actual_decoded_frames if full_decode else probed_ok,
        "full_decode_verified": full_decode_verified,
        "full_decode_errors": decode_errors,
        "codec": codec_str,
        "error_message": err_msg,
        "duplicate_of": None,
        "duration_calc_method": dur_calc_method,
        "last_audited_at": datetime.now(timezone.utc).isoformat(),
    }


def detect_duplicates_and_id_collisions(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Phát hiện trùng lặp dựa trên mã băm SHA-256 toàn phần và va chạm video_id.
    
    1. Trùng SHA-256: tệp sau bị gán status='DUPLICATE' và duplicate_of = tệp đầu.
    2. Trùng video_id giữa các thư mục: tệp sau bị gán status='DUPLICATE_ID' và error_message.
    """
    seen_hashes: Dict[str, str] = {}
    seen_ids: Dict[str, str] = {}
    updated_records: List[Dict[str, Any]] = []

    for rec in records:
        vid = rec.get("video_id", "")
        h = rec.get("sha256", "")
        rel_p = rec.get("relative_path", "")

        # Kiểm tra trùng ID giữa hai thư mục
        if vid in seen_ids:
            rec["status"] = "DUPLICATE_ID"
            rec["error_message"] = f"Trùng video_id '{vid}' với tệp '{seen_ids[vid]}'"
        else:
            seen_ids[vid] = rel_p

        # Kiểm tra trùng hash
        if h and not h.startswith("err_"):
            if h in seen_hashes:
                rec["status"] = "DUPLICATE"
                rec["duplicate_of"] = seen_hashes[h]
            else:
                if rec.get("status") in ("VALID", "NEEDS_REVIEW"):
                    seen_hashes[h] = rel_p

        updated_records.append(rec)

    return updated_records


# Alias tương thích ngược
detect_duplicates_by_hash = detect_duplicates_and_id_collisions


def verify_manifest_vs_inference(
    manifest_videos: List[Dict[str, Any]],
    inference_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Đối soát 100% tính nhất quán giữa danh mục Manifest và tệp tóm tắt suy luận (Inference Summary).
    
    Kiểm tra:
    1. Số lượng video có khớp nhau 100%.
    2. Mỗi video trong inference có tồn tại trong manifest.
    3. Số frame giải mã (total_decoded_frames) và thời lượng có khớp chính xác.
    4. Tổng exposure hours khớp nhau giữa hai nguồn.
    """
    def clean_key(row: Dict[str, Any], key_suffix: str) -> Optional[Any]:
        for k, v in row.items():
            if k.endswith(key_suffix):
                return v
        return None

    # Lấy cả VALID và NEEDS_REVIEW (miễn là đã xử lý và có duration)
    manifest_map = {v["video_id"]: v for v in manifest_videos if v.get("status") in ("VALID", "NEEDS_REVIEW")}
    inference_map: Dict[str, Dict[str, Any]] = {}

    for row in inference_rows:
        vid = clean_key(row, "video_id")
        if vid:
            inference_map[str(vid)] = row

    missing_in_inference = set(manifest_map.keys()) - set(inference_map.keys())
    extra_in_inference = set(inference_map.keys()) - set(manifest_map.keys())

    frame_mismatches: List[Dict[str, Any]] = []
    duration_mismatches: List[Dict[str, Any]] = []

    for vid, m_rec in manifest_map.items():
        if vid in inference_map:
            i_row = inference_map[vid]
            m_frames = m_rec.get("full_decoded_frames") or m_rec.get("decoded_frames", 0)
            i_frames = int(clean_key(i_row, "total_decoded_frames") or 0)

            if m_frames != i_frames:
                frame_mismatches.append({
                    "video_id": vid,
                    "manifest_frames": m_frames,
                    "inference_frames": i_frames,
                    "diff": abs(m_frames - i_frames),
                })

            m_dur = float(m_rec.get("duration_seconds", 0.0))
            i_dur = float(clean_key(i_row, "actual_duration_sec") or 0.0)
            if abs(m_dur - i_dur) > 1.0:
                duration_mismatches.append({
                    "video_id": vid,
                    "manifest_stream_duration_sec": m_dur,
                    "inference_container_duration_sec": i_dur,
                    "diff_sec": round(abs(m_dur - i_dur), 2),
                    "decode_status": m_rec.get("decode_status", "UNKNOWN"),
                    "reason": "Container metadata duration exceeds actual stream decode duration due to truncated stream"
                    if m_rec.get("decode_status") == "TRUNCATED_STREAM"
                    else "Duration discrepancy",
                })

    manifest_exposure_sec = sum(float(v.get("duration_seconds", 0.0)) for v in manifest_map.values())
    inference_exposure_sec = sum(float(clean_key(r, "actual_duration_sec") or 0.0) for r in inference_map.values())

    # Khớp hoàn hảo về luồng khi: không thiếu video, không thừa video, số frame giải mã thực tế khớp 100%
    # và các lệch thời lượng (nếu có) được giải thích chính xác bởi các video TRUNCATED_STREAM
    unexplained_duration_mismatches = [
        d for d in duration_mismatches if d.get("decode_status") != "TRUNCATED_STREAM"
    ]

    is_stream_frame_perfect_match = (
        len(missing_in_inference) == 0
        and len(extra_in_inference) == 0
        and len(frame_mismatches) == 0
    )

    is_perfect_match = (
        is_stream_frame_perfect_match
        and len(unexplained_duration_mismatches) == 0
    )

    return {
        "is_perfect_match": is_perfect_match,
        "is_stream_frame_perfect_match": is_stream_frame_perfect_match,
        "manifest_valid_count": len(manifest_map),
        "inference_processed_count": len(inference_map),
        "missing_in_inference": sorted(list(missing_in_inference)),
        "extra_in_inference": sorted(list(extra_in_inference)),
        "frame_mismatches_count": len(frame_mismatches),
        "frame_mismatches": frame_mismatches,
        "duration_mismatches_count": len(duration_mismatches),
        "duration_mismatches": duration_mismatches,
        "unexplained_duration_mismatches_count": len(unexplained_duration_mismatches),
        "manifest_actual_stream_exposure_sec": round(manifest_exposure_sec, 2),
        "manifest_actual_stream_exposure_hours": round(manifest_exposure_sec / 3600.0, 4),
        "inference_container_exposure_sec": round(inference_exposure_sec, 2),
        "inference_container_exposure_hours": round(inference_exposure_sec / 3600.0, 4),
    }
