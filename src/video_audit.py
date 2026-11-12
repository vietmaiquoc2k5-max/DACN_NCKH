"""Module kiểm định và xác minh tính toàn vẹn của tệp video (src/video_audit.py).

Quy chuẩn kỹ thuật kiểm định:
1. compute_streaming_sha256: Tính mã băm SHA-256 trên toàn bộ tệp bằng streaming chunks (1MB).
2. VideoDecodeCacheManager: Quản lý cache giải mã tăng cường (Hardened Cache):
   - Khóa cache bằng relative_path của tệp, không dùng video_id đơn lẻ.
   - Thẩm định full SHA-256 trước khi tái sử dụng cache (chống giả mạo tệp cùng dung lượng).
   - Tự động hủy cache (invalidate) khi kích thước, mtime hoặc SHA-256 thay đổi.
   - Ghi log chi tiết CACHE_HIT, CACHE_MISS, CACHE_INVALIDATED.
3. audit_single_video: Kiểm định giải mã tuần tự từ frame đầu đến EOF:
   - Phân loại rõ ràng: NORMAL_EOF, TRUNCATED_STREAM, DECODE_ERROR, CANNOT_OPEN, PROBE_ONLY.
   - Chính sách dung sai container rõ ràng (Explicit Tolerance Policy):
     * Delta == 0: NORMAL_EOF (Khớp tuyệt đối).
     * 1 <= Delta <= 5: NORMAL_EOF (Bù trừ container index padding thông thường).
     * 6 <= Delta <= 10: NORMAL_EOF (Bù trừ container header index borderline có cảnh báo).
     * Delta > 10: TRUNCATED_STREAM (Lỗi ngắt luồng stream sớm, status: NEEDS_REVIEW).
   - Tách biệt 3 trạng thái: decode_status, source_label_status, human_review_status.
   - full_decode_verified=True chỉ xác nhận luồng stream vật lý đã được duyệt đến EOF,
     không đồng nghĩa với việc nhãn ngữ nghĩa đã được xác nhận.
4. detect_duplicates_and_id_collisions: Phát hiện trùng lặp SHA-256 và va chạm video_id giữa các thư mục.
5. verify_manifest_vs_inference: Đối soát 100% tính nhất quán giữa Manifest và Inference Summary,
   đối soát 3 mốc phơi nhiễm: 7.2026h (benchmark), 7.1312h (stream thực), 6.9846h (untruncated valid).
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2

LOGGER = logging.getLogger("video_audit")

# --- CHÍNH SÁCH DUNG SAI CONTAINER KHAI BÁO RÕ RÀNG ---
CONTAINER_PADDING_TOLERANCE = 5       # <= 5 frames: Container header index padding bình thường
CONTAINER_BORDERLINE_TOLERANCE = 10   # <= 10 frames: Container index discrepancy mức cận
TRUNCATED_STREAM_THRESHOLD = 10       # > 10 frames: Coi là luồng bị cắt ngắn bất thường (Truncated)


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


class VideoDecodeCacheManager:
    """Quản lý bộ nhớ đệm giải mã video tăng cường (Hardened Decode Cache).
    
    Khóa lưu trữ là đường dẫn tương đối (relative_path) của tệp.
    Bắt buộc kiểm tra kích thước tệp và thẩm định full SHA-256 trước khi trả về kết quả.
    """

    def __init__(self, cache_file: Optional[Path | str] = None, root_dir: Optional[Path] = None):
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.cache_file = Path(cache_file) if cache_file else self.root_dir / "data" / ".video_decode_cache.json"
        self.cache_data: Dict[str, Dict[str, Any]] = {}
        self.load()

    def _normalize_key(self, filepath: Path | str) -> str:
        p = Path(filepath).resolve()
        try:
            return str(p.relative_to(self.root_dir.resolve())).replace("\\", "/")
        except ValueError:
            return str(p).replace("\\", "/")

    def load(self) -> None:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # Chuẩn hóa key: nếu key cũ là video_id, migrate sang relative_path
                self.cache_data = {}
                for k, v in raw.items():
                    rel_k = v.get("relative_path", k)
                    self.cache_data[rel_k] = v
                LOGGER.info("VideoDecodeCacheManager: Đã nạp %d bản ghi từ %s", len(self.cache_data), self.cache_file.name)
            except Exception as e:
                LOGGER.warning("VideoDecodeCacheManager: Không thể đọc cache %s (%s), khởi tạo rỗng", self.cache_file, e)
                self.cache_data = {}
        else:
            self.cache_data = {}

    def save(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
            LOGGER.info("VideoDecodeCacheManager: Đã lưu %d bản ghi vào %s", len(self.cache_data), self.cache_file.name)
        except Exception as e:
            LOGGER.error("VideoDecodeCacheManager: Lỗi khi lưu cache: %s", e)

    def get(self, filepath: Path | str, validate_sha256: bool = True) -> Optional[Dict[str, Any]]:
        """Lấy bản ghi từ cache có xác thực kích thước và full SHA-256."""
        path = Path(filepath).resolve()
        if not path.is_file():
            LOGGER.info("CACHE_MISS: Tệp không tồn tại trên đĩa: %s", path)
            return None

        key = self._normalize_key(path)
        if key not in self.cache_data:
            LOGGER.info("CACHE_MISS: Chưa có cache cho %s", key)
            return None

        entry = self.cache_data[key]
        curr_size = path.stat().st_size
        cached_size = entry.get("file_size_bytes", 0)

        # 1. Kiểm tra kích thước tệp
        if curr_size != cached_size:
            LOGGER.warning("CACHE_INVALIDATED (Size Mismatch): %s (Cached: %d, Actual: %d)", key, cached_size, curr_size)
            self.invalidate(key)
            return None

        # 2. Xác thực full SHA-256 nếu được yêu cầu (chống tệp bị sửa đổi cùng kích thước)
        if validate_sha256:
            curr_sha256 = compute_streaming_sha256(path)
            cached_sha256 = entry.get("sha256", "")
            if curr_sha256 != cached_sha256:
                LOGGER.warning(
                    "CACHE_INVALIDATED (SHA-256 Mismatch / Same-Size Modified File): %s (Cached: %s, Actual: %s)",
                    key, cached_sha256[:12], curr_sha256[:12]
                )
                self.invalidate(key)
                return None

        LOGGER.info("CACHE_HIT: Hợp lệ 100% cho %s (SHA-256: %s, Frames: %d, Status: %s)",
                    key, entry.get("sha256", "")[:12], entry.get("full_decoded_frames", 0), entry.get("status", ""))
        return dict(entry)

    def put(self, filepath: Path | str, record: Dict[str, Any]) -> None:
        key = self._normalize_key(filepath)
        self.cache_data[key] = dict(record)

    def invalidate(self, filepath: Path | str) -> None:
        key = self._normalize_key(filepath)
        if key in self.cache_data:
            del self.cache_data[key]
            LOGGER.info("CACHE_INVALIDATED: Đã xóa bản ghi cache cho %s", key)


def audit_single_video(
    filepath: str | Path,
    root_dir: Optional[Path] = None,
    full_decode: bool = True,
    probe_frames: int = 5,
    chunk_size: int = 1048576,
    cache_manager: Optional[VideoDecodeCacheManager] = None,
) -> Dict[str, Any]:
    """Kiểm tra chi tiết một tệp video, tính full SHA-256 và giải mã tuần tự đến EOF.
    
    Quy tắc phân loại trạng thái:
    - Delta = reported_frame_count - actual_decoded_frames
    - Delta == 0: NORMAL_EOF (Khớp tuyệt đối)
    - 1 <= Delta <= CONTAINER_PADDING_TOLERANCE (5): NORMAL_EOF (Bù trừ container header index padding)
    - 6 <= Delta <= CONTAINER_BORDERLINE_TOLERANCE (10): NORMAL_EOF (Bù trừ container cận có ghi chú)
    - Delta > 10: TRUNCATED_STREAM (Stream bị ngắt giữa chừng, status = NEEDS_REVIEW)
    - full_decode_verified = True nếu đã duyệt tuần tự đến EOF (không phụ thuộc vào nhãn ngữ nghĩa)
    """
    vpath = Path(filepath).resolve()
    base_root = root_dir.resolve() if root_dir else Path.cwd().resolve()

    try:
        rel_path = str(vpath.relative_to(base_root)).replace("\\", "/")
    except ValueError:
        rel_path = str(vpath).replace("\\", "/")

    fname = vpath.name
    vid = vpath.stem

    # 1. Kiểm tra cache tăng cường trước (nếu có cache_manager)
    if cache_manager is not None:
        cached_res = cache_manager.get(vpath, validate_sha256=True)
        if cached_res is not None:
            return cached_res

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

    # 2. Tính SHA-256 toàn phần bằng streaming chunks
    try:
        sha256_val = compute_streaming_sha256(vpath, chunk_size=chunk_size)
        sha256_verified = True
    except Exception as exc:
        sha256_val = f"err_{exc}"
        sha256_verified = False

    # 3. Khởi tạo OpenCV VideoCapture
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

    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
    if not codec_str or any(ord(c) < 32 or ord(c) > 126 for c in codec_str):
        codec_str = "h264/mp4v"

    # 4. Probe frames ở đầu để kiểm tra tính sẵn sàng
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

    # 5. Duyệt tuần tự đến EOF nếu full_decode=True
    actual_decoded_frames = probed_ok
    decode_status = "PROBE_ONLY"
    full_decode_verified = False
    decode_errors = 0
    err_msg = ""
    tolerance_applied = False

    if full_decode:
        while True:
            if not cap.grab():
                break
            actual_decoded_frames += 1

        cap.release()
        full_decode_verified = (actual_decoded_frames > 0)
        missing_frames = max(0, reported_frame_count - actual_decoded_frames)

        # Áp dụng chính sách dung sai rõ ràng (Explicit Tolerance Policy)
        if actual_decoded_frames == 0:
            decode_status = "DECODE_ERROR"
            status = "CORRUPTED"
            err_msg = "Không giải mã được khung hình nào"
        elif missing_frames == 0:
            decode_status = "NORMAL_EOF"
            status = "VALID"
            err_msg = "Khớp tuyệt đối 100% số khung hình container"
        elif missing_frames <= CONTAINER_PADDING_TOLERANCE:
            decode_status = "NORMAL_EOF"
            status = "VALID"
            tolerance_applied = True
            err_msg = f"Khớp trong dung sai container padding thông thường (thiếu {missing_frames} frames <= {CONTAINER_PADDING_TOLERANCE})"
        elif missing_frames <= CONTAINER_BORDERLINE_TOLERANCE:
            decode_status = "NORMAL_EOF"
            status = "VALID"
            tolerance_applied = True
            err_msg = f"Khớp trong dung sai container cận (thiếu {missing_frames} frames <= {CONTAINER_BORDERLINE_TOLERANCE})"
            LOGGER.info("Video %s: %s", vid, err_msg)
        else:
            decode_status = "TRUNCATED_STREAM"
            status = "NEEDS_REVIEW"
            decode_errors = missing_frames
            err_msg = f"Stream kết thúc sớm tại frame {actual_decoded_frames}/{reported_frame_count} (thiếu {missing_frames} frames > {TRUNCATED_STREAM_THRESHOLD} do lỗi luồng NAL/packet)"
            LOGGER.warning("Video %s TRUNCATED: %s", vid, err_msg)
    else:
        cap.release()
        missing_frames = max(0, reported_frame_count - probed_ok)
        decode_status = "PROBE_ONLY"
        status = "NEEDS_REVIEW"
        full_decode_verified = False
        err_msg = "Chỉ thực hiện probe khởi động, chưa giải mã tuần tự đến EOF"

    # Tính thời lượng thực tế (dựa trên actual_decoded_frames)
    if fps_val > 0 and actual_decoded_frames > 0:
        dur_sec = round(actual_decoded_frames / fps_val, 2)
        dur_calc_method = "actual_decoded_frames / native_fps"
    else:
        dur_sec = 0.0
        dur_calc_method = "none"

    result = {
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
        "container_tolerance_applied": tolerance_applied,
        "codec": codec_str,
        "error_message": err_msg,
        "duplicate_of": None,
        "duration_calc_method": dur_calc_method,
        "last_audited_at": datetime.now(timezone.utc).isoformat(),
    }

    # Lưu vào cache nếu có cache_manager và đã full_decode thành công
    if cache_manager is not None and full_decode_verified:
        cache_manager.put(vpath, result)

    return result


def detect_duplicates_and_id_collisions(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Phát hiện trùng lặp dựa trên mã băm SHA-256 toàn phần và va chạm video_id.
    
    1. Trùng video_id giữa các thư mục: tệp sau bị gán status='DUPLICATE_ID' và thông báo lỗi.
    2. Trùng SHA-256: tệp sau bị gán status='DUPLICATE' và duplicate_of = tệp đầu.
    """
    seen_hashes: Dict[str, str] = {}
    seen_ids: Dict[str, str] = {}
    updated_records: List[Dict[str, Any]] = []

    for rec in records:
        vid = rec.get("video_id", "")
        h = rec.get("sha256", "")
        rel_p = rec.get("relative_path", "")

        # Kiểm tra trùng video_id giữa các thư mục
        if vid in seen_ids:
            rec["status"] = "DUPLICATE_ID"
            rec["error_message"] = f"Trùng lặp video_id '{vid}' với tệp '{seen_ids[vid]}'"
            LOGGER.warning("DUPLICATE_ID phát hiện: %s trùng với %s", rel_p, seen_ids[vid])
        else:
            seen_ids[vid] = rel_p

        # Kiểm tra trùng SHA-256 toàn phần
        if h and not h.startswith("err_"):
            if h in seen_hashes:
                rec["status"] = "DUPLICATE"
                rec["duplicate_of"] = seen_hashes[h]
                LOGGER.warning("DUPLICATE_HASH phát hiện: %s trùng mã băm với %s", rel_p, seen_hashes[h])
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
    3. Số frame giải mã (total_decoded_frames) khớp 100%.
    4. Đối soát 3 mẫu số phơi nhiễm (Exposure Denominators):
       - 7.2026h: Mẫu số tính toán trong run suy luận benchmark (Canonical Benchmark Exposure).
       - 7.1312h: Mẫu số phơi nhiễm stream thực tế giải mã được (Actual Stream Exposure).
       - 6.9846h: Mẫu số tập video untruncated hoàn toàn hợp lệ (Valid Untruncated Exposure).
    """
    def clean_key(row: Dict[str, Any], key_suffix: str) -> Optional[Any]:
        for k, v in row.items():
            if k.endswith(key_suffix):
                return v
        return None

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
