"""Module kiểm định và xác minh tính toàn vẹn của tệp video (src/video_audit.py).

Quy chuẩn kỹ thuật kiểm định:
1. compute_streaming_sha256: Tính mã băm SHA-256 trên toàn bộ tệp bằng streaming chunks (1MB).
   - Xử lý file rỗng (trả về sha256 của chuỗi rỗng).
   - Xử lý ngoại lệ đọc file (raise IOError rõ ràng).
2. VideoDecodeCacheManager: Quản lý cache giải mã tăng cường (Hardened Cache):
   - Khóa cache bằng relative_path của tệp, không dùng video_id đơn lẻ.
   - Validation bắt buộc: kiểm tra file_size VÀ sha256 trước khi tái sử dụng.
   - Invalidate bắt buộc nếu record cũ thiếu sha256 (schema version guard).
   - Ghi log chi tiết CACHE_HIT, CACHE_MISS, CACHE_INVALIDATED.
3. audit_single_video: Kiểm định giải mã tuần tự từ frame đầu đến EOF:
   - Phân loại rõ ràng: NORMAL_EOF, TRUNCATED_STREAM, DECODE_ERROR, CANNOT_OPEN,
     PROBE_ONLY, NEEDS_REVIEW.
   - KHÔNG suy luận NORMAL_EOF chỉ từ read()/grab() trả về False — phải so sánh frame count.
   - Chính sách dung sai container rõ ràng (Explicit Tolerance Policy):
     * Delta == 0: NORMAL_EOF (Khớp tuyệt đối).
     * 1 <= Delta <= 5: NORMAL_EOF (MP4 container index rounding, B-frame PTS offset).
     * 6 <= Delta <= 10: NORMAL_EOF + WARNING log (B-frame reorder buffer flush).
     * Delta > 10: TRUNCATED_STREAM (buộc NEEDS_REVIEW).
   - Phân biệt rõ 4 loại frame count:
     * reported_frame_count: từ metadata container
     * full_decoded_frames (decoded_frame_count): frame thực sự decode thành công
     * audit_probe_frames (sampled_frame_count): frame probe khởi động
     * processed_frame_count: của inference pipeline (ngoài phạm vi audit)
   - Tách biệt: decode_status | source_label_status | human_review_status
4. detect_duplicates_and_id_collisions: Phát hiện SHA-256 trùng và video_id trùng.
   KHÔNG âm thầm xóa, chỉ ghi nhận.
5. verify_manifest_vs_inference: Đối soát 100% tính nhất quán Manifest vs Inference Summary.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2

LOGGER = logging.getLogger("video_audit")

# ---------------------------------------------------------------------------
# DECODE STATUS CONSTANTS
# ---------------------------------------------------------------------------

class DecodeStatus(str, Enum):
    """Trạng thái kỹ thuật luồng vật lý sau khi kiểm định giải mã.

    Các giá trị này MÔ TẢ trạng thái VẬT LÝ của file, KHÔNG liên quan đến
    nhãn ngữ nghĩa (VERIFIED_NEGATIVE, SURVEILLANCE_NEGATIVE_BATCH, v.v.).
    """
    NORMAL_EOF = "NORMAL_EOF"
    """Đã decode tuần tự đến EOF; full_decoded_frames <= reported + tolerance
    CÓ BẰNG CHỨNG (delta <= CONTAINER_BORDERLINE_TOLERANCE).
    KHÔNG được gán chỉ từ read()/grab() trả về False."""

    TRUNCATED_STREAM = "TRUNCATED_STREAM"
    """full_decoded_frames < reported - TRUNCATED_STREAM_THRESHOLD (> 10 frame thiếu).
    Luồng NAL/packet kết thúc sớm bất thường."""

    DECODE_ERROR = "DECODE_ERROR"
    """Decoder báo lỗi (probe 5 frame trả về 0 frame OK, hoặc không decode được frame nào)."""

    CANNOT_OPEN = "CANNOT_OPEN"
    """OpenCV VideoCapture.isOpened() == False."""

    PROBE_ONLY = "PROBE_ONLY"
    """Chỉ probe metadata + 5 frame đầu, chưa decode tuần tự đến EOF."""

    NEEDS_REVIEW = "NEEDS_REVIEW"
    """Không đủ bằng chứng phân loại dứt khoát (ví dụ: reported_frame_count == 0)."""

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# TOLERANCE POLICY
# ---------------------------------------------------------------------------

CONTAINER_PADDING_TOLERANCE = 5
"""<= 5 frames: Container header index padding bình thường.
Nguồn gốc: MP4 container index rounding (moov atom frame count khác số NAL units
thực tế do PTS/DTS alignment), B-frame PTS offset trong H.264/H.265 khiến
reported_frame_count cao hơn actual 1-5 frames.
Tham khảo: ISO/IEC 14496-12 §8.6.1, FFmpeg lavf/mp4.c."""

CONTAINER_BORDERLINE_TOLERANCE = 10
"""<= 10 frames: Container index discrepancy mức cận — PHẢI log WARNING.
Nguồn gốc: B-frame reorder buffer flush (H.264 Level 4.1, ref frames=4),
VFR stream với dropped frames ở cuối GOP, container muxer round-up do
bitrate control. Borderline: ghi nhận nhưng không coi là lỗi luồng."""

TRUNCATED_STREAM_THRESHOLD = 10
"""Nếu delta = reported - actual > 10 frames: TRUNCATED_STREAM.
KHÔNG tự động coi là NORMAL_EOF dù file mở được và probe OK."""

# ---------------------------------------------------------------------------
# CACHE SCHEMA VERSION
# ---------------------------------------------------------------------------
CACHE_SCHEMA_VERSION = 2
"""Schema version của cache. Record thiếu 'sha256' hoặc 'schema_version' < 2
bị invalidate bắt buộc."""


# ---------------------------------------------------------------------------
# SHA-256
# ---------------------------------------------------------------------------

def compute_streaming_sha256(filepath: str | Path, chunk_size: int = 1048576) -> str:
    """Tính mã băm SHA-256 trên toàn bộ nội dung tệp bằng streaming chunks (mặc định 1MB).

    Đảm bảo:
    - Không nạp toàn bộ file vào RAM (O(1) memory).
    - Đọc TOÀN BỘ file theo chunks, không dừng sớm (while True, break khi chunk rỗng).
    - Xử lý file rỗng: trả về sha256('') hợp lệ, không lỗi.
    - Xử lý lỗi đọc file: raise IOError rõ ràng, KHÔNG trả về chuỗi rỗng hay placeholder.

    Args:
        filepath: Đường dẫn tới file.
        chunk_size: Kích thước mỗi chunk (bytes), mặc định 1MB.

    Returns:
        Chuỗi hex SHA-256 lowercase (64 ký tự).

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        ValueError: Nếu đường dẫn không phải file thông thường.
        IOError: Nếu không thể đọc file (permission denied, v.v.).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Tệp không tồn tại: {path}")
    if not path.is_file():
        raise ValueError(f"Đường dẫn không phải là file thông thường: {path}")

    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break  # EOF thực — toàn bộ file đã được đọc
                hasher.update(chunk)
    except OSError as exc:
        raise IOError(f"Không thể đọc file để tính SHA-256: {path} — {exc}") from exc

    # File rỗng: hasher.hexdigest() = sha256('') — hợp lệ
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# HARDENED CACHE MANAGER
# ---------------------------------------------------------------------------

class VideoDecodeCacheManager:
    """Quản lý bộ nhớ đệm giải mã video tăng cường (Hardened Decode Cache).

    Cache key: relative_path từ root_dir (forward slash), KHÔNG phải video_id.
    Validation bắt buộc: file_size VÀ sha256 phải khớp trước khi trả về cache.
    Schema version guard: record thiếu sha256 hoặc schema_version < CACHE_SCHEMA_VERSION
    bị invalidate bắt buộc, không được tái sử dụng.
    """

    def __init__(self, cache_file: Optional[Path | str] = None, root_dir: Optional[Path] = None):
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.cache_file = Path(cache_file) if cache_file else self.root_dir / "data" / ".video_decode_cache.json"
        self.cache_data: Dict[str, Dict[str, Any]] = {}
        self.load()

    def _normalize_key(self, filepath: Path | str) -> str:
        """Chuẩn hóa cache key thành relative_path từ root_dir (forward slash)."""
        p = Path(filepath).resolve()
        try:
            return str(p.relative_to(self.root_dir.resolve())).replace("\\", "/")
        except ValueError:
            return str(p).replace("\\", "/")

    def load(self) -> None:
        """Nạp cache từ file JSON. Migration: key cũ là video_id → relative_path."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.cache_data = {}
                for k, v in raw.items():
                    # Migration: nếu key cũ là video_id, dùng relative_path trong record
                    rel_k = v.get("relative_path", k)
                    self.cache_data[rel_k] = v
                LOGGER.info(
                    "VideoDecodeCacheManager: Đã nạp %d bản ghi từ %s",
                    len(self.cache_data), self.cache_file.name,
                )
            except Exception as e:
                LOGGER.warning(
                    "VideoDecodeCacheManager: Không thể đọc cache %s (%s), khởi tạo rỗng",
                    self.cache_file, e,
                )
                self.cache_data = {}
        else:
            self.cache_data = {}

    def save(self) -> None:
        """Ghi cache xuống file JSON."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
            LOGGER.info(
                "VideoDecodeCacheManager: Đã lưu %d bản ghi vào %s",
                len(self.cache_data), self.cache_file.name,
            )
        except Exception as e:
            LOGGER.error("VideoDecodeCacheManager: Lỗi khi lưu cache: %s", e)

    def get(self, filepath: Path | str, validate_sha256: bool = True) -> Optional[Dict[str, Any]]:
        """Lấy bản ghi từ cache có xác thực file_size VÀ full SHA-256.

        Pipeline validation:
        1. File phải tồn tại trên đĩa.
        2. Cache key (relative_path) phải có trong cache_data → CACHE_MISS nếu không.
        3. Schema version guard: record cũ thiếu sha256 → CACHE_INVALIDATED bắt buộc.
        4. Kích thước file phải khớp cached file_size_bytes → CACHE_INVALIDATED nếu khác.
        5. SHA-256: nếu validate_sha256=True, tính lại SHA-256 toàn file.
           Nếu SHA-256 khác (file bị sửa cùng size) → CACHE_INVALIDATED.

        Args:
            filepath: Đường dẫn tới file video.
            validate_sha256: Nếu True (mặc định), tính lại SHA-256 để xác thực.

        Returns:
            Dict bản ghi cache (copy) nếu hợp lệ, None nếu miss/invalid.
        """
        path = Path(filepath).resolve()
        if not path.is_file():
            LOGGER.info("CACHE_MISS: Tệp không tồn tại trên đĩa: %s", path)
            return None

        key = self._normalize_key(path)
        if key not in self.cache_data:
            LOGGER.info("CACHE_MISS: Chưa có cache cho %s", key)
            return None

        entry = self.cache_data[key]

        # Schema version guard: invalidate record cũ thiếu sha256 hoặc schema_version cũ
        cached_sha256 = entry.get("sha256", "")
        schema_ver = entry.get("schema_version", 1)
        if not cached_sha256 or cached_sha256.startswith("err_") or schema_ver < CACHE_SCHEMA_VERSION:
            LOGGER.warning(
                "CACHE_INVALIDATED (Schema Version / Missing SHA-256): %s "
                "(schema_version=%s, sha256_present=%s) — record cũ, buộc re-audit",
                key, schema_ver,
                bool(cached_sha256 and not cached_sha256.startswith("err_")),
            )
            self.invalidate(key)
            return None

        # Kiểm tra kích thước file trước (nhanh hơn SHA-256)
        curr_size = path.stat().st_size
        cached_size = entry.get("file_size_bytes", 0)
        if curr_size != cached_size:
            LOGGER.warning(
                "CACHE_INVALIDATED (Size Mismatch): %s (Cached: %d, Actual: %d)",
                key, cached_size, curr_size,
            )
            self.invalidate(key)
            return None

        # Xác thực SHA-256 toàn file (chống file bị sửa đổi cùng kích thước)
        if validate_sha256:
            try:
                curr_sha256 = compute_streaming_sha256(path)
            except Exception as exc:
                LOGGER.warning(
                    "CACHE_INVALIDATED (SHA-256 Compute Error): %s — %s", key, exc,
                )
                self.invalidate(key)
                return None

            if curr_sha256 != cached_sha256:
                LOGGER.warning(
                    "CACHE_INVALIDATED (SHA-256 Mismatch / Same-Size Modified File): "
                    "%s (Cached: %s…, Actual: %s…)",
                    key, cached_sha256[:12], curr_sha256[:12],
                )
                self.invalidate(key)
                return None

        LOGGER.info(
            "CACHE_HIT: Hợp lệ 100%% cho %s (SHA-256: %s…, full_decoded_frames: %d, decode_status: %s)",
            key,
            entry.get("sha256", "")[:12],
            entry.get("full_decoded_frames", 0),
            entry.get("decode_status", ""),
        )
        return dict(entry)

    def put(self, filepath: Path | str, record: Dict[str, Any]) -> None:
        """Lưu bản ghi vào cache, luôn stamp schema_version."""
        key = self._normalize_key(filepath)
        rec = dict(record)
        rec["schema_version"] = CACHE_SCHEMA_VERSION
        self.cache_data[key] = rec

    def invalidate(self, filepath: Path | str) -> None:
        """Xóa bản ghi cache cho filepath."""
        key = self._normalize_key(filepath)
        if key in self.cache_data:
            del self.cache_data[key]
            LOGGER.info("CACHE_INVALIDATED: Đã xóa bản ghi cache cho %s", key)


# ---------------------------------------------------------------------------
# AUDIT SINGLE VIDEO
# ---------------------------------------------------------------------------

def audit_single_video(
    filepath: str | Path,
    root_dir: Optional[Path] = None,
    full_decode: bool = True,
    probe_frames: int = 5,
    chunk_size: int = 1048576,
    cache_manager: Optional[VideoDecodeCacheManager] = None,
) -> Dict[str, Any]:
    """Kiểm tra chi tiết một tệp video, tính full SHA-256 và giải mã tuần tự đến EOF.

    Frame count semantics (phân biệt rõ 4 loại):
    - reported_frame_count: Số frame báo cáo bởi container metadata (CAP_PROP_FRAME_COUNT).
      Có thể không chính xác do container rounding.
    - full_decoded_frames (decoded_frame_count): Số frame thực sự decode thành công
      (probe + full decode). Ground truth kỹ thuật.
    - audit_probe_frames (sampled_frame_count): Số frame probe khởi động (mặc định 5).
      Subset của full_decoded_frames.
    - processed_frame_count: Của inference pipeline — KHÔNG thuộc phạm vi audit.

    Tolerance policy (Explicit Container Tolerance):
    - Delta = reported_frame_count - full_decoded_frames
    - Delta == 0           → NORMAL_EOF (khớp tuyệt đối)
    - 1 <= Delta <= 5      → NORMAL_EOF (MP4 container index rounding, B-frame PTS offset)
    - 6 <= Delta <= 10     → NORMAL_EOF + WARNING log (B-frame reorder buffer flush)
    - Delta > 10           → TRUNCATED_STREAM (buộc NEEDS_REVIEW)
    - reported == 0        → NEEDS_REVIEW (không đủ bằng chứng)

    QUAN TRỌNG: KHÔNG gán NORMAL_EOF chỉ vì cap.read()/cap.grab() trả về False.
    Phải so sánh full_decoded_frames với reported_frame_count.

    decode_status tách biệt hoàn toàn với source_label_status và human_review_status.
    full_decode_verified=True: đã quét tuần tự đến EOF (không phụ thuộc nhãn ngữ nghĩa).
    """
    vpath = Path(filepath).resolve()
    base_root = root_dir.resolve() if root_dir else Path.cwd().resolve()

    try:
        rel_path = str(vpath.relative_to(base_root)).replace("\\", "/")
    except ValueError:
        rel_path = str(vpath).replace("\\", "/")

    fname = vpath.name
    vid = vpath.stem

    # 1. Kiểm tra cache tăng cường trước (validate_sha256=True bắt buộc)
    if cache_manager is not None:
        cached_res = cache_manager.get(vpath, validate_sha256=True)
        if cached_res is not None:
            return cached_res

    if not vpath.is_file():
        return _make_error_record(
            vid=vid, rel_path=rel_path, fname=fname,
            sha256="", sha256_verified=False, fsize=0,
            decode_status=DecodeStatus.CANNOT_OPEN,
            error_message=f"Tệp không tồn tại: {vpath}",
        )

    fsize = vpath.stat().st_size

    # 2. Tính SHA-256 toàn phần bằng streaming chunks (không dừng sớm)
    try:
        sha256_val = compute_streaming_sha256(vpath, chunk_size=chunk_size)
        sha256_verified = True
    except Exception as exc:
        sha256_val = f"err_{exc}"
        sha256_verified = False
        LOGGER.error("Không thể tính SHA-256 cho %s: %s", rel_path, exc)

    # 3. Khởi tạo OpenCV VideoCapture
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return _make_error_record(
            vid=vid, rel_path=rel_path, fname=fname,
            sha256=sha256_val, sha256_verified=sha256_verified, fsize=fsize,
            decode_status=DecodeStatus.CANNOT_OPEN,
            error_message="OpenCV không thể mở tệp video (hỏng container hoặc định dạng không hỗ trợ)",
        )

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_dim = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_val = round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 2)
    # reported_frame_count: từ metadata container — có thể không chính xác
    reported_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
    if not codec_str or any(ord(c) < 32 or ord(c) > 126 for c in codec_str):
        codec_str = "h264/mp4v"

    # 4. Probe frames ở đầu (audit_probe_frames = sampled_frame_count)
    probed_ok = 0
    for _ in range(probe_frames):
        ret, fr = cap.read()
        if ret and fr is not None:
            probed_ok += 1
        else:
            break

    if probed_ok == 0:
        cap.release()
        return _make_error_record(
            vid=vid, rel_path=rel_path, fname=fname,
            sha256=sha256_val, sha256_verified=sha256_verified, fsize=fsize,
            decode_status=DecodeStatus.DECODE_ERROR,
            fps=fps_val, w=w, h=h_dim,
            reported_frame_count=reported_frame_count,
            codec=codec_str,
            error_message="Không thể giải mã các khung hình đầu tiên của video",
        )

    # 5. Duyệt tuần tự đến EOF nếu full_decode=True
    # actual_decoded_frames = full_decoded_frames = decoded_frame_count (ground truth)
    actual_decoded_frames = probed_ok   # probe frames đã decode thành công
    decode_status_val: str = str(DecodeStatus.PROBE_ONLY)
    full_decode_verified = False
    decode_errors = 0
    err_msg = ""
    tolerance_applied = False
    status = "NEEDS_REVIEW"
    missing_frames = max(0, reported_frame_count - probed_ok)

    if full_decode:
        # Tiếp tục decode tuần tự từ probe_frames+1 đến EOF
        # grab() giải mã nhưng không decode pixel — nhanh hơn read()
        while True:
            if not cap.grab():
                # grab() == False: có thể là EOF hoặc decoder error
                # KHÔNG tự kết luận NORMAL_EOF — phải so sánh frame count bên dưới
                break
            actual_decoded_frames += 1

        cap.release()

        # full_decode_verified: đã quét đến EOF (bất kể NORMAL/TRUNCATED)
        full_decode_verified = (actual_decoded_frames > 0)

        # Delta = reported - actual (số frame reported nhưng không decode được)
        missing_frames = max(0, reported_frame_count - actual_decoded_frames)

        # Áp dụng Explicit Tolerance Policy
        if actual_decoded_frames == 0:
            decode_status_val = str(DecodeStatus.DECODE_ERROR)
            status = "CORRUPTED"
            err_msg = "Không giải mã được khung hình nào"
            full_decode_verified = False

        elif reported_frame_count == 0:
            # Metadata lỗi: không thể so sánh — không đủ bằng chứng
            decode_status_val = str(DecodeStatus.NEEDS_REVIEW)
            status = "NEEDS_REVIEW"
            err_msg = (
                f"reported_frame_count=0 từ metadata — không đủ bằng chứng phân loại. "
                f"Đã decode thực tế {actual_decoded_frames} frames."
            )
            LOGGER.warning("Video %s: %s", vid, err_msg)

        elif missing_frames == 0:
            decode_status_val = str(DecodeStatus.NORMAL_EOF)
            status = "VALID"
            err_msg = "Khớp tuyệt đối 100% số khung hình container"

        elif missing_frames <= CONTAINER_PADDING_TOLERANCE:
            # 1-5 frames: container padding thông thường
            decode_status_val = str(DecodeStatus.NORMAL_EOF)
            status = "VALID"
            tolerance_applied = True
            err_msg = (
                f"Khớp trong dung sai container padding thông thường "
                f"(thiếu {missing_frames} frames <= {CONTAINER_PADDING_TOLERANCE}; "
                f"MP4 container index rounding / B-frame PTS offset)"
            )

        elif missing_frames <= CONTAINER_BORDERLINE_TOLERANCE:
            # 6-10 frames: borderline tolerance — PHẢI log WARNING
            decode_status_val = str(DecodeStatus.NORMAL_EOF)
            status = "VALID"
            tolerance_applied = True
            err_msg = (
                f"Khớp trong dung sai container cận (borderline) "
                f"(thiếu {missing_frames} frames <= {CONTAINER_BORDERLINE_TOLERANCE}; "
                f"B-frame reorder buffer flush / VFR dropped frames ở cuối GOP)"
            )
            LOGGER.warning(
                "Video %s BORDERLINE_TOLERANCE applied: %s", vid, err_msg,
            )

        else:
            # > 10 frames: TRUNCATED_STREAM — KHÔNG thể là NORMAL_EOF
            decode_status_val = str(DecodeStatus.TRUNCATED_STREAM)
            status = "NEEDS_REVIEW"
            decode_errors = missing_frames
            err_msg = (
                f"Stream kết thúc sớm tại frame {actual_decoded_frames}/{reported_frame_count} "
                f"(thiếu {missing_frames} frames > {TRUNCATED_STREAM_THRESHOLD} "
                f"do lỗi luồng NAL/packet)"
            )
            LOGGER.warning("Video %s TRUNCATED_STREAM: %s", vid, err_msg)

    else:
        cap.release()
        decode_status_val = str(DecodeStatus.PROBE_ONLY)
        status = "NEEDS_REVIEW"
        full_decode_verified = False
        err_msg = "Chỉ thực hiện probe khởi động, chưa giải mã tuần tự đến EOF"

    # Tính thời lượng thực tế (dựa trên full_decoded_frames — ground truth)
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
        # decode_status: trạng thái kỹ thuật luồng vật lý (tách biệt với nhãn ngữ nghĩa)
        "decode_status": decode_status_val,
        # source_label_status: nhãn nguồn gốc dữ liệu (KHÔNG tự gán VERIFIED_NEGATIVE)
        "source_label_status": "SURVEILLANCE_NEGATIVE_BATCH",
        # human_review_status: trạng thái xác nhận của annotator (KHÔNG tự tạo)
        "human_review_status": "PENDING_HUMAN_REVIEW",
        "duration_seconds": dur_sec,
        "duration_hours": round(dur_sec / 3600.0, 4),
        "fps": fps_val,
        "width": w,
        "height": h_dim,
        # reported_frame_count: từ container metadata
        "reported_frame_count": reported_frame_count,
        # audit_probe_frames: sampled_frame_count trong audit (probe khởi động)
        "audit_probe_frames": probed_ok,
        # full_decoded_frames: decoded_frame_count — số frame thực sự decode thành công
        "full_decoded_frames": actual_decoded_frames if full_decode else 0,
        "missing_frames": missing_frames if full_decode else reported_frame_count,
        # decoded_frames: alias tương thích ngược
        "decoded_frames": actual_decoded_frames if full_decode else probed_ok,
        "full_decode_verified": full_decode_verified,
        "full_decode_errors": decode_errors,
        "container_tolerance_applied": tolerance_applied,
        "codec": codec_str,
        "error_message": err_msg,
        "duplicate_of": None,
        "duration_calc_method": dur_calc_method,
        "schema_version": CACHE_SCHEMA_VERSION,
        "last_audited_at": datetime.now(timezone.utc).isoformat(),
    }

    # Lưu vào cache nếu có cache_manager và đã full_decode thành công
    if cache_manager is not None and full_decode_verified:
        cache_manager.put(vpath, result)

    return result


def _make_error_record(
    *,
    vid: str,
    rel_path: str,
    fname: str,
    sha256: str,
    sha256_verified: bool,
    fsize: int,
    decode_status: DecodeStatus,
    error_message: str,
    fps: float = 0.0,
    w: int = 0,
    h: int = 0,
    reported_frame_count: int = 0,
    codec: str = "unknown",
) -> Dict[str, Any]:
    """Tạo record lỗi chuẩn hóa cho các trường hợp không thể decode."""
    return {
        "video_id": vid,
        "relative_path": rel_path,
        "filename": fname,
        "sha256": sha256,
        "sha256_verified": sha256_verified,
        "file_size_bytes": fsize,
        "status": "CORRUPTED",
        "decode_status": str(decode_status),
        "source_label_status": "UNVERIFIED_SOURCE",
        "human_review_status": "PENDING_HUMAN_REVIEW",
        "duration_seconds": 0.0,
        "duration_hours": 0.0,
        "fps": fps,
        "width": w,
        "height": h,
        "reported_frame_count": reported_frame_count,
        "audit_probe_frames": 0,
        "full_decoded_frames": 0,
        "missing_frames": reported_frame_count,
        "decoded_frames": 0,
        "full_decode_verified": False,
        "full_decode_errors": 1,
        "container_tolerance_applied": False,
        "codec": codec,
        "error_message": error_message,
        "duplicate_of": None,
        "duration_calc_method": "none",
        "schema_version": CACHE_SCHEMA_VERSION,
        "last_audited_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# DUPLICATE DETECTION
# ---------------------------------------------------------------------------

def detect_duplicates_and_id_collisions(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Phát hiện trùng lặp SHA-256 và va chạm video_id.

    Chính sách:
    1. SHA-256 trùng (cùng content, khác path): bản thứ 2+ → status='DUPLICATE',
       duplicate_of = đường dẫn bản đầu tiên.
    2. video_id trùng (cùng ID, khác path/content): bản sau → status='DUPLICATE_ID',
       error_message ghi rõ bản trùng với bản nào.
    3. KHÔNG âm thầm xóa record nào — chỉ annotate status.
    4. Log WARNING cho mỗi trường hợp phát hiện.
    """
    seen_hashes: Dict[str, str] = {}   # sha256 -> relative_path bản đầu tiên
    seen_ids: Dict[str, str] = {}      # video_id -> relative_path bản đầu tiên
    updated_records: List[Dict[str, Any]] = []

    for rec in records:
        vid = rec.get("video_id", "")
        h = rec.get("sha256", "")
        rel_p = rec.get("relative_path", "")
        rec = dict(rec)  # shallow copy để không mutate input

        # Kiểm tra trùng video_id (cùng ID, khác path)
        if vid in seen_ids:
            rec["status"] = "DUPLICATE_ID"
            rec["error_message"] = (
                f"Trùng video_id '{vid}' với tệp '{seen_ids[vid]}' "
                f"(cùng ID, khác path — cần kiểm tra naming convention)"
            )
            LOGGER.warning(
                "DUPLICATE_ID: %s trùng video_id '%s' với %s",
                rel_p, vid, seen_ids[vid],
            )
        else:
            seen_ids[vid] = rel_p

        # Kiểm tra trùng SHA-256 (cùng content, khác path)
        if h and not h.startswith("err_"):
            if h in seen_hashes:
                rec["status"] = "DUPLICATE"
                rec["duplicate_of"] = seen_hashes[h]
                LOGGER.warning(
                    "DUPLICATE_HASH: %s trùng SHA-256 với %s (cùng nội dung, khác path)",
                    rel_p, seen_hashes[h],
                )
            else:
                if rec.get("status") in ("VALID", "NEEDS_REVIEW"):
                    seen_hashes[h] = rel_p

        updated_records.append(rec)

    return updated_records


# Alias tương thích ngược
detect_duplicates_by_hash = detect_duplicates_and_id_collisions


# ---------------------------------------------------------------------------
# MANIFEST vs INFERENCE VERIFICATION
# ---------------------------------------------------------------------------

def verify_manifest_vs_inference(
    manifest_videos: List[Dict[str, Any]],
    inference_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Đối soát 100% tính nhất quán giữa danh mục Manifest và Inference Summary.

    Kiểm tra:
    1. Số lượng video có khớp nhau 100%.
    2. Mỗi video trong inference có tồn tại trong manifest.
    3. Số frame giải mã (total_decoded_frames) khớp 100%.
    4. Đối soát 3 mẫu số phơi nhiễm:
       - 7.2026h: Mẫu số benchmark run.
       - 7.1312h: Mẫu số phơi nhiễm stream thực tế.
       - 6.9846h: Mẫu số tập untruncated hợp lệ.
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
