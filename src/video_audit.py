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
   - KHÔNG suy luận NORMAL_EOF chỉ từ read()/grab() trả về False hay metadata.
   - NORMAL_EOF yêu cầu FFmpeg decode hết stream, exit code 0, progress=end,
     stderr sạch, và frame count khớp chính xác giữa FFmpeg, OpenCV và container.
   - Bất kỳ chênh lệch frame count nào cũng NEEDS_REVIEW; chưa có dung sai
     thực nghiệm được thiết lập cho codec/container/PTS/DTS trong repository.
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
import math
import os
import shutil
import subprocess
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
    """FFmpeg cleanly decoded the full stream; FFmpeg, OpenCV, and container counts agree."""

    TRUNCATED_STREAM = "TRUNCATED_STREAM"
    """An explicit decoder error occurred after at least one frame was decoded."""

    DECODE_ERROR = "DECODE_ERROR"
    """No frame could be decoded from the input stream."""

    CANNOT_OPEN = "CANNOT_OPEN"
    """OpenCV VideoCapture.isOpened() == False."""

    PROBE_ONLY = "PROBE_ONLY"
    """Only initial frames were probed; no full-stream EOF check was run."""

    NEEDS_REVIEW = "NEEDS_REVIEW"
    """Không đủ bằng chứng phân loại dứt khoát (ví dụ: reported_frame_count == 0)."""

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# TOLERANCE POLICY
# ---------------------------------------------------------------------------

CONTAINER_PADDING_TOLERANCE = 0
"""No automatic tolerance is evidence-backed in this repository. The historical
5-frame / 0.25s values are disabled; count discrepancies require review."""

CONTAINER_BORDERLINE_TOLERANCE = 0
"""No borderline tolerance is evidence-backed; count discrepancies require review."""

TRUNCATED_STREAM_THRESHOLD = 0
"""A count delta alone is not proof of truncation; explicit decoder error evidence is required."""


def probe_video_with_ffprobe(filepath: str | Path) -> Optional[Dict[str, Any]]:
    """Thử sử dụng ffprobe (nếu có trên hệ thống) để thẩm định luồng stream và mã kết thúc.

    Returns:
        Dict metadata ffprobe nếu ffprobe tồn tại và chạy thành công, None nếu không có ffprobe.
    """
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return None
    try:
        cmd = [
            ffprobe_bin,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,duration,codec_name,r_frame_rate",
            "-show_entries", "format=duration,nb_streams",
            "-of", "json",
            str(filepath),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        if res.returncode == 0:
            return json.loads(res.stdout)
    except Exception as exc:
        LOGGER.debug("FFprobe không khả dụng hoặc lỗi cho %s: %s", filepath, exc)
    return None


# Strict full-stream verification.
def run_external_stream_decode_check(filepath: str | Path, timeout_sec: int = 15) -> Dict[str, Any]:
    """Decode every video frame with FFmpeg and require a clean end marker."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return {"tool_available": False, "tool_name": None, "decode_capable": False,
                "exit_code": None, "stderr": "", "is_clean_decode": None,
                "error_detected": False, "error_message": "FFmpeg unavailable; EOF unverified",
                "nb_read_frames": None, "progress_end": False}
    cmd = [ffmpeg_bin, "-nostdin", "-nostats", "-v", "error", "-xerror", "-i", str(filepath),
           "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "null", "-", "-progress", "pipe:1"]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, timeout=timeout_sec)
        progress: Dict[str, str] = {}
        for line in res.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                progress[key.strip()] = value.strip()
        frame_text = progress.get("frame", "")
        count = int(frame_text) if frame_text.isdigit() else None
        err = res.stderr.strip()
        progress_end = progress.get("progress") == "end"
        error = res.returncode != 0 or bool(err)
        clean = res.returncode == 0 and not error and progress_end and count is not None
        return {"tool_available": True, "tool_name": "ffmpeg", "decode_capable": True,
                "exit_code": res.returncode, "stderr": err, "is_clean_decode": clean,
                "error_detected": error, "error_message": err if error else "",
                "nb_read_frames": count, "progress_end": progress_end}
    except subprocess.TimeoutExpired:
        return {"tool_available": True, "tool_name": "ffmpeg", "decode_capable": True,
                "exit_code": -1, "stderr": "Command timed out", "is_clean_decode": False,
                "error_detected": True, "error_message": f"FFmpeg timed out after {timeout_sec}s",
                "nb_read_frames": None, "progress_end": False}
    except Exception as exc:
        return {"tool_available": True, "tool_name": "ffmpeg", "decode_capable": True,
                "exit_code": -1, "stderr": str(exc), "is_clean_decode": False,
                "error_detected": True, "error_message": str(exc),
                "nb_read_frames": None, "progress_end": False}


def classify_decode_termination(
    actual_decoded_frames: int,
    reported_frame_count: int,
    fps: float,
    filepath: Optional[Path | str] = None,
    ffprobe_meta: Optional[Dict[str, Any]] = None,
    external_check: Optional[Dict[str, Any]] = None,
) -> Tuple[DecodeStatus, str, bool, int, str, bool]:
    """Classify only from full-stream evidence; never infer EOF from OpenCV stop."""
    ext = external_check or {}
    ffmpeg_ready = bool(ext.get("tool_available") and ext.get("decode_capable"))
    ext_count = ext.get("nb_read_frames")
    if ext.get("error_detected") or (ffmpeg_ready and ext.get("exit_code") != 0):
        missing = max(0, reported_frame_count - actual_decoded_frames) if reported_frame_count > 0 else 0
        state = DecodeStatus.TRUNCATED_STREAM if actual_decoded_frames > 0 else DecodeStatus.DECODE_ERROR
        return state, "NEEDS_REVIEW", False, max(1, missing), str(ext.get("error_message") or ext.get("stderr") or "FFmpeg decode error"), False
    if actual_decoded_frames <= 0:
        return DecodeStatus.DECODE_ERROR, "CORRUPTED", False, max(1, reported_frame_count), "No frames decoded", False
    if not math.isfinite(fps) or fps <= 0 or fps > 240 or reported_frame_count <= 0:
        return DecodeStatus.NEEDS_REVIEW, "NEEDS_REVIEW", False, 0, "Invalid FPS or missing/zero reported frame count", False
    if not ffmpeg_ready:
        missing = max(0, reported_frame_count - actual_decoded_frames)
        return DecodeStatus.NEEDS_REVIEW, "NEEDS_REVIEW", False, missing, "No FFmpeg full-stream decode evidence; OpenCV stop is inconclusive", False
    if not ext.get("is_clean_decode") or not ext.get("progress_end") or ext_count is None:
        return DecodeStatus.NEEDS_REVIEW, "NEEDS_REVIEW", False, 0, "FFmpeg did not establish clean progress=end and decoded-frame count", False
    if actual_decoded_frames != reported_frame_count or ext_count != reported_frame_count or ext_count != actual_decoded_frames:
        return DecodeStatus.NEEDS_REVIEW, "NEEDS_REVIEW", False, abs(reported_frame_count - actual_decoded_frames), "OpenCV, FFmpeg, and metadata frame counts disagree; no tolerance is validated", False
    return DecodeStatus.NORMAL_EOF, "VALID", False, 0, "FFmpeg clean exit at progress=end; OpenCV, FFmpeg, and metadata frame counts agree", True


# ---------------------------------------------------------------------------
# CACHE SCHEMA VERSION
# ---------------------------------------------------------------------------
CACHE_SCHEMA_VERSION = 3
"""Schema version of strict full-decode cache. Older cached VALID results are
invalidated because previous versions could accept unverified EOF/tolerances.
Record thiếu 'sha256' hoặc schema_version < 3
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

    No frame-count tolerance is applied. NORMAL_EOF requires clean FFmpeg full
    decode evidence and exact frame-count agreement with OpenCV and container.
    Any uncorroborated termination or count mismatch is NEEDS_REVIEW.

    decode_status tách biệt hoàn toàn với source_label_status và human_review_status.
    full_decode_verified=True only when the strict EOF evidence is complete.
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
    fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    # Keep records JSON-safe while still forcing review for invalid metadata.
    fps_val = round(fps_raw, 2) if math.isfinite(fps_raw) else 0.0
    # reported_frame_count: từ metadata container — có thể không chính xác
    reported_raw = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    reported_frame_count = int(reported_raw) if math.isfinite(reported_raw) else 0

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

    if probed_ok == 0 and reported_frame_count <= 0:
        cap.release()
        return _make_error_record(
            vid=vid, rel_path=rel_path, fname=fname,
            sha256=sha256_val, sha256_verified=sha256_verified, fsize=fsize,
            decode_status=DecodeStatus.NEEDS_REVIEW,
            fps=fps_val, w=w, h=h_dim, reported_frame_count=reported_frame_count,
            codec=codec_str,
            error_message="No frames and no positive container frame count; EOF and decodability are unverified",
        )

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
        # Continue full sequential pixel decoding. A failed read alone is not EOF
        # evidence; the independent FFmpeg pass below must also complete cleanly.
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            actual_decoded_frames += 1

        cap.release()

        # Thử nghiệm với ffprobe / ffmpeg nếu có
        external_check = run_external_stream_decode_check(vpath)
        ffprobe_meta = probe_video_with_ffprobe(vpath)

        # Phân loại trạng thái kết thúc giải mã (Stream Termination Classification)
        (
            decode_status_obj,
            status,
            tolerance_applied,
            decode_errors,
            err_msg,
            full_decode_verified,
        ) = classify_decode_termination(
            actual_decoded_frames=actual_decoded_frames,
            reported_frame_count=reported_frame_count,
            fps=fps_val,
            filepath=rel_path,
            ffprobe_meta=ffprobe_meta,
            external_check=external_check,
        )
        decode_status_val = str(decode_status_obj)
        missing_frames = max(0, reported_frame_count - actual_decoded_frames)

    else:
        cap.release()
        decode_status_val = str(DecodeStatus.PROBE_ONLY)
        status = "NEEDS_REVIEW"
        full_decode_verified = False
        err_msg = "Chỉ thực hiện probe khởi động, chưa giải mã tuần tự đến EOF"
        ffprobe_meta = None
        external_check = None

    # Tính thời lượng thực tế (dựa trên full_decoded_frames — ground truth)
    if fps_val > 0 and actual_decoded_frames > 0:
        dur_sec = round(actual_decoded_frames / fps_val, 2)
        dur_calc_method = "actual_decoded_frames / native_fps"
    else:
        dur_sec = 0.0
        dur_calc_method = "none"

    if external_check and external_check.get("tool_available") and external_check.get("is_clean_decode"):
        stream_backend = f"external_{external_check['tool_name']}_full_decode"
    elif ffprobe_meta:
        stream_backend = "ffprobe_stream_metadata"
    else:
        stream_backend = "opencv_pts_frame_analysis (ffprobe_not_found_fallback)"

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
        "stream_verification_backend": stream_backend,
        "external_tool_check": external_check,
        "ffprobe_metadata": ffprobe_meta,
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
        "stream_verification_backend": "none (open_failed)",
        "external_tool_check": None,
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
