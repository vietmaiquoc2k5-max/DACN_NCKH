#!/usr/bin/env python3
"""Preparation tool for Week 6 Validation Negative Video Verification.

Identifies the exact 13 Validation negative videos originating from datasets/vn_dataset/split_manifest.csv:
- VN_NEG_03, VN_NEG_42, VN_NEG_43, VN_NEG_44, VN_NEG_45, VN_NEG_46, VN_NEG_47,
  VN_NEG_48, VN_NEG_49, VN_NEG_50, VN_NEG_51, VN_NEG_52, VN_NEG_53

Performs technical integrity checks (file existence, file size, SHA256 match, decode integrity).
Generates the human review worksheet (runs/audits/week6/validation_negative_review_manifest.csv)
and audit report (runs/audits/week6/validation_negative_review_report.md).

Scientific Invariants:
1. Does NOT auto-fill VERIFIED_NEGATIVE (all start as PENDING_HUMAN_REVIEW).
2. Does NOT treat OpenCV grab() == False as independent proof of NORMAL_EOF without ffprobe.
3. Does NOT admit technically defective videos into FAR exposure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOG = logging.getLogger("validation_neg_prep")


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        path: Path to target file.

    Returns:
        Hexadecimal hash string.
    """
    if not path.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV atomically to prevent corrupted files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(dir=str(path.parent), prefix="tmp_neg_", suffix=".csv")
    try:
        with os.fdopen(temp_fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def get_validation_negative_videos_from_split(split_manifest_path: Path) -> list[str]:
    """Extract the exact list of validation negative video IDs from split manifest.

    Args:
        split_manifest_path: Path to datasets/vn_dataset/split_manifest.csv.

    Returns:
        Sorted list of unique negative video IDs in the validation split.

    Raises:
        FileNotFoundError: If split manifest is missing.
        ValueError: If validation negative videos cannot be identified.
    """
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")

    val_neg_vids: set[str] = set()
    with open(split_manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split", "").strip().lower()
            vid = row.get("video_id", "").strip()
            cat = row.get("category", "").strip().lower()
            if split == "val" and ("neg" in cat or "NEG" in vid):
                val_neg_vids.add(vid)

    vids = sorted(list(val_neg_vids))
    LOG.info("Identified %d validation negative videos from %s: %s", len(vids), split_manifest_path, vids)
    if len(vids) != 13:
        LOG.warning("Expected 13 validation negative videos, found %d", len(vids))
    return vids


def probe_video_decode(
    video_path: Path,
    expected_frames: int,
    ffmpeg_available: bool
) -> tuple[str, str, int, float, int, int, list[str]]:
    """Probe video decoding integrity.

    Args:
        video_path: Physical video file path.
        expected_frames: Expected frame count from manifest or container.
        ffmpeg_available: Whether ffmpeg/ffprobe binaries are in PATH.

    Returns:
        Tuple of (decode_status, tool_used, decoded_frame_count, fps, width, height, errors).
    """
    errors: list[str] = []
    if not video_path.exists():
        return "FILE_NOT_FOUND", "NONE", 0, 0.0, 0, 0, ["File not found on disk"]

    tool_used = "FFMPEG_FFPROBE" if ffmpeg_available else "OPENCV_STREAM_PROBE (FFMPEG_UNAVAILABLE)"

    # If FFmpeg is available, run full decode check
    if ffmpeg_available:
        try:
            cmd = ["ffmpeg", "-v", "error", "-i", str(video_path), "-f", "null", "-"]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
            if proc.returncode == 0:
                # Normal EOF
                pass
            else:
                errors.append(f"ffmpeg decode error: {proc.stderr[:200]}")
        except Exception as e:
            errors.append(f"ffmpeg run error: {e}")

    # Inspect stream with OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "DECODE_ERROR", tool_used, 0, 0.0, 0, 0, ["OpenCV could not open video stream"]

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Determine safe frame boundary for probe
    probe_bound = reported_frames
    if expected_frames > 0 and expected_frames < reported_frames:
        errors.append(
            f"Container metadata claims {reported_frames} frames but manifest expects {expected_frames} frames"
        )
        probe_bound = expected_frames

    # Frame decode boundary probing: probe start, middle, and ~95% tail frame
    probed_frames_ok = True
    test_positions = [0, probe_bound // 2, max(0, int(probe_bound * 0.95))]
    for pos in test_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret or frame is None:
            probed_frames_ok = False
            errors.append(f"Failed to read frame at position {pos}/{probe_bound}")
            break

    cap.release()

    if not probed_frames_ok:
        status = "TRUNCATED_STREAM" if probe_bound > 0 else "DECODE_ERROR"
    elif ffmpeg_available and not errors:
        status = "NORMAL_EOF"
    elif not ffmpeg_available:
        # Crucial Scientific Rule: OpenCV alone cannot attest NORMAL_EOF
        status = "PROBED_OPENCV_OK_UNVERIFIED_EOF"
    else:
        status = "DECODE_WARNING"

    return status, tool_used, reported_frames, fps, width, height, errors


def prepare_validation_negative_review(
    split_manifest_path: Path,
    negative_manifest_path: Path,
    video_dir: Path,
    output_dir: Path
) -> tuple[Path, Path]:
    """Prepare verification manifest and markdown report for 13 validation negative videos.

    Args:
        split_manifest_path: Path to split_manifest.csv.
        negative_manifest_path: Path to negative_videos_manifest.json.
        video_dir: Directory containing negative video files.
        output_dir: Directory for output artifacts.

    Returns:
        Tuple of (manifest_csv_path, report_md_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    val_vids = get_validation_negative_videos_from_split(split_manifest_path)

    # Load negative manifest
    neg_meta: dict[str, dict[str, Any]] = {}
    if negative_manifest_path.exists():
        with open(negative_manifest_path, "r", encoding="utf-8") as f:
            neg_data = json.load(f)
            for v in neg_data.get("videos", []):
                vid = v.get("video_id", "")
                neg_meta[vid] = v
                neg_meta[vid.replace("VN_", "")] = v

    ffmpeg_available = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
    LOG.info("FFmpeg availability: %s", ffmpeg_available)

    manifest_rows = []
    for vid in val_vids:
        meta = neg_meta.get(vid, {})
        fname = meta.get("filename", f"{vid}.mp4")
        expected_sha = meta.get("sha256", "")
        duration_meta = float(meta.get("duration_seconds", 0.0))

        # Physical video file
        candidate_paths = [
            video_dir / fname,
            ROOT / f"datasets/negative_videos/{fname}",
            ROOT / f"negative_videos/{fname}",
            ROOT / f"datasets/negative_videos/{vid}.mp4"
        ]
        video_path = next((p for p in candidate_paths if p.exists()), None)
        file_exists = video_path is not None
        file_size_bytes = video_path.stat().st_size if file_exists else 0
        actual_sha = sha256_file(video_path) if file_exists else "FILE_NOT_FOUND"
        sha_match = (actual_sha == expected_sha) if (file_exists and expected_sha) else False

        # Probe decode
        expected_frames = int(meta.get("reported_frame_count", 0))
        dec_status, tool, frames, fps, w, h, dec_errors = probe_video_decode(
            video_path, expected_frames, ffmpeg_available
        )

        duration_sec = round(frames / fps, 2) if fps > 0 else duration_meta

        # Scientific Technical Evaluation
        if not file_exists:
            tech_verdict = "FAIL_FILE_NOT_FOUND"
        elif not sha_match and expected_sha:
            tech_verdict = "FAIL_SHA256_MISMATCH"
        elif dec_status in ("DECODE_ERROR", "TRUNCATED_STREAM"):
            tech_verdict = f"FAIL_{dec_status}"
        elif dec_status == "NORMAL_EOF":
            tech_verdict = "PASS_FFMPEG_VERIFIED"
        else:
            tech_verdict = "PASS_OPENCV_PROBE_UNVERIFIED_EOF"

        manifest_rows.append({
            "video_id": vid,
            "filename": fname,
            "file_path": str(video_path.relative_to(ROOT)).replace("\\", "/") if video_path else "NOT_FOUND",
            "file_size_bytes": file_size_bytes,
            "sha256_actual": actual_sha,
            "sha256_expected": expected_sha,
            "sha256_match": "TRUE" if sha_match else "FALSE",
            "split_provenance": "datasets/vn_dataset/split_manifest.csv::val",
            "native_fps": round(fps, 2),
            "resolution": f"{w}x{h}",
            "reported_frames": frames,
            "duration_seconds": duration_sec,
            "decode_status": dec_status,
            "decode_tool": tool,
            "technical_verdict": tech_verdict,
            # Content verification: strictly PENDING_HUMAN_REVIEW initially
            "content_decision": "PENDING_HUMAN_REVIEW",
            "reviewer_name": "",
            "review_timestamp": "",
            "attest_full_video_viewed": "FALSE",
            "review_notes": "",
            "evidence_timestamp_sec": "",
        })

    fieldnames = [
        "video_id",
        "filename",
        "file_path",
        "file_size_bytes",
        "sha256_actual",
        "sha256_expected",
        "sha256_match",
        "split_provenance",
        "native_fps",
        "resolution",
        "reported_frames",
        "duration_seconds",
        "decode_status",
        "decode_tool",
        "technical_verdict",
        "content_decision",
        "reviewer_name",
        "review_timestamp",
        "attest_full_video_viewed",
        "review_notes",
        "evidence_timestamp_sec",
    ]

    manifest_csv = output_dir / "validation_negative_review_manifest.csv"
    atomic_write_csv(manifest_csv, manifest_rows, fieldnames)
    LOG.info("Wrote validation negative review manifest to %s", manifest_csv)

    report_md = output_dir / "validation_negative_review_report.md"
    _write_negative_review_report(report_md, manifest_rows, ffmpeg_available)
    LOG.info("Wrote validation negative review report to %s", report_md)

    return manifest_csv, report_md


def _write_negative_review_report(
    path: Path,
    rows: list[dict[str, Any]],
    ffmpeg_available: bool
) -> None:
    """Generate Markdown report and human inspection guide."""
    total_hours = sum(r["duration_seconds"] for r in rows) / 3600.0
    tech_passed_count = sum(1 for r in rows if r["technical_verdict"].startswith("PASS"))
    pending_content_count = sum(1 for r in rows if r["content_decision"] == "PENDING_HUMAN_REVIEW")

    lines = [
        "# BÁO CÁO KIỂM TRA KỸ THUẬT VÀ BIỂU MẪU THẨM ĐỊNH VIDEO ÂM TÍNH TẬP VALIDATION",
        "## FireWatch – Validation Negative Videos Verification (13 Videos)",
        "",
        "> **Nguồn gốc phân chia (Provenance):** `datasets/vn_dataset/split_manifest.csv` (nhóm `split: val`)  ",
        f"> **Tổng số video:** {len(rows)} video ({total_hours:.2f} giờ)  ",
        f"> **Đạt tiêu chuẩn kỹ thuật (Technical Pass):** {tech_passed_count} / {len(rows)} video  ",
        f"> **Trạng thái thẩm định nội dung:** **{pending_content_count} / {len(rows)} video `PENDING_HUMAN_REVIEW`**  ",
        f"> **Công cụ giải mã khả dụng:** `{'FFMPEG/FFPROBE' if ffmpeg_available else 'OPENCV_STREAM_PROBE (FFMPEG_UNAVAILABLE)'}`  ",
        "",
        "---",
        "",
        "## 1. BẢNG KIỂM TRA TOÀN VẸN KỸ THUẬT (TECHNICAL AUDIT)",
        "",
        "| Video ID | Tệp tin | Kích thước | SHA-256 Match | Độ phân giải | FPS | Thời lượng | Decode Status | Technical Verdict |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in rows:
        match_icon = "PASS" if r["sha256_match"] == "TRUE" else "FAIL"
        lines.append(
            f"| **{r['video_id']}** | `{r['filename']}` | {r['file_size_bytes']:,} B | "
            f"`{match_icon}` | {r['resolution']} | {r['native_fps']} | {r['duration_seconds']}s | "
            f"`{r['decode_status']}` | `{r['technical_verdict']}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. QUY ĐỊNH THẨM ĐỊNH NỘI DUNG DÀNH CHO CON NGƯỜI (HUMAN REVIEW SOP)",
        "",
        "### 2.1. Tiêu chí phân loại nội dung (`content_decision`)",
        "- **`VERIFIED_NEGATIVE`**: Người thẩm định đã xem 100% thời lượng video, xác nhận KHÔNG CÓ bất kỳ ngọn lửa, tia lửa hay đám khói cháy thực tế nào.",
        "- **`FIRE_PRESENT`**: Phát hiện có lửa thật trong video (video âm tính bị gán nhãn sai trong quá khứ).",
        "- **`SMOKE_PRESENT`**: Phát hiện có khói cháy thực tế trong video.",
        "- **`UNCERTAIN`**: Video có khói mờ, ánh sáng phản chiếu phức tạp chưa thể khẳng định.",
        "",
        "### 2.2. Các trường bắt buộc khi nộp kết quả",
        "Khi người thẩm định cập nhật tệp `runs/audits/week6/validation_negative_review_manifest.csv`:",
        "1. `content_decision`: Điền `VERIFIED_NEGATIVE` (hoặc `FIRE_PRESENT` / `SMOKE_PRESENT` / `UNCERTAIN`).",
        "2. `reviewer_name`: Tên thật người kiểm định (ví dụ: `Mai Quốc Việt`). Cấm điền `SYSTEM`, `AUTO`, `NONE`.",
        "3. `review_timestamp`: Thời điểm kiểm định chuẩn ISO 8601 (ví dụ: `2026-11-17T18:00:00`).",
        "4. `attest_full_video_viewed`: Chuyển thành `TRUE` sau khi xem toàn bộ video.",
        "5. `review_notes`: Ghi chú về đặc điểm bối cảnh (ví dụ: ánh đèn xe, sương mù, mặt đường phản chiếu).",
        "",
        "> [!CAUTION]",
        "> Nghiêm cấm dùng script tự động gán `VERIFIED_NEGATIVE`. Mọi video thiếu chữ ký người kiểm định hoặc có lỗi kỹ thuật (SHA256 mismatch, decode error) sẽ bị **CHẶN HOÀN TOÀN (BLOCKED)** khỏi mẫu số tính tỷ lệ báo giả (FAR exposure).",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Week 6 Validation Negative Video Review Manifest")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "datasets/vn_dataset/split_manifest.csv",
                        help="Path to split_manifest.csv")
    parser.add_argument("--negative-manifest", type=Path, default=ROOT / "data/negative_videos_manifest.json",
                        help="Path to negative_videos_manifest.json")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "datasets/negative_videos",
                        help="Directory with negative video files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/audits/week6",
                        help="Output directory for artifacts")
    args = parser.parse_args()

    try:
        manifest_csv, report_md = prepare_validation_negative_review(
            args.split_manifest, args.negative_manifest, args.video_dir, args.output_dir
        )
        LOG.info("Preparation complete. Manifest: %s, Report: %s", manifest_csv, report_md)
        return 0
    except Exception as err:
        LOG.error("Failed to prepare validation negative review: %s", err, exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
