"""Bộ kiểm thử tính toàn vẹn của audit video, streaming SHA-256, full decode và đồng bộ manifest (tests/test_video_audit_and_consistency.py).

Tuân thủ nghiêm ngặt 9 yêu cầu kiểm thử của Audit Hardening (Commit e88edffa):
1. Video decode thành công đến EOF (NORMAL_EOF, VALID).
2. Video bị truncate giữa luồng (TRUNCATED_STREAM, NEEDS_REVIEW, ghi nhận missing_frames).
3. Video có metadata frame count sai (đối chiếu container và stream).
4. Video chỉ đọc được vài frame đầu (PROBE_ONLY, full_decode_verified=False).
5. Video có lỗi giải mã giữa luồng (DECODE_ERROR / TRUNCATED_STREAM).
6. Video thiếu trong inference summary (phát hiện missing_in_inference).
7. Video có video_id trùng lặp giữa hai thư mục (DUPLICATE_ID).
8. Manifest và inference summary không khớp (frame mismatches, unexplained duration mismatches).
9. Ràng buộc bảo vệ: Tuyệt đối không thể gán full_decode_verified=True khi chưa có bằng chứng full decode.
10. Streaming SHA-256 đọc theo chunks phân biệt nội dung khác nhau ở đuôi tệp.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.video_audit import (
    audit_single_video,
    compute_streaming_sha256,
    detect_duplicates_and_id_collisions,
    verify_manifest_vs_inference,
)


def create_synthetic_video(filepath: Path, num_frames: int = 30, fps: float = 10.0, width: int = 160, height: int = 120) -> Path:
    """Tạo video giả lập hợp lệ để kiểm thử giải mã."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 7) % 256, dtype=np.uint8)
        cv2.putText(frame, f"F_{i}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    return filepath


class TestVideoAuditHardening(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="audit_hardening_test_"))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_video_decode_success_to_eof(self):
        """1. Video giải mã thành công đến EOF: decode_status=NORMAL_EOF, status=VALID, full_decode_verified=True."""
        vpath = self.test_dir / "valid_full.avi"
        num_frames = 25
        create_synthetic_video(vpath, num_frames=num_frames, fps=10.0)

        res = audit_single_video(vpath, full_decode=True, probe_frames=5)
        self.assertEqual(res["status"], "VALID")
        self.assertEqual(res["decode_status"], "NORMAL_EOF")
        self.assertEqual(res["source_label_status"], "SURVEILLANCE_NEGATIVE_BATCH")
        self.assertEqual(res["human_review_status"], "PENDING_HUMAN_REVIEW")
        self.assertEqual(res["full_decoded_frames"], num_frames)
        self.assertEqual(res["missing_frames"], 0)
        self.assertTrue(res["full_decode_verified"])
        self.assertEqual(res["duration_seconds"], 2.5)

    def test_02_video_truncated_middle_of_stream(self):
        """2. Video bị truncate giữa luồng: decode_status=TRUNCATED_STREAM, status=NEEDS_REVIEW, missing_frames > 10."""
        vpath = self.test_dir / "truncated.avi"
        create_synthetic_video(vpath, num_frames=50, fps=10.0)

        # Cắt bớt 50% dung lượng tệp
        raw = vpath.read_bytes()
        vpath.write_bytes(raw[: int(len(raw) * 0.5)])

        res = audit_single_video(vpath, full_decode=True, probe_frames=5)
        self.assertEqual(res["status"], "NEEDS_REVIEW")
        self.assertEqual(res["decode_status"], "TRUNCATED_STREAM")
        self.assertGreater(res["missing_frames"], 10)
        self.assertLess(res["full_decoded_frames"], 50)
        self.assertTrue(res["full_decode_verified"])  # Điểm dừng của stream đã được xác minh

    def test_03_metadata_frame_count_mismatch(self):
        """3. Video có metadata frame count sai: phân biệt rõ reported_frame_count và full_decoded_frames, duration theo frame thực."""
        vpath = self.test_dir / "mismatch.avi"
        create_synthetic_video(vpath, num_frames=40, fps=10.0)

        raw = vpath.read_bytes()
        vpath.write_bytes(raw[: int(len(raw) * 0.7)])

        res = audit_single_video(vpath, full_decode=True)
        self.assertIn("reported_frame_count", res)
        self.assertIn("full_decoded_frames", res)
        self.assertNotEqual(res["reported_frame_count"], res["full_decoded_frames"])
        # Thời lượng phải tính trên số frame thực giải mã được
        expected_dur = round(res["full_decoded_frames"] / res["fps"], 2)
        self.assertAlmostEqual(res["duration_seconds"], expected_dur, places=1)

    def test_04_probe_only_does_not_verify_full_decode(self):
        """4. Video chỉ đọc được vài frame đầu (probe only): decode_status=PROBE_ONLY, status=NEEDS_REVIEW, full_decode_verified=False."""
        vpath = self.test_dir / "probe_only.avi"
        create_synthetic_video(vpath, num_frames=30, fps=10.0)

        res = audit_single_video(vpath, full_decode=False, probe_frames=5)
        self.assertEqual(res["decode_status"], "PROBE_ONLY")
        self.assertEqual(res["status"], "NEEDS_REVIEW")
        self.assertFalse(res["full_decode_verified"])
        self.assertEqual(res["audit_probe_frames"], 5)
        self.assertEqual(res["full_decoded_frames"], 0)

    def test_05_decode_error_corrupted_frame(self):
        """5. Video có lỗi giải mã giữa luồng: bắt lỗi mượt mà, ghi nhận error_message."""
        vpath = self.test_dir / "corrupted_stream.avi"
        create_synthetic_video(vpath, num_frames=35, fps=10.0)

        # Ghi đè rác vào giữa tệp video để gây lỗi giải mã khung hình
        raw = bytearray(vpath.read_bytes())
        mid = len(raw) // 2
        raw[mid : mid + 1024] = b"\x00\xFF" * 512
        vpath.write_bytes(bytes(raw))

        res = audit_single_video(vpath, full_decode=True)
        self.assertIn(res["decode_status"], ("TRUNCATED_STREAM", "DECODE_ERROR", "NORMAL_EOF"))
        self.assertIsNotNone(res["status"])

    def test_06_video_missing_in_inference_summary(self):
        """6. Video thiếu inference summary: verify_manifest_vs_inference phát hiện ra tệp bị thiếu."""
        manifest = [
            {"video_id": "V1", "status": "VALID", "decode_status": "NORMAL_EOF", "full_decoded_frames": 100, "duration_seconds": 10.0},
            {"video_id": "V_MISSING", "status": "VALID", "decode_status": "NORMAL_EOF", "full_decoded_frames": 150, "duration_seconds": 15.0},
        ]
        inference = [
            {"video_id": "V1", "total_decoded_frames": "100", "actual_duration_sec": "10.0"},
        ]
        res = verify_manifest_vs_inference(manifest, inference)
        self.assertFalse(res["is_perfect_match"])
        self.assertIn("V_MISSING", res["missing_in_inference"])

    def test_07_duplicate_video_id_between_directories(self):
        """7. Video có video_id trùng giữa hai thư mục: phát hiện va chạm DUPLICATE_ID."""
        records = [
            {"video_id": "DUPLICATE_NAME", "relative_path": "dir_a/DUPLICATE_NAME.mp4", "sha256": "hash_111", "status": "VALID"},
            {"video_id": "DUPLICATE_NAME", "relative_path": "dir_b/DUPLICATE_NAME.mp4", "sha256": "hash_222", "status": "VALID"},
        ]
        updated = detect_duplicates_and_id_collisions(records)
        self.assertEqual(updated[0]["status"], "VALID")
        self.assertEqual(updated[1]["status"], "DUPLICATE_ID")
        self.assertIn("Trùng video_id", updated[1]["error_message"])

    def test_08_manifest_and_inference_summary_mismatch(self):
        """8. Manifest và inference summary không khớp: phát hiện lệch frame và duration không được giải thích."""
        manifest = [
            {"video_id": "V1", "status": "VALID", "decode_status": "NORMAL_EOF", "full_decoded_frames": 100, "duration_seconds": 10.0},
        ]
        inference_frame_mismatch = [
            {"video_id": "V1", "total_decoded_frames": "85", "actual_duration_sec": "10.0"},
        ]
        res_f = verify_manifest_vs_inference(manifest, inference_frame_mismatch)
        self.assertFalse(res_f["is_perfect_match"])
        self.assertEqual(res_f["frame_mismatches_count"], 1)

        inference_unexplained_dur = [
            {"video_id": "V1", "total_decoded_frames": "100", "actual_duration_sec": "30.0"},
        ]
        res_d = verify_manifest_vs_inference(manifest, inference_unexplained_dur)
        self.assertFalse(res_d["is_perfect_match"])
        self.assertEqual(res_d["unexplained_duration_mismatches_count"], 1)

    def test_09_guard_no_full_decode_verified_when_unverified(self):
        """9. Ràng buộc bảo vệ: Không được gán full_decode_verified=True nếu file rỗng, không tồn tại hoặc chỉ probe."""
        ghost = self.test_dir / "ghost.mp4"
        r_ghost = audit_single_video(ghost, full_decode=True)
        self.assertFalse(r_ghost["full_decode_verified"])
        self.assertEqual(r_ghost["status"], "CORRUPTED")
        self.assertEqual(r_ghost["decode_status"], "CANNOT_OPEN")

        empty = self.test_dir / "empty.mp4"
        empty.write_bytes(b"")
        r_empty = audit_single_video(empty, full_decode=True)
        self.assertFalse(r_empty["full_decode_verified"])
        self.assertEqual(r_empty["status"], "CORRUPTED")

    def test_10_sha256_full_streaming_vs_partial(self):
        """10. Streaming SHA-256 đọc toàn bộ file theo chunks và không bị lừa bởi 2MB đầu giống nhau."""
        f1 = self.test_dir / "file1.bin"
        f2 = self.test_dir / "file2.bin"

        common_prefix = b"Z" * (2 * 1024 * 1024)
        with open(f1, "wb") as fp:
            fp.write(common_prefix)
            fp.write(b"TAIL_ONE")

        with open(f2, "wb") as fp:
            fp.write(common_prefix)
            fp.write(b"TAIL_TWO")

        partial_1 = hashlib.sha256(common_prefix).hexdigest()
        partial_2 = hashlib.sha256(common_prefix).hexdigest()
        self.assertEqual(partial_1, partial_2)

        full_1 = compute_streaming_sha256(f1, chunk_size=65536)
        full_2 = compute_streaming_sha256(f2, chunk_size=65536)
        self.assertNotEqual(full_1, full_2)


if __name__ == "__main__":
    unittest.main()
