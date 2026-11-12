"""Bộ kiểm thử tính toàn vẹn của audit video, streaming SHA-256, full decode và đồng bộ manifest (tests/test_video_audit_and_consistency.py).

Bao gồm 8 ca kiểm thử bắt buộc theo yêu cầu nghiệm thu Tuần 4:
1. SHA-256 xác nhận toàn bộ file bằng streaming (phân biệt rõ với partial 2MB hash).
2. Video giải mã đầy đủ tuần tự từ đầu đến EOF.
3. Video lỗi ở giữa luồng (bị ngắt giữa chừng, không đọc được đến EOF).
4. Video chỉ giải mã được vài frame đầu (probe only).
5. Phát hiện video trùng lặp bằng SHA-256 đầy đủ.
6. Metadata container frame_count không khớp với số frame giải mã thực tế.
7. Manifest và inference summary không khớp (phát hiện lệch số lượng, frame count hoặc exposure).
8. Ràng buộc bảo vệ: Tuyệt đối không được gán full_decode_verified=True khi chưa giải mã đầy đủ đến EOF.
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
    detect_duplicates_by_hash,
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


class TestVideoAuditAndConsistency(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="video_audit_test_"))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_sha256_full_streaming_vs_partial(self):
        """1. Xác nhận compute_streaming_sha256 đọc toàn bộ nội dung tệp theo chunk và không bị lừa bởi 2MB đầu giống nhau."""
        f1 = self.test_dir / "file1.bin"
        f2 = self.test_dir / "file2.bin"

        # Tạo 2 file có 2.5 MB, trong đó 2 MB đầu hoàn toàn giống nhau, phần đuôi khác nhau
        common_prefix = b"A" * (2 * 1024 * 1024)
        with open(f1, "wb") as fp:
            fp.write(common_prefix)
            fp.write(b"TAIL_1_DIFFERENT_DATA")

        with open(f2, "wb") as fp:
            fp.write(common_prefix)
            fp.write(b"TAIL_2_DIFFERENT_DATA")

        # Partial hash (chỉ đọc 2MB đầu) sẽ bị trùng lặp sai lệch
        partial_h1 = hashlib.sha256(common_prefix).hexdigest()
        partial_h2 = hashlib.sha256(common_prefix).hexdigest()
        self.assertEqual(partial_h1, partial_h2, "Partial hash không thể phân biệt 2 tệp")

        # Streaming hash toàn phần phải phân biệt chính xác 100%
        full_h1 = compute_streaming_sha256(f1, chunk_size=65536)
        full_h2 = compute_streaming_sha256(f2, chunk_size=65536)
        self.assertNotEqual(full_h1, full_h2, "Full streaming SHA-256 phải phát hiện được nội dung khác biệt ở đuôi file")

    def test_02_video_full_decode_to_eof(self):
        """2. Xác nhận video giải mã đầy đủ từ frame 0 đến EOF, đếm chính xác số frame và gán full_decode_verified=True."""
        vpath = self.test_dir / "valid_test.avi"
        num_frames = 25
        create_synthetic_video(vpath, num_frames=num_frames, fps=10.0)

        res = audit_single_video(vpath, full_decode=True, probe_frames=5)
        self.assertEqual(res["status"], "VALID")
        self.assertEqual(res["full_decoded_frames"], num_frames)
        self.assertEqual(res["audit_probe_frames"], 5)
        self.assertTrue(res["full_decode_verified"])
        self.assertEqual(res["full_decode_errors"], 0)
        self.assertEqual(res["duration_seconds"], 2.5)
        self.assertTrue(res["sha256_verified"])
        self.assertEqual(len(res["sha256"]), 64)

    def test_03_video_decode_error_middle_of_stream(self):
        """3. Xác nhận xử lý khi video bị hỏng ở giữa: giải mã được một phần, dừng sớm và không crash."""
        vpath = self.test_dir / "truncated.avi"
        create_synthetic_video(vpath, num_frames=40, fps=10.0)

        file_bytes = vpath.read_bytes()
        truncated_bytes = file_bytes[: int(len(file_bytes) * 0.6)]
        vpath.write_bytes(truncated_bytes)

        res = audit_single_video(vpath, full_decode=True, probe_frames=5)
        self.assertGreater(res["audit_probe_frames"], 0)
        self.assertLess(res["full_decoded_frames"], 40)
        self.assertGreaterEqual(res["full_decoded_frames"], res["audit_probe_frames"])

    def test_04_video_probe_only_does_not_verify_full_decode(self):
        """4. Khi chỉ probe vài frame (full_decode=False), full_decode_verified BẮT BUỘC phải là False."""
        vpath = self.test_dir / "probe_test.avi"
        create_synthetic_video(vpath, num_frames=30, fps=10.0)

        res = audit_single_video(vpath, full_decode=False, probe_frames=5)
        self.assertEqual(res["audit_probe_frames"], 5)
        self.assertFalse(res["full_decode_verified"])
        self.assertEqual(res["full_decoded_frames"], 0)

    def test_05_video_duplicate_detection_by_full_hash(self):
        """5. Phát hiện tệp trùng lặp chính xác bằng full SHA-256."""
        v1 = self.test_dir / "orig.avi"
        v2 = self.test_dir / "diff.avi"
        v3 = self.test_dir / "copy_of_orig.avi"

        create_synthetic_video(v1, num_frames=10, fps=10.0)
        create_synthetic_video(v2, num_frames=20, fps=10.0)
        shutil.copy(v1, v3)

        r1 = audit_single_video(v1, full_decode=False)
        r2 = audit_single_video(v2, full_decode=False)
        r3 = audit_single_video(v3, full_decode=False)

        records = detect_duplicates_by_hash([r1, r2, r3])
        self.assertEqual(records[0]["status"], "VALID")
        self.assertEqual(records[1]["status"], "VALID")
        self.assertEqual(records[2]["status"], "DUPLICATE")
        self.assertEqual(records[2]["duplicate_of"], records[0]["relative_path"])

    def test_06_metadata_frame_count_mismatch(self):
        """6. Khi metadata container frame_count khác thực tế giải mã, hệ thống ghi nhận đúng cả 2 trường và tính duration theo frame thực."""
        vpath = self.test_dir / "mismatch.avi"
        create_synthetic_video(vpath, num_frames=30, fps=10.0)

        raw = vpath.read_bytes()
        vpath.write_bytes(raw[: int(len(raw) * 0.7)])

        res = audit_single_video(vpath, full_decode=True)
        self.assertIn("reported_frame_count", res)
        self.assertIn("full_decoded_frames", res)
        if res["full_decoded_frames"] > 0:
            expected_dur = round(res["full_decoded_frames"] / res["fps"], 2)
            self.assertAlmostEqual(res["duration_seconds"], expected_dur, places=1)

    def test_07_manifest_and_inference_summary_mismatch(self):
        """7. verify_manifest_vs_inference phát hiện sai lệch khi thiếu video hoặc lệch frame count."""
        manifest_videos = [
            {"video_id": "V1", "status": "VALID", "full_decoded_frames": 100, "duration_seconds": 10.0},
            {"video_id": "V2", "status": "VALID", "full_decoded_frames": 200, "duration_seconds": 20.0},
        ]

        # Trường hợp 1: Khớp hoàn hảo
        inf_perfect = [
            {"video_id": "V1", "total_decoded_frames": "100", "actual_duration_sec": "10.0"},
            {"video_id": "V2", "total_decoded_frames": "200", "actual_duration_sec": "20.0"},
        ]
        res_p = verify_manifest_vs_inference(manifest_videos, inf_perfect)
        self.assertTrue(res_p["is_perfect_match"])

        # Trường hợp 2: Thiếu video V2 trong inference
        inf_missing = [
            {"video_id": "V1", "total_decoded_frames": "100", "actual_duration_sec": "10.0"},
        ]
        res_m = verify_manifest_vs_inference(manifest_videos, inf_missing)
        self.assertFalse(res_m["is_perfect_match"])
        self.assertIn("V2", res_m["missing_in_inference"])

        # Trường hợp 3: Lệch số frame
        inf_diff_frames = [
            {"video_id": "V1", "total_decoded_frames": "100", "actual_duration_sec": "10.0"},
            {"video_id": "V2", "total_decoded_frames": "150", "actual_duration_sec": "20.0"},
        ]
        res_d = verify_manifest_vs_inference(manifest_videos, inf_diff_frames)
        self.assertFalse(res_d["is_perfect_match"])
        self.assertEqual(res_d["frame_mismatches_count"], 1)

    def test_08_guard_no_full_decode_verified_when_unverified(self):
        """8. Ràng buộc bảo vệ tuyệt đối: Không được gán full_decode_verified=True nếu file không tồn tại, lỗi mở file, hoặc chỉ probe."""
        # 1. File không tồn tại
        non_existent = self.test_dir / "ghost.mp4"
        r_ghost = audit_single_video(non_existent, full_decode=True)
        self.assertFalse(r_ghost["full_decode_verified"])
        self.assertEqual(r_ghost["status"], "CORRUPTED")

        # 2. File rỗng
        empty_file = self.test_dir / "empty.mp4"
        empty_file.write_bytes(b"")
        r_empty = audit_single_video(empty_file, full_decode=True)
        self.assertFalse(r_empty["full_decode_verified"])
        self.assertEqual(r_empty["status"], "CORRUPTED")

        # 3. File hợp lệ nhưng chỉ probe (full_decode=False)
        vpath = self.test_dir / "valid.avi"
        create_synthetic_video(vpath, num_frames=15, fps=10.0)
        r_probe = audit_single_video(vpath, full_decode=False)
        self.assertFalse(r_probe["full_decode_verified"])


if __name__ == "__main__":
    unittest.main()
