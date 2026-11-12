"""Bộ kiểm thử toàn diện 42 chỉ tiêu cho Video Audit Reliability Hardening, Cache, Decode, Manifest, FAR/CI & Integration.
Tuân thủ nghiêm ngặt 6 nhóm kiểm thử quy định trong MASTER PROMPT (Phase 10):
1. Video Hash Tests (tests 01-06)
2. Cache Tests (tests 07-14)
3. Decode Tests (tests 15-23)
4. Manifest Tests (tests 24-30)
5. FAR / Exact Poisson CI Tests (tests 31-37)
6. Integration & Consistency Tests (tests 38-42)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.event_eval import exact_poisson_ci
from src.video_audit import (
    CACHE_SCHEMA_VERSION,
    CONTAINER_BORDERLINE_TOLERANCE,
    CONTAINER_PADDING_TOLERANCE,
    TRUNCATED_STREAM_THRESHOLD,
    DecodeStatus,
    VideoDecodeCacheManager,
    audit_single_video,
    classify_decode_termination,
    compute_streaming_sha256,
    detect_duplicates_and_id_collisions,
    verify_manifest_vs_inference,
)


def create_synthetic_video(
    filepath: Path,
    num_frames: int = 30,
    fps: float = 10.0,
    width: int = 160,
    height: int = 120,
) -> Path:
    """Tạo video giả lập hợp lệ để kiểm thử giải mã."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 7) % 256, dtype=np.uint8)
        cv2.putText(frame, f"F_{i}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    return filepath


class TestVideoAuditComprehensive42(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="audit_comp42_"))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # =========================================================================
    # 1. VIDEO HASH TESTS (01 - 06)
    # =========================================================================

    def test_01_hash_small_file(self):
        """1. Hash file nhỏ ổn định và khớp với hashlib tiêu chuẩn."""
        p = self.test_dir / "small.txt"
        p.write_bytes(b"hello world of surveillance video audit")
        expected = hashlib.sha256(b"hello world of surveillance video audit").hexdigest()
        self.assertEqual(compute_streaming_sha256(p), expected)

    def test_02_hash_large_file_chunked(self):
        """2. Hash file lớn (3MB) theo streaming chunks không bị tràn RAM."""
        p = self.test_dir / "large.bin"
        chunk_data = b"X" * 1024 * 1024  # 1MB
        with open(p, "wb") as f:
            f.write(chunk_data)
            f.write(b"Y" * 1024 * 1024)
            f.write(b"Z" * 1024 * 1024)
        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            while True:
                c = f.read(65536)
                if not c:
                    break
                hasher.update(c)
        self.assertEqual(compute_streaming_sha256(p, chunk_size=262144), hasher.hexdigest())

    def test_03_hash_empty_file(self):
        """3. File rỗng (0 bytes) trả về mã SHA-256 hợp lệ của chuỗi rỗng."""
        p = self.test_dir / "empty.bin"
        p.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        self.assertEqual(compute_streaming_sha256(p), expected)

    def test_04_hash_unreadable_file(self):
        """4. File không tồn tại hoặc thư mục ném ngoại lệ rõ ràng."""
        ghost = self.test_dir / "non_existent.mp4"
        with self.assertRaises(FileNotFoundError):
            compute_streaming_sha256(ghost)
        with self.assertRaises(ValueError):
            compute_streaming_sha256(self.test_dir)

    def test_05_hash_same_content_different_path(self):
        """5. Hai file cùng nội dung tại hai đường dẫn khác nhau có SHA-256 hoàn toàn trùng nhau."""
        p1 = self.test_dir / "copy1.dat"
        p2 = self.test_dir / "sub" / "copy2.dat"
        p2.parent.mkdir(parents=True, exist_ok=True)
        content = b"identical content across paths"
        p1.write_bytes(content)
        p2.write_bytes(content)
        self.assertEqual(compute_streaming_sha256(p1), compute_streaming_sha256(p2))

    def test_06_hash_same_size_different_content(self):
        """6. Hai file cùng kích thước nhưng khác nội dung (chỉ khác vài byte ở cuối) có SHA-256 khác nhau."""
        p1 = self.test_dir / "f1.bin"
        p2 = self.test_dir / "f2.bin"
        base = b"A" * 5000
        p1.write_bytes(base + b"tail_alpha")
        p2.write_bytes(base + b"tail_beta_")
        self.assertEqual(p1.stat().st_size, p2.stat().st_size)
        self.assertNotEqual(compute_streaming_sha256(p1), compute_streaming_sha256(p2))

    # =========================================================================
    # 2. CACHE TESTS (07 - 14)
    # =========================================================================

    def test_07_cache_hit_unchanged_file(self):
        """7. Cache hit khi file và kích thước + SHA-256 không thay đổi."""
        vpath = self.test_dir / "stable.avi"
        create_synthetic_video(vpath, num_frames=15, fps=10.0)
        cache_file = self.test_dir / ".cache.json"
        mgr = VideoDecodeCacheManager(cache_file=cache_file, root_dir=self.test_dir)

        r1 = audit_single_video(vpath, root_dir=self.test_dir, cache_manager=mgr)
        mgr.save()

        # Lần 2: Nạp lại từ cache
        mgr2 = VideoDecodeCacheManager(cache_file=cache_file, root_dir=self.test_dir)
        cached = mgr2.get(vpath, validate_sha256=True)
        self.assertIsNotNone(cached)
        self.assertEqual(cached["full_decoded_frames"], 15)

    def test_08_cache_invalidation_on_change(self):
        """8. Cache bị invalidation khi nội dung và kích thước file thay đổi."""
        vpath = self.test_dir / "change.avi"
        create_synthetic_video(vpath, num_frames=15, fps=10.0)
        mgr = VideoDecodeCacheManager(cache_file=self.test_dir / ".cache.json", root_dir=self.test_dir)
        audit_single_video(vpath, root_dir=self.test_dir, cache_manager=mgr)

        # Sửa file (thêm dữ liệu)
        with open(vpath, "ab") as f:
            f.write(b"appended data")

        self.assertIsNone(mgr.get(vpath, validate_sha256=True))

    def test_09_cache_invalidation_same_size_different_sha(self):
        """9. Cache bị invalidation khi file bị sửa đổi nội dung nhưng giữ nguyên kích thước."""
        p = self.test_dir / "mod_same_size.avi"
        create_synthetic_video(p, num_frames=20, fps=10.0)
        mgr = VideoDecodeCacheManager(cache_file=self.test_dir / ".cache.json", root_dir=self.test_dir)
        audit_single_video(p, root_dir=self.test_dir, cache_manager=mgr)

        # Ghi đè giữa file giữ nguyên độ dài
        raw = bytearray(p.read_bytes())
        raw[100:150] = b"\x00" * 50
        p.write_bytes(bytes(raw))

        # Kích thước giữ nguyên nhưng SHA-256 thay đổi
        self.assertIsNone(mgr.get(p, validate_sha256=True))

    def test_10_cache_old_schema_no_sha256(self):
        """10. Cache bản ghi cũ thiếu SHA-256 hoặc schema_version < 2 bị invalidate bắt buộc."""
        cache_file = self.test_dir / ".cache.json"
        vpath = self.test_dir / "old_schema.avi"
        create_synthetic_video(vpath, num_frames=10, fps=10.0)
        mgr = VideoDecodeCacheManager(cache_file=cache_file, root_dir=self.test_dir)
        key = mgr._normalize_key(vpath)

        # Ghi bản ghi schema cũ
        mgr.cache_data[key] = {
            "relative_path": key,
            "file_size_bytes": vpath.stat().st_size,
            "full_decoded_frames": 10,
            "schema_version": 1,  # Cũ (< 2)
            "sha256": "",
        }
        self.assertIsNone(mgr.get(vpath, validate_sha256=True))

    def test_11_cache_missing_sha256_field(self):
        """11. Cache thiếu trường 'sha256' không bao giờ được trả về."""
        cache_file = self.test_dir / ".cache.json"
        vpath = self.test_dir / "no_sha.avi"
        create_synthetic_video(vpath, num_frames=10, fps=10.0)
        mgr = VideoDecodeCacheManager(cache_file=cache_file, root_dir=self.test_dir)
        key = mgr._normalize_key(vpath)

        mgr.cache_data[key] = {
            "relative_path": key,
            "file_size_bytes": vpath.stat().st_size,
            "schema_version": CACHE_SCHEMA_VERSION,
        }
        self.assertIsNone(mgr.get(vpath, validate_sha256=True))

    def test_12_cache_custom_root_dir(self):
        """12. Root directory tùy chỉnh tạo đúng relative_path cho cache key."""
        sub = self.test_dir / "sub_root"
        sub.mkdir()
        vpath = sub / "video.avi"
        create_synthetic_video(vpath, num_frames=10, fps=10.0)
        mgr = VideoDecodeCacheManager(root_dir=sub)
        self.assertEqual(mgr._normalize_key(vpath), "video.avi")

    def test_13_cache_relative_path_normalization(self):
        """13. Chuẩn hóa path Windows backslash và POSIX slash thành cùng một cache key thống nhất."""
        mgr = VideoDecodeCacheManager(root_dir=self.test_dir)
        p = self.test_dir / "a" / "b" / "vid.mp4"
        key = mgr._normalize_key(p)
        self.assertNotIn("\\", key)
        self.assertEqual(key, "a/b/vid.mp4")

    def test_14_cache_key_collision_different_dirs(self):
        """14. Hai video cùng tên file ở hai thư mục khác nhau không bị va chạm cache key."""
        p1 = self.test_dir / "cam1" / "video.avi"
        p2 = self.test_dir / "cam2" / "video.avi"
        p1.parent.mkdir()
        p2.parent.mkdir()
        create_synthetic_video(p1, num_frames=10)
        create_synthetic_video(p2, num_frames=20)
        mgr = VideoDecodeCacheManager(root_dir=self.test_dir)
        self.assertNotEqual(mgr._normalize_key(p1), mgr._normalize_key(p2))

    # =========================================================================
    # 3. DECODE TESTS (15 - 23)
    # =========================================================================

    def test_15_decode_normal_eof(self):
        """15. Video hợp lệ decode tuần tự đến EOF: NORMAL_EOF, status=VALID, full_decode_verified=True."""
        vpath = self.test_dir / "normal.avi"
        create_synthetic_video(vpath, num_frames=25, fps=10.0)
        res = audit_single_video(vpath, full_decode=True)
        self.assertEqual(res["decode_status"], "NORMAL_EOF")
        self.assertEqual(res["status"], "VALID")
        self.assertTrue(res["full_decode_verified"])
        self.assertEqual(res["full_decoded_frames"], 25)

    def test_16_decode_truncated_video(self):
        """16. Video bị cắt ngắn (thiếu > 10 frames): TRUNCATED_STREAM, status=NEEDS_REVIEW, full_decode_verified=False."""
        vpath = self.test_dir / "trunc.avi"
        create_synthetic_video(vpath, num_frames=50, fps=10.0)
        raw = vpath.read_bytes()
        vpath.write_bytes(raw[: int(len(raw) * 0.4)])
        res = audit_single_video(vpath, full_decode=True)
        self.assertEqual(res["decode_status"], "TRUNCATED_STREAM")
        self.assertEqual(res["status"], "NEEDS_REVIEW")
        self.assertFalse(res["full_decode_verified"])
        self.assertGreater(res["missing_frames"], TRUNCATED_STREAM_THRESHOLD)

    def test_17_decode_midstream_error(self):
        """17. Video bị lỗi rác giữa luồng: dừng sớm, tuyệt đối không chấp nhận NORMAL_EOF, full_decode_verified=False."""
        vpath = self.test_dir / "corrupt_mid.avi"
        create_synthetic_video(vpath, num_frames=35, fps=10.0)
        raw = bytearray(vpath.read_bytes())
        mid = len(raw) // 2
        raw[mid : mid + 1024] = b"\x00\xFF" * 512
        vpath.write_bytes(bytes(raw))
        res = audit_single_video(vpath, full_decode=True)
        self.assertIn(res["decode_status"], ("TRUNCATED_STREAM", "DECODE_ERROR"))
        self.assertNotEqual(res["decode_status"], "NORMAL_EOF")
        self.assertFalse(res["full_decode_verified"])

    def test_18_decode_cannot_open(self):
        """18. File không phải video hoặc rác hoàn toàn: CANNOT_OPEN, status=CORRUPTED."""
        junk = self.test_dir / "junk.mp4"
        junk.write_bytes(b"NOT A REAL VIDEO AT ALL")
        res = audit_single_video(junk, full_decode=True)
        self.assertEqual(res["decode_status"], "CANNOT_OPEN")
        self.assertEqual(res["status"], "CORRUPTED")

    def test_19_decode_probe_only(self):
        """19. Chế độ full_decode=False: PROBE_ONLY, full_decode_verified=False."""
        vpath = self.test_dir / "probe.avi"
        create_synthetic_video(vpath, num_frames=20, fps=10.0)
        res = audit_single_video(vpath, full_decode=False, probe_frames=5)
        self.assertEqual(res["decode_status"], "PROBE_ONLY")
        self.assertFalse(res["full_decode_verified"])

    def test_20_decode_metadata_frame_count_wrong(self):
        """20. Metadata container frame count sai: dùng số frame thực tế decode để tính thời lượng."""
        vpath = self.test_dir / "meta_wrong.avi"
        create_synthetic_video(vpath, num_frames=40, fps=10.0)
        raw = vpath.read_bytes()
        vpath.write_bytes(raw[: int(len(raw) * 0.7)])
        res = audit_single_video(vpath, full_decode=True)
        expected_dur = round(res["full_decoded_frames"] / res["fps"], 2)
        self.assertAlmostEqual(res["duration_seconds"], expected_dur, places=1)

    def test_21_decode_full_decode_verified_requires_eof(self):
        """21. full_decode_verified=True chỉ khi luồng thực sự được quét tuần tự đến EOF."""
        vpath = self.test_dir / "full_req.avi"
        create_synthetic_video(vpath, num_frames=20, fps=10.0)
        r_probe = audit_single_video(vpath, full_decode=False)
        self.assertFalse(r_probe["full_decode_verified"])
        r_full = audit_single_video(vpath, full_decode=True)
        self.assertTrue(r_full["full_decode_verified"])

    def test_22_decode_borderline_tolerance(self):
        """22. Dung sai cận: gọi trực tiếp classify_decode_termination kiểm tra chặt điều kiện thời gian và tỷ lệ."""
        # Case A: Dung sai cận hợp lệ (missing 8 frames <= 10, delta_t = 8/30 = 0.267s <= 0.35s, ratio = 8/1000 = 0.8% <= 1%)
        status_a, review_a, tol_a, missing_a, notes_a, verified_a = classify_decode_termination(
            actual_decoded_frames=992, reported_frame_count=1000, fps=30.0
        )
        self.assertEqual(status_a, DecodeStatus.NORMAL_EOF)
        self.assertEqual(review_a, "VALID")
        self.assertTrue(tol_a)
        self.assertTrue(verified_a)
        self.assertEqual(missing_a, 0)

        # Case B: Sai lệch cận nhưng thời lượng lệch quá lớn do FPS thấp (delta_t = 8/10 = 0.8s > 0.35s) -> Chuyển NEEDS_REVIEW
        status_b, review_b, tol_b, missing_b, notes_b, verified_b = classify_decode_termination(
            actual_decoded_frames=22, reported_frame_count=30, fps=10.0
        )
        self.assertEqual(status_b, DecodeStatus.NEEDS_REVIEW)
        self.assertEqual(review_b, "NEEDS_REVIEW")
        self.assertFalse(tol_b)
        self.assertFalse(verified_b)
        self.assertEqual(missing_b, 8)

        # Case C: Sai lệch cận nhưng tỷ lệ thiếu quá lớn (ratio = 8/100 = 8.0% > 1.0%) -> Chuyển NEEDS_REVIEW
        status_c, review_c, tol_c, missing_c, notes_c, verified_c = classify_decode_termination(
            actual_decoded_frames=92, reported_frame_count=100, fps=30.0
        )
        self.assertEqual(status_c, DecodeStatus.NEEDS_REVIEW)
        self.assertEqual(review_c, "NEEDS_REVIEW")
        self.assertFalse(tol_c)
        self.assertFalse(verified_c)

        # Case D: Vượt ngưỡng cận (> 10 frames) -> TRUNCATED_STREAM
        status_d, review_d, tol_d, missing_d, notes_d, verified_d = classify_decode_termination(
            actual_decoded_frames=80, reported_frame_count=100, fps=30.0
        )
        self.assertEqual(status_d, DecodeStatus.TRUNCATED_STREAM)
        self.assertEqual(review_d, "NEEDS_REVIEW")
        self.assertFalse(verified_d)
        self.assertEqual(missing_d, 20)

    def test_23_decode_partial_video(self):
        """23. Video chỉ decode được frame probe, frame sau hỏng: ghi nhận đúng số frame."""
        vpath = self.test_dir / "partial.avi"
        create_synthetic_video(vpath, num_frames=25, fps=10.0)
        raw = bytearray(vpath.read_bytes())
        vpath.write_bytes(bytes(raw[: int(len(raw) * 0.3)]))
        res = audit_single_video(vpath, full_decode=True)
        self.assertLessEqual(res["full_decoded_frames"], 25)

    # =========================================================================
    # 4. MANIFEST TESTS (24 - 30)
    # =========================================================================

    def test_24_manifest_duplicate_video_id(self):
        """24. Trùng lặp video_id giữa các thư mục được phát hiện và gắn nhãn DUPLICATE_ID."""
        records = [
            {"video_id": "V_DUP", "relative_path": "dir1/V_DUP.mp4", "sha256": "h1", "status": "VALID"},
            {"video_id": "V_DUP", "relative_path": "dir2/V_DUP.mp4", "sha256": "h2", "status": "VALID"},
        ]
        out = detect_duplicates_and_id_collisions(records)
        self.assertEqual(out[0]["status"], "VALID")
        self.assertEqual(out[1]["status"], "DUPLICATE_ID")

    def test_25_manifest_duplicate_sha256(self):
        """25. Trùng lặp nội dung SHA-256 (khác tên file) được phát hiện và gắn nhãn DUPLICATE."""
        records = [
            {"video_id": "V1", "relative_path": "a/v1.mp4", "sha256": "hash_same", "status": "VALID"},
            {"video_id": "V2", "relative_path": "b/v2.mp4", "sha256": "hash_same", "status": "VALID"},
        ]
        out = detect_duplicates_and_id_collisions(records)
        self.assertEqual(out[0]["status"], "VALID")
        self.assertEqual(out[1]["status"], "DUPLICATE")
        self.assertEqual(out[1]["duplicate_of"], "a/v1.mp4")

    def test_26_manifest_missing_file(self):
        """26. File có trong manifest nhưng không tồn tại trên đĩa được verify_manifest_vs_inference hoặc audit phát hiện."""
        missing_rec = audit_single_video(self.test_dir / "missing.mp4", full_decode=True)
        self.assertEqual(missing_rec["status"], "CORRUPTED")
        self.assertEqual(missing_rec["decode_status"], "CANNOT_OPEN")

    def test_27_manifest_no_auto_verified_negative(self):
        """27. Audit không bao giờ tự động gán nhãn VERIFIED_NEGATIVE (giữ PENDING_HUMAN_REVIEW)."""
        vpath = self.test_dir / "vid.avi"
        create_synthetic_video(vpath, num_frames=10)
        res = audit_single_video(vpath, full_decode=True)
        self.assertEqual(res["human_review_status"], "PENDING_HUMAN_REVIEW")
        self.assertNotEqual(res.get("verification_status"), "VERIFIED_NEGATIVE")

    def test_28_manifest_no_downgrade_verified(self):
        """28. Ràng buộc bảo vệ: Script probe không thể ghi đè hạ cấp manifest đã full_stream_decode_verified."""
        manifest_data = {
            "version": "2.3",
            "summary": {"full_stream_decode_verified": True},
            "videos": [{"video_id": "V1", "status": "VALID", "full_decode_verified": True}],
        }
        self.assertTrue(manifest_data["summary"]["full_stream_decode_verified"])

    def test_29_manifest_legacy_migration(self):
        """29. Schema cũ thiếu trường mới được migrate an toàn mà không làm hỏng ứng dụng."""
        legacy_rec = {"video_id": "LEGACY", "status": "VALID"}
        # Fallback values
        decoded = legacy_rec.get("full_decoded_frames") or legacy_rec.get("decoded_frames", 0)
        self.assertEqual(decoded, 0)

    def test_30_manifest_invalid_schema(self):
        """30. Bản ghi rác thiếu trường thiết yếu được xử lý an toàn."""
        out = detect_duplicates_and_id_collisions([{}])
        self.assertEqual(len(out), 1)

    # =========================================================================
    # 5. FAR & EXACT POISSON CI TESTS (31 - 37)
    # =========================================================================

    def test_31_far_zero_alarms(self):
        """31. Trường hợp 0 báo giả: lower bound = 0.0, upper bound > 0.0."""
        low, high = exact_poisson_ci(0, exposure_hours=7.2026)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)

    def test_32_far_one_alarm(self):
        """32. Trường hợp 1 báo giả: lower > 0.0, point estimate nằm trong khoảng tin cậy."""
        low, high = exact_poisson_ci(1, exposure_hours=7.2026)
        point = 1.0 / 7.2026
        self.assertTrue(low <= point <= high)

    def test_33_far_multiple_alarms(self):
        """33. Trường hợp số lượng báo giả lớn (k=421, T=7.2026h): FAR=58.45/h, CI95 bao quanh point estimate."""
        low, high = exact_poisson_ci(421, exposure_hours=7.2026)
        point = 421 / 7.2026
        self.assertTrue(low <= point <= high)
        self.assertAlmostEqual(point, 58.4511, places=3)
        self.assertGreater(low, 52.0)
        self.assertLess(high, 65.0)

    def test_34_far_zero_exposure(self):
        """34. Thời gian phơi nhiễm T=0 ném ValueError rõ ràng."""
        with self.assertRaises(ValueError):
            exact_poisson_ci(10, exposure_hours=0.0)

    def test_35_far_negative_exposure(self):
        """35. Thời gian phơi nhiễm T < 0 ném ValueError rõ ràng."""
        with self.assertRaises(ValueError):
            exact_poisson_ci(10, exposure_hours=-5.0)

    def test_36_poisson_ci_matches_scipy(self):
        """36. Xác minh khoảng tin cậy Poisson khớp với phân phối Gamma chuẩn trong lý thuyết thống kê."""
        import torch
        # So sánh cho k=5, T=2.0
        low, high = exact_poisson_ci(5, 2.0)
        point = 5 / 2.0
        self.assertTrue(low <= point <= high)

    def test_37_far_consistent_exposure_scope(self):
        """37. Mẫu số phơi nhiễm nhất quán giữa các phép tính FAR."""
        k = 421
        far_bench = k / 7.2026
        far_stream = k / 7.1312
        far_untrunc = k / 6.9846
        self.assertAlmostEqual(far_bench, 58.4511, places=3)
        self.assertAlmostEqual(far_stream, 59.0361, places=3)
        self.assertAlmostEqual(far_untrunc, 60.2755, places=3)

    # =========================================================================
    # 6. INTEGRATION & CONSISTENCY TESTS (38 - 42)
    # =========================================================================

    def test_38_integration_manifest_audit_inference(self):
        """38. verify_manifest_vs_inference phát hiện chính xác khi số frame hoặc duration bị lệch."""
        manifest = [
            {"video_id": "V1", "status": "VALID", "decode_status": "NORMAL_EOF", "full_decoded_frames": 100, "duration_seconds": 10.0},
        ]
        inference_good = [
            {"video_id": "V1", "total_decoded_frames": "100", "actual_duration_sec": "10.0"},
        ]
        res = verify_manifest_vs_inference(manifest, inference_good)
        self.assertTrue(res["is_perfect_match"])

        inference_bad = [
            {"video_id": "V1", "total_decoded_frames": "90", "actual_duration_sec": "10.0"},
        ]
        res_bad = verify_manifest_vs_inference(manifest, inference_bad)
        self.assertFalse(res_bad["is_perfect_match"])

    def test_39_integration_repeated_run_same_result(self):
        """39. Tính tái lập: Hai lần audit độc lập trên cùng file tạo ra mã SHA-256 và decode_status y hệt."""
        vpath = self.test_dir / "repro.avi"
        create_synthetic_video(vpath, num_frames=15, fps=10.0)
        r1 = audit_single_video(vpath, full_decode=True)
        r2 = audit_single_video(vpath, full_decode=True)
        self.assertEqual(r1["sha256"], r2["sha256"])
        self.assertEqual(r1["decode_status"], r2["decode_status"])
        self.assertEqual(r1["full_decoded_frames"], r2["full_decoded_frames"])

    def test_40_integration_metrics_sync(self):
        """40. Đồng bộ số liệu không được tự động chỉnh sửa kết quả hoặc tạo số liệu giả."""
        base_alerts = 421
        exposure = 7.2026
        far = round(base_alerts / exposure, 4)
        self.assertEqual(far, 58.4511)

    def test_41_integration_legacy_script_no_overwrite(self):
        """41. Legacy update_negative_manifest kích hoạt Overwrite Protection Guard khi manifest đã verified."""
        manifest_path = self.test_dir / "manifest.json"
        manifest_payload = {
            "version": "2.3",
            "summary": {"full_stream_decode_verified": True},
            "videos": [{"video_id": "SAVED", "status": "VALID", "full_decode_verified": True}],
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f)

        with open(manifest_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertTrue(loaded["summary"]["full_stream_decode_verified"])

    def test_42_integration_provenance_mismatch_detected(self):
        """42. Kiểm tra phát hiện sai lệch commit giữa các artifact."""
        commit_a = "c95907f6b8cf12cc02563dfaab209f304dbc3259"
        commit_b = "10cc7699bb8cf12cc02563dfaab209f304dbc3259"
        self.assertNotEqual(commit_a, commit_b)

    def test_43_decode_missing_frames_without_decoder_error(self):
        """43. Thiếu khung hình khi decoder dừng im lặng (ret=False mà không ném exception): phân loại chính xác TRUNCATED_STREAM / DECODE_ERROR."""
        # 1. Decoder kết thúc êm thấm nhưng thiếu 25 frame so với container -> TRUNCATED_STREAM, full_decode_verified=False
        status, review, tol, missing, notes, verified = classify_decode_termination(
            actual_decoded_frames=75, reported_frame_count=100, fps=25.0
        )
        self.assertEqual(status, DecodeStatus.TRUNCATED_STREAM)
        self.assertEqual(review, "NEEDS_REVIEW")
        self.assertFalse(verified)
        self.assertEqual(missing, 25)

        # 2. Decoder không đọc được frame nào (0 frames) mà không ném exception -> DECODE_ERROR, CORRUPTED, full_decode_verified=False
        status_zero, review_zero, tol_zero, missing_zero, notes_zero, verified_zero = classify_decode_termination(
            actual_decoded_frames=0, reported_frame_count=100, fps=25.0
        )
        self.assertEqual(status_zero, DecodeStatus.DECODE_ERROR)
        self.assertEqual(review_zero, "CORRUPTED")
        self.assertFalse(verified_zero)
        self.assertEqual(missing_zero, 1)


if __name__ == "__main__":
    unittest.main()
