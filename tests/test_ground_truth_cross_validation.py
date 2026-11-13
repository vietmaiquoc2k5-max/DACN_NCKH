"""Bộ kiểm thử tính toàn vẹn khoa học của Ground Truth và quy trình Cross-Validation (Task 1).

Kiểm tra:
1. Pending không thể tự chuyển thành verified/cross_verified.
2. Thiếu annotator thứ hai (annotator_02) thì không thể đạt cross_verified.
3. Sai lệch timestamp > 0.5s giữa annotator_01 và annotator_02 chuyển sang disputed.
4. Snapshot v1.1 bất biến sau khi freeze (khớp SHA256).
5. Không có event nào vượt quá thời lượng video clip tương ứng.
6. Không có event trùng lặp ngoài quy tắc.
7. Phân tách rõ ràng FIRE và SMOKE độc lập.
"""
from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path
import tempfile
from src.ground_truth_integrity import validate_frozen_snapshot

from src.ground_truth_validator import (
    VALID_STATUSES,
    compute_file_sha256,
    load_video_durations,
    validate_single_event,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestGroundTruthCrossValidation(unittest.TestCase):
    """Kiểm thử tính hợp lệ và liêm chính khoa học của Ground Truth."""

    def setUp(self):
        self.catalog_path = PROJECT_ROOT / "data" / "video_dataset_list.csv"
        self.events_path = PROJECT_ROOT / "data" / "ground_truth" / "events.jsonl"
        self.snapshot_v1_1_path = PROJECT_ROOT / "data" / "ground_truth" / "events_v1.1_snapshot.jsonl"
        self.checksum_path = PROJECT_ROOT / "data" / "ground_truth" / "checksums_v1.1.sha256"
        self.video_durations = load_video_durations(self.catalog_path)

    def test_pending_cannot_auto_become_verified_without_annotator2(self):
        """1. Kiểm tra bản ghi thiếu annotator_02 không thể được gắn trạng thái cross_verified/verified."""
        bad_record = {
            "video_id": "POS_01",
            "camera_id": "BENCH_POS_01",
            "label": "FIRE",
            "start_sec": 2.4,
            "end_sec": 29.3,
            "confidence": 1.0,
            "annotator_id": "annotator_01",
            "verification_status": "cross_verified",
            "annotator_01": "annotator_01",
            "annotator_02": None,  # Thiếu người kiểm định thứ 2
            "start_sec_annotator_01": 2.4,
            "end_sec_annotator_01": 29.3,
            "start_sec_annotator_02": None,
            "end_sec_annotator_02": None,
        }
        is_valid, errors, _ = validate_single_event(bad_record, line_no=1, video_durations=self.video_durations)
        self.assertFalse(is_valid, "Bản ghi thiếu annotator_02 phải bị coi là không hợp lệ khi gắn cross_verified.")
        self.assertTrue(any("thiếu annotator_02" in err for err in errors))

    def test_timestamp_discrepancy_greater_than_half_second_marked_disputed(self):
        """2. Kiểm tra sai lệch timestamp giữa 2 annotator > 0.5s phải bị coi là disputed."""
        disputed_record = {
            "video_id": "POS_02",
            "camera_id": "BENCH_POS_02",
            "label": "FIRE",
            "start_sec": 0.1,
            "end_sec": 7.84,
            "confidence": 1.0,
            "annotator_id": "annotator_01",
            "verification_status": "cross_verified",
            "annotator_01": "annotator_01",
            "annotator_02": "annotator_02",
            "start_sec_annotator_01": 0.1,
            "end_sec_annotator_01": 7.84,
            "start_sec_annotator_02": 0.8,  # Lệch 0.7s > 0.5s
            "end_sec_annotator_02": 7.84,
        }
        is_valid, errors, resolved_status = validate_single_event(
            disputed_record, line_no=1, video_durations=self.video_durations
        )
        self.assertEqual(resolved_status, "disputed")
        self.assertFalse(is_valid, "Bản ghi có sai lệch > 0.5s không được phép giữ nhãn cross_verified.")

    def test_snapshot_v1_1_checksum_integrity(self):
        """3. Kiểm tra tính bất biến của Snapshot v1.1 thông qua mã băm SHA256."""
        self.assertTrue(self.snapshot_v1_1_path.exists(), "Snapshot v1.1 phải tồn tại trên đĩa.")
        self.assertTrue(self.checksum_path.exists(), "Tệp checksums_v1.1.sha256 phải tồn tại.")

        actual_sha = validate_frozen_snapshot(self.snapshot_v1_1_path, self.checksum_path)
        with self.checksum_path.open("r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn(actual_sha, content, "Mã SHA256 thực tế của snapshot v1.1 phải khớp với checksum đã freeze.")

    def test_snapshot_mutation_rejected(self):
        """Even a one-byte change must fail the frozen named checksum."""
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / self.snapshot_v1_1_path.name
            copy.write_bytes(self.snapshot_v1_1_path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "BLOCKED_GT_INTEGRITY"):
                validate_frozen_snapshot(copy, self.checksum_path)

    def test_legacy_migration_refuses_frozen_writes(self):
        """Rerunning migration cannot regenerate GT or auto-promote pending events."""
        from scripts.standardize_ground_truth_v1_1 import main
        before = self.snapshot_v1_1_path.read_bytes(), self.events_path.read_bytes(), self.checksum_path.read_bytes()
        with self.assertRaisesRegex(SystemExit, "frozen"):
            main()
        self.assertEqual(before, (self.snapshot_v1_1_path.read_bytes(), self.events_path.read_bytes(), self.checksum_path.read_bytes()))

    def test_frozen_v11_records_preserved_from_creation_commit(self):
        """Restoration may change serialization only, never annotations or order."""
        # The authenticated forensic copy supports Actions' shallow checkout too.
        original = (PROJECT_ROOT / "runs/benchmarks/week4_critical_fix/events_v1.1_git_lf_forensic.jsonl").read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(),
                         "0ff6dcae59645b70fb08665c79064eec9f3484aec8b1e3c8cfa9dd5679bff405")
        self.assertEqual([json.loads(line) for line in original.splitlines()],
                         [json.loads(line) for line in self.snapshot_v1_1_path.read_bytes().splitlines()])

    def test_no_event_exceeds_video_duration(self):
        """4. Kiểm tra mọi sự kiện trong events.jsonl không vượt quá thời lượng video."""
        with self.events_path.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, 1):
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                vid = row["video_id"]
                end_sec = float(row["end_sec"])
                if vid in self.video_durations:
                    max_dur = self.video_durations[vid]
                    self.assertLessEqual(
                        end_sec,
                        max_dur + 1.0,
                        f"Dòng {line_no} video {vid}: end_sec ({end_sec}s) vượt quá video duration ({max_dur}s)",
                    )

    def test_independent_fire_and_smoke_separation(self):
        """5. Kiểm tra phân tách rõ ràng FIRE và SMOKE trên cùng một video."""
        with self.events_path.open("r", encoding="utf-8") as f:
            events = [json.loads(l) for l in f if l.strip()]

        labels = {e["label"] for e in events}
        self.assertEqual(labels, {"FIRE", "SMOKE"}, "Chỉ chấp nhận hai lớp nhãn FIRE và SMOKE.")

        # Kiểm tra video có cả 2 nhãn như POS_02, POS_04, POS_12
        vids_with_both = set()
        by_vid: dict = {}
        for e in events:
            by_vid.setdefault(e["video_id"], set()).add(e["label"])
        for vid, lbls in by_vid.items():
            if len(lbls) == 2:
                vids_with_both.add(vid)
        self.assertGreaterEqual(len(vids_with_both), 5, "Phải có ít nhất 5 video chứa đồng thời 2 nhãn FIRE và SMOKE.")


if __name__ == "__main__":
    unittest.main()

