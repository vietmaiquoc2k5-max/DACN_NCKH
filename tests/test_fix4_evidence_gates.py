"""Cross-platform clean checkout and evidence-gate regressions for FIX-4."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
import io
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from src.ground_truth_integrity import validate_frozen_snapshot
from src.inference_frame_evidence import frame_evidence_for_video, inspect_frame_jsonl
from src.week4_acceptance import REQUIRED_REUSE_CHECKS, acceptance, reuse_decision
from scripts import run_phases4_to_8_recovery as recovery

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = "data/ground_truth/events_v1.1_snapshot.jsonl"
CHECKSUM = "data/ground_truth/checksums_v1.1.sha256"


class TestFix4EvidenceGates(unittest.TestCase):
    def test_clean_checkout_preserves_crlf_in_both_git_modes(self):
        """Perform real fresh checkouts; this same test executes on Linux CI."""
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        for mode in ("false", "true"):
            with self.subTest(core_autocrlf=mode), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "checkout"
                subprocess.run(["git", "clone", "--shared", "--no-checkout", str(ROOT), str(target)],
                               check=True, capture_output=True)
                subprocess.run(["git", "-C", str(target), "-c", f"core.autocrlf={mode}", "checkout", head,
                                "--", ".gitattributes", SNAPSHOT, CHECKSUM], check=True, capture_output=True)
                data = (target / SNAPSHOT).read_bytes()
                self.assertEqual(data.count(b"\r\n"), 35)
                self.assertEqual(validate_frozen_snapshot(target / SNAPSHOT, target / CHECKSUM),
                                 "8c6f30bbd057d5b5ae453feb64eb010387c2087ab9350ccea922ba04325bf7c3")

    def test_lf_conversion_is_detected_without_hash_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / Path(SNAPSHOT).name
            lf = (ROOT / SNAPSHOT).read_bytes().replace(b"\r\n", b"\n")
            copy.write_bytes(lf)
            self.assertEqual(hashlib.sha256(lf).hexdigest(),
                             "0ff6dcae59645b70fb08665c79064eec9f3484aec8b1e3c8cfa9dd5679bff405")
            with self.assertRaisesRegex(ValueError, "BLOCKED_GT_INTEGRITY"):
                validate_frozen_snapshot(copy, ROOT / CHECKSUM)

    def make_audit(self, rows):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "frames.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return inspect_frame_jsonl(path)

    def frame(self, **changes):
        return dict({"video_id": "v", "frame_index": 0, "timestamp_sec": 0.0, "detections": [],
                     "run_id": "r", "source_sha256": "source", "checkpoint_sha256": "checkpoint",
                     "timestamp_source": "source_pts", "detector_input_config": {"imgsz": 640}}, **changes)

    def evidence(self, audit, **changes):
        arguments = dict(run_id="r", source_sha256="source", checkpoint_sha256="checkpoint",
                         expected_samples=1, detector_config={"imgsz": 640}, reference_timestamps={0: 0.0})
        arguments.update(changes)
        return frame_evidence_for_video("v", [audit], **arguments)

    def test_bound_frame_evidence_matches_independent_reference(self):
        audit = self.make_audit([self.frame()])
        self.assertTrue(self.evidence(audit)["full_timeline_independently_verified"])
        self.assertIsNone(self.evidence(audit, reference_timestamps=None)["frame_timestamps"])
        self.assertFalse(self.evidence(audit, reference_timestamps={0: 0.1})["frame_timestamps"])

    def test_missing_timestamp_or_frame_index_never_proves_coverage(self):
        for field in ("timestamp_sec", "frame_index", "timestamp_source", "detector_input_config"):
            with self.subTest(field=field):
                row = self.frame()
                row.pop(field)
                evidence = self.evidence(self.make_audit([row]))
                self.assertFalse(evidence["full_timeline_independently_verified"])
                self.assertIsNone(evidence["frame_timestamps"])

    def test_wrong_video_or_run_cannot_supply_target_timestamps(self):
        for changes in ({"video_id": "other"}, {"run_id": "other"}, {"source_sha256": "other"},
                        {"checkpoint_sha256": "other"}):
            with self.subTest(changes=changes):
                evidence = self.evidence(self.make_audit([self.frame(**changes)]))
                self.assertEqual(evidence["bound_run_evidence_count"], 0)
                self.assertIsNone(evidence["frame_timestamps"])

    def test_aggregate_counts_do_not_replace_missing_frame_fields(self):
        audit = self.make_audit([{"video_id": "v", "timestamp_sec": 0.0, "detections": []}])
        evidence = self.evidence(audit)
        self.assertEqual(audit["videos"]["v"]["sampled_frame_records"], 1)
        self.assertIsNone(evidence["frame_timestamps"])
        self.assertIsNone(evidence["sampled_frame_counts_verified"])

    def test_bound_timestamp_corruption_or_count_mismatch_is_failed_check(self):
        duplicate = self.make_audit([self.frame(), self.frame()])
        self.assertFalse(self.evidence(duplicate, expected_samples=2)["frame_timestamps"])
        self.assertFalse(self.evidence(self.make_audit([self.frame()]), expected_samples=2)["frame_timestamps"])
        invalid = self.make_audit([self.frame(timestamp_sec=-1)])
        self.assertFalse(self.evidence(invalid)["frame_timestamps"])

    def test_missing_source_blocks_only_the_affected_video(self):
        checks = {key: True for key in REQUIRED_REUSE_CHECKS}
        self.assertEqual(reuse_decision(checks)["recommendation"], "REUSE_VERIFIED")
        missing = dict(checks, source_hashes=None)
        self.assertEqual(reuse_decision(missing, missing_input=True)["recommendation"], "BLOCKED_MISSING_INPUT")
        self.assertIsNone(reuse_decision(missing, missing_input=True)["rerun_required"])

    def test_pending_positive_gt_blocks_official_even_if_other_gates_pass(self):
        result = acceptance(6, 60, True, True, True, True, True, positive_gt_complete=False)
        self.assertFalse(result["operating_point_official_eligible"])
        self.assertEqual(result["week4_acceptance_status"], "PARTIAL")
        unknown = acceptance(6, 60, True, True, True, True, True)
        self.assertFalse(unknown["operating_point_official_eligible"])

    def test_tracked_historical_output_is_rejected_without_writes(self):
        historical = ROOT / "runs/benchmarks/week4_critical_fix/recalculated"
        path = historical / "inference_rerun_decision.json"
        before = path.read_bytes()
        with patch.object(recovery, "out_dir", recovery.out_dir), patch("sys.argv", ["recovery", "--output-dir", str(historical)]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                recovery.main()
            self.assertEqual(error.exception.code, 2)
        self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
