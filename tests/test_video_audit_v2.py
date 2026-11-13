"""V2 evidence gates, real CPU decode and preservation; no scientific promotion."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ground_truth_studio.config import Settings
from src.video_audit import classify_decode_termination
from src.video_audit_v2 import POLICY, audit_video, classify, inspect_timeline, resume_record, run_tool


def evidence():
    return {"sha256_verified": True, "opencv_actual_decoded_frames": 3, "opencv_reported_frame_count": 8,
        "ffprobe_nb_read_frames": 3, "ffprobe_nb_read_packets": 3, "ffmpeg_decoded_frames": 3,
        "nb_frames_metadata": None, "ffmpeg_progress_end": True, "time_base": "1/10", "stream_duration_ts": 30,
        "stream_start_pts": 0, "tools": {name: {"exit_code": 0, "stderr": "", "timed_out": False} for name in ("counts", "timeline", "ffmpeg")},
        "timestamp_continuity": {"continuous": True, "frame_records": 3, "packet_records": 3,
                                 "first_frame_pts_ticks": 0, "duration_ticks": 30, "duration_sum_ticks": 30}}


class TestV2Gates(unittest.TestCase):
    def test_clean_metadata_mismatch(self):
        result = classify(evidence())
        self.assertEqual(result["decode_status"], "NORMAL_EOF")
        self.assertEqual(result["reason_code"], "CONTAINER_FRAME_COUNT_MISMATCH_BUT_DECODE_VERIFIED")

    def test_counts_equal(self):
        self.assertTrue(classify(evidence())["technical_eligible"])

    def test_ffprobe_disagrees(self):
        self.assertFalse(classify(dict(evidence(), ffprobe_nb_read_frames=4))["technical_eligible"])

    def test_opencv_actual_disagrees(self):
        self.assertFalse(classify(dict(evidence(), opencv_actual_decoded_frames=2))["technical_eligible"])

    def test_nonzero_exit_not_proof_of_corruption(self):
        row = evidence()
        row["tools"]["ffmpeg"]["exit_code"] = 1
        self.assertEqual(classify(row)["decode_status"], "NEEDS_REVIEW")

    def test_missing_progress_end(self):
        self.assertFalse(classify(dict(evidence(), ffmpeg_progress_end=False))["technical_eligible"])

    def test_decoder_errors(self):
        row = evidence()
        row["tools"]["ffmpeg"]["stderr"] = "Decoding error: Invalid NAL unit size"
        self.assertEqual(classify(row)["decode_status"], "TRUNCATED_STREAM")
        row["opencv_actual_decoded_frames"] = 0
        self.assertEqual(classify(row)["decode_status"], "DECODE_ERROR")

    def timeline(self, durations, positions=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "pts.txt"
        positions = positions or [sum(durations[:index]) for index in range(len(durations))]
        path.write_text("\n".join(f"packet|pts={p}|dts={p}|duration={d}|flags=___\nframe|pts={p}|best_effort_timestamp={p}|duration={d}" for p, d in zip(positions, durations)))
        return inspect_timeline(path)

    def test_timestamp_discontinuity(self):
        result = self.timeline([10, 10, 10], [0, 11, 21])
        self.assertFalse(result["continuous"])

    def test_vfr_explicit_durations(self):
        row = evidence()
        row["timestamp_continuity"] = self.timeline([5, 15, 10])
        self.assertTrue(row["timestamp_continuity"]["variable_frame_durations"])
        self.assertEqual(classify(row)["duration_seconds"], 3.)

    def test_timeout(self):
        with tempfile.TemporaryDirectory() as name:
            result = run_tool([sys.executable, "-c", "import time; time.sleep(2)"], Path(name), "timeout", 0.02)
            self.assertTrue(result["timed_out"])
        row = evidence()
        row["tools"]["ffmpeg"]["timed_out"] = True
        self.assertEqual(classify(row)["decode_status"], "NEEDS_REVIEW")

    def test_sha_mismatch(self):
        self.assertFalse(classify(dict(evidence(), sha256_verified=False))["technical_eligible"])

    def test_no_human_promotion(self):
        row = classify(evidence())
        self.assertEqual(row["human_review_status"], "PENDING_HUMAN_REVIEW")
        self.assertFalse(row["semantic_verified_negative"])

    def test_duration_not_container_or_fps_estimate(self):
        row = evidence()
        row.update(container_duration=999, opencv_fps=999)
        self.assertEqual(classify(row)["duration_seconds"], 3.)
        row["stream_duration_ts"] = 31
        self.assertFalse(classify(row)["technical_eligible"])

    def test_no_arbitrary_delta_tolerance(self):
        for reported in (3, 4, 300000000):
            self.assertTrue(classify(dict(evidence(), opencv_reported_frame_count=reported))["technical_eligible"])
        self.assertFalse(classify(dict(evidence(), ffmpeg_decoded_frames=4))["technical_eligible"])

    def test_explicit_metadata_sample_count_contradiction_excluded(self):
        self.assertFalse(classify(dict(evidence(), nb_frames_metadata=4))["technical_eligible"])

    def test_v1_v2_diff(self):
        ext = {"tool_available": True, "decode_capable": True, "nb_read_frames": 3, "exit_code": 0,
               "error_detected": False, "is_clean_decode": True, "progress_end": True}
        self.assertEqual(str(classify_decode_termination(3, 8, 30, external_check=ext)[0]), "NEEDS_REVIEW")
        self.assertEqual(classify(evidence())["decode_status"], "NORMAL_EOF")

    def test_technical_hours_not_scientific_exposure(self):
        row = classify(evidence())
        self.assertTrue(row["technical_eligible"])
        self.assertFalse(row["scientific_far_eligible"])
        self.assertIsNone(row["scientific_far_per_hour"])

    def test_preflight_refuses_existing_output_without_overwrite(self):
        from scripts.recover_negative_decode_v2 import preflight
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            marker = folder / "v1_artifact.json"
            marker.write_bytes(b"historical bytes")
            with self.assertRaisesRegex(ValueError, "Output exists"):
                preflight(folder)
            self.assertEqual(marker.read_bytes(), b"historical bytes")

    def test_priority_uses_decode_evidence_not_alerts(self):
        from scripts.recover_negative_decode_v2 import group
        row = {"full_decoded_frames": 3, "external_tool_check": {"is_clean_decode": True, "nb_read_frames": 3},
               "ffprobe_metadata": {"packet_timestamp_summary": {"status": "AVAILABLE", "dts_regressions_in_packet_order": 0}},
               "alerts": 99999}
        self.assertEqual(group(row), "A")
        row["external_tool_check"]["nb_read_frames"] = 2
        self.assertEqual(group(row), "B")


class TestV2ActualDecode(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = Settings.load()
        if not self.settings.ffmpeg or not self.settings.ffprobe:
            self.skipTest("FFmpeg/ffprobe unavailable")
        self.path = self.root / "synthetic.mp4"
        subprocess.run([self.settings.ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10", "-t", "1", "-c:v", "libx264", "-threads", "1", str(self.path)], check=True, timeout=30)
        from src.video_audit import compute_streaming_sha256
        self.sha256 = compute_streaming_sha256
        self.entry = {"video_id": "SYNTHETIC_ONLY", "relative_path": self.path.name, "sha256": self.sha256(self.path)}

    def audit(self):
        return audit_video(self.entry, self.root, self.root / "record", self.settings.ffmpeg, self.settings.ffprobe, 30)

    def test_actual_decode_source_and_v1_bytes_preserved(self):
        original = self.path.read_bytes()
        v1 = Path(__file__).resolve().parent.parent / "src/video_audit.py"
        old = v1.read_bytes()
        row = self.audit()
        self.assertTrue(row["technical_eligible"], row["exclusion_reasons"])
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(v1.read_bytes(), old)

    def test_resume_and_tampered_artifact_blocked(self):
        self.audit()
        folder = self.root / "record"
        self.assertEqual(resume_record(folder, self.entry, self.root)["audit_policy_version"], POLICY)
        (folder / "ffmpeg.stdout").write_text("changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            resume_record(folder, self.entry, self.root)

    def test_resume_changed_source_blocked(self):
        self.audit()
        self.path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source/policy mismatch"):
            resume_record(self.root / "record", self.entry, self.root)

    def test_corrupt_actual_input_not_accepted(self):
        self.path.write_bytes(b"broken mp4 source")
        self.entry["sha256"] = self.sha256(self.path)
        self.assertFalse(self.audit()["technical_eligible"])
