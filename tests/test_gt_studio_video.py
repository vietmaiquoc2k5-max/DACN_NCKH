"""Synthetic video, media streaming and timestamp regression tests."""
from __future__ import annotations

import urllib.error
import urllib.request
import subprocess
from unittest.mock import PropertyMock, patch

from gt_studio_fixtures import StudioCase
from ground_truth_studio.config import Settings, confined
from ground_truth_studio.integrity_service import catalog
from ground_truth_studio.video_service import MediaServer


class TestStudioVideo(StudioCase):
    def test_catalog_has_24_positive_79_negative_without_leaking_onsets(self):
        rows = catalog(self.settings)
        self.assertEqual(sum(row["kind"] == "positive" for row in rows), 24)
        self.assertEqual(sum(row["kind"] == "negative" for row in rows), 79)
        self.assertFalse(any("fire_start_sec" in row or "label" in row for row in rows))

    def test_missing_and_corrupt_video_are_blocked(self):
        with self.assertRaisesRegex(ValueError, "BLOCKED_INPUT"):
            self.videos.inspect(self.repository.video("NEG_02"))
        self.negative.write_bytes(b"corrupt media")
        with self.assertRaisesRegex(ValueError, "BLOCKED_INPUT"):
            self.videos.inspect(dict(self.repository.video("NEG_01"), expected_sha256=None))

    def test_sha_mismatch(self):
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.videos.inspect(dict(self.repository.video("NEG_01"), expected_sha256="0" * 64))

    def test_exact_frame_forward_backward_and_estimated_timestamp(self):
        video = self.repository.video("POS_01")
        with patch.object(Settings, "ffprobe", new_callable=PropertyMock, return_value=None):
            for index in (0, 15, 16, 3):
                image, point = self.videos.frame(video, index)
                self.assertAlmostEqual(float(image.mean()), index * 10, delta=3)
                self.assertEqual(point["frame_index"], index)
                self.assertAlmostEqual(point["timestamp_sec"], index / 10)
                self.assertEqual(point["timestamp_provenance"], "ESTIMATED_FRAME_INDEX_OVER_FPS")

    def test_vfr_seek_uses_pts_not_index_over_fps(self):
        metadata = {"source_pts": [10., 10.04, 10.15, 10.45], "fps": 30., "frame_count": 4}
        self.assertEqual(self.videos.index_at_time(metadata, .2), 2)
        self.assertEqual(self.videos.index_at_time(metadata, .45), 3)
        with self.assertRaises(ValueError):
            self.videos.index_at_time(metadata, float("nan"))

    def test_real_vfr_fixture_uses_source_pts_and_exact_ordinal(self):
        if not self.settings.ffmpeg or not self.settings.ffprobe:
            self.skipTest("FFmpeg/ffprobe unavailable for VFR integration fixture")
        source = self.root / "FireSmokeLabeler/videos/source.avi"
        output = source.with_name("vfr.mp4")
        command = [self.settings.ffmpeg, "-nostdin", "-v", "error", "-i", str(source),
                   "-vf", "setpts='if(lt(N,10),N/(10*TB),(1+(N-10)/5)/TB)'", "-fps_mode", "vfr",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)]
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        video = {"relative_path": "FireSmokeLabeler/videos/vfr.mp4", "expected_sha256": None}
        metadata = self.videos.inspect(video)
        self.assertEqual(metadata["timestamp_provenance"], "SOURCE_PTS")
        self.assertEqual(len(metadata["source_pts"]), 20)
        image, point = self.videos.frame(video, 12)
        self.assertEqual(point["timestamp_sec"], metadata["source_pts"][12] - metadata["source_pts"][0])
        self.assertAlmostEqual(float(image.mean()), 120, delta=5)

    def test_streaming_range_and_path_traversal(self):
        server = MediaServer(self.repository)
        self.addCleanup(server.close)
        url = server.grant(self.tokens["first"], self.negative)
        request = urllib.request.Request(url, headers={"Range": "bytes=5-16"})
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), self.negative.read_bytes()[5:17])
        with self.assertRaises(ValueError):
            confined(self.root, "../private/ground_truth_studio.sqlite3")
        with self.assertRaises(ValueError):
            server.grant(self.tokens["first"], self.private / "negative_annotator_01.csv")
        self.repository.logout(self.tokens["first"])
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url)
        self.assertEqual(error.exception.code, 403)
