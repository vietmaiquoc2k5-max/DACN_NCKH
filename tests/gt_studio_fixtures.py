"""Synthetic-only Studio fixtures. These are never real human review evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ground_truth_studio.annotation_service import AnnotationService
from ground_truth_studio.config import Settings
from ground_truth_studio.integrity_service import sha256
from ground_truth_studio.models import Annotation
from ground_truth_studio.review_repository import CSV_FIELDS, ReviewRepository
from ground_truth_studio.video_service import VideoService


class StudioCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root, self.private = base / "repo", base / "private"
        (self.root / "data/ground_truth").mkdir(parents=True)
        (self.root / "datasets/negative_videos").mkdir(parents=True)
        (self.root / "FireSmokeLabeler/videos").mkdir(parents=True)
        self.private.mkdir()
        source = self.root / "FireSmokeLabeler/videos/source.avi"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10., (64, 48))
        if not writer.isOpened():
            raise RuntimeError("Synthetic MJPG writer unavailable")
        for index in range(20):
            writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
        writer.release()
        self.negative = self.root / "datasets/negative_videos/negative.avi"
        self.negative.write_bytes(source.read_bytes())
        with (self.root / "data/video_dataset_list.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["video_id", "relative_path", "file_name"])
            writer.writeheader()
            writer.writerows({"video_id": f"POS_{i:02}", "relative_path": "datasets/missing/source.avi", "file_name": "source.avi"} for i in range(1, 25))
        entries = [{"video_id": f"NEG_{i:02}", "relative_path": "datasets/negative_videos/negative.avi" if i == 1 else f"datasets/missing/{i}.avi", "sha256": sha256(self.negative)} for i in range(1, 80)]
        (self.root / "data/negative_videos_manifest.json").write_text(json.dumps({"videos": entries}))
        events = [{"video_id": f"POS_{i % 24 + 1:02}", "label": "FIRE" if i % 2 == 0 else "SMOKE", "start_sec": .2, "end_sec": .8,
                   "verification_status": "cross_verified" if i < 11 else "pending_verification"} for i in range(35)]
        snapshot = self.root / "data/ground_truth/events_v1.1_snapshot.jsonl"
        snapshot.write_bytes(("\r\n".join(json.dumps(event) for event in events) + "\r\n").encode())
        (snapshot.parent / "checksums_v1.1.sha256").write_text(sha256(snapshot) + "  " + snapshot.name + "\n")
        for name, fields in CSV_FIELDS.items():
            with (self.private / name).open("w", encoding="utf-8", newline="") as stream:
                csv.DictWriter(stream, fieldnames=fields).writeheader()
        self.settings = Settings(self.root, self.private)
        self.repository = ReviewRepository(self.settings)
        self.videos = VideoService(self.settings)
        self.addCleanup(self.videos.close)
        self.annotations = AnnotationService(self.repository, self.videos)
        self.tokens = {}
        for name, role in [("first", "annotator_01"), ("second", "annotator_02"), ("judge", "adjudicator"), ("owner", "admin")]:
            self.repository.provision(name, "synthetic-person-" + name, role, "synthetic-password-123", "TEST_FIXTURE_ONLY")
            self.tokens[name] = self.repository.login(name, "synthetic-password-123")

    def event(self, video_id="POS_01", label="FIRE", start=2, end=8):
        video = self.repository.video(video_id)
        _, a = self.videos.frame(video, start)
        _, b = self.videos.frame(video, end)
        return Annotation.create(label, a, b).to_dict()

    def submit(self, reviewer, video_id="POS_01", events=None, decision=""):
        token = self.tokens[reviewer]
        review = self.repository.start_review(token, video_id)
        payload = dict(review["payload"], events=events or [], decision=decision,
                       attest_full_view=True, evidence_ref="synthetic-fixture://not-human-evidence")
        return self.annotations.save(token, review["id"], review["revision"], payload, submit=True)
