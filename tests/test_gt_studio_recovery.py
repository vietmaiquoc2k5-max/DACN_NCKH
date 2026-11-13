"""Synthetic catalog recovery, migration preservation and single-person research mode."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from gt_studio_fixtures import StudioCase
from ground_truth_studio.annotation_service import AnnotationService
from ground_truth_studio.catalog_recovery import recover_catalog
from ground_truth_studio.config import Settings
from ground_truth_studio.integrity_service import catalog, sha256
from ground_truth_studio.review_repository import ReviewRepository
from ground_truth_studio.single_reviewer import STATUS, export_single_reviewer


class TestStudioRecovery(StudioCase):
    def candidate_fixture(self):
        directory = self.root / "datasets/negative_videos"
        raw = self.negative.read_bytes()
        records = []
        for index in range(1, 64):
            path = directory / f"VN_NEG_{index:02}.mp4"
            path.write_bytes(raw)
            records.append({"video_id": path.stem, "relative_path": path.relative_to(self.root).as_posix(), "sha256": sha256(path)})
        records += [{"video_id": f"negsVideo{i}", "relative_path": f"datasets/test_videos/neg/{i}.avi", "sha256": "missing"} for i in range(16)]
        for index in range(1, 81):
            (directory / f"VN_POS_{index:02}.mp4").write_bytes(raw)
        (self.root / "data/negative_videos_manifest.json").write_text(json.dumps({"videos": records}))
        return Settings(self.root, self.private.parent / "recovery_private")

    def test_63_negative_80_candidate_16_missing_without_id_mixing(self):
        settings = self.candidate_fixture()
        repository = ReviewRepository(settings)
        self.assertEqual(len(repository.working_videos("negative")), 63)
        self.assertEqual(len(repository.working_videos("positive_candidate")), 80)
        self.assertEqual(len(repository.working_videos("positive")), 24)
        self.assertEqual(sum(not row["available"] for row in repository.videos()), 16)
        self.assertFalse(any(row["video_id"].startswith("VN_POS") for row in repository.working_videos("negative")))
        original = [row["video_id"] for row in catalog(settings)]
        self.assertEqual(original, [row["video_id"] for row in catalog(settings)])

    def test_discovery_hashes_duplicates_and_keeps_missing_manifest(self):
        settings = self.candidate_fixture()
        manifest = (self.root / "data/negative_videos_manifest.json").read_bytes()
        output = self.root / "runs/benchmarks/synthetic_recovery"
        with patch("ground_truth_studio.catalog_recovery.probe_metadata", return_value={"probe_status": "METADATA_ONLY", "duration_seconds": 2., "fps": 10., "frame_count": 20}):
            recover_catalog(settings, output)
        report = json.loads((output / "catalog_reconciliation_report.json").read_text())
        self.assertEqual(report["physical_prefixed_videos"], 143)
        self.assertEqual(report["missing_historical_count"], 16)
        self.assertTrue(report["duplicate_hash_groups"])
        self.assertIn("VN_POS_50", report["duplicate_hash_groups"][0]["video_ids"])
        self.assertEqual(manifest, (self.root / "data/negative_videos_manifest.json").read_bytes())

    def test_ui_catalog_dropdowns_exclude_missing_and_separate_candidates(self):
        from streamlit.testing.v1 import AppTest
        settings = self.candidate_fixture()
        repository = ReviewRepository(settings)
        repository.provision("solo", "synthetic-ui-solo", "annotator_01", "synthetic-password-123", "SYNTHETIC_ONLY")
        token = repository.login("solo", "synthetic-password-123")
        config = settings.private / "settings.json"
        config.write_text(json.dumps({"repo_root": str(settings.repo), "private_dir": str(settings.private)}))
        entry = Path(__file__).resolve().parent.parent / "ground_truth_studio/app.py"
        with patch.dict("os.environ", {"GT_STUDIO_CONFIG": str(config), "GT_STUDIO_PRIVATE_DIR": str(settings.private)}):
            app = AppTest.from_file(entry, default_timeout=60)
            app.session_state["auth_token"] = token
            app.run()
            for page, count, prefix in [("Negative Review", 63, "VN_NEG"), ("VN_POS Candidate Review", 80, "VN_POS"), ("Positive Review", 24, "POS_")]:
                next(item for item in app.radio if item.label == "Điều hướng").set_value(page).run()
                options = next(item for item in app.selectbox if item.label == "Video nguồn").options
                self.assertEqual(len(options), count)
                self.assertTrue(all(item.startswith(prefix) for item in options))
                self.assertFalse(app.exception)

    def test_migration_backup_preserves_user_draft_and_is_idempotent(self):
        review = self.repository.start_review(self.tokens["first"], "POS_01")
        with self.repository.connection() as db:
            db.execute("PRAGMA user_version=0")
            before = [tuple(row) for row in db.execute("SELECT * FROM users ORDER BY id")]
        first = ReviewRepository(self.settings)
        self.assertEqual(first.review(self.tokens["first"], review["id"])["payload"], review["payload"])
        backups = list((self.private / "backups").glob("*.sqlite3"))
        self.assertTrue(backups)
        second = ReviewRepository(self.settings)
        self.assertEqual(len(list((self.private / "backups").glob("*.sqlite3"))), len(backups))
        with second.connection() as db:
            self.assertEqual([tuple(row) for row in db.execute("SELECT * FROM users ORDER BY id")], before)
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 2)

    def test_one_account_submit_resume_export_and_blind_round(self):
        settings = Settings(self.root, self.private.parent / "solo_private")
        repository = ReviewRepository(settings)
        repository.provision("solo", "synthetic-solo-person", "annotator_01", "synthetic-password-123", "SYNTHETIC_ONLY")
        token = repository.login("solo", "synthetic-password-123")
        service = AnnotationService(repository, self.videos)
        review = repository.start_review(token, "POS_01")
        payload = dict(review["payload"], events=[self.event()], attest_full_view=True, evidence_ref="SYNTHETIC_FIXTURE_ONLY")
        submitted = service.save(token, review["id"], 0, payload, submit=True)
        again = ReviewRepository(settings)
        bundle = export_single_reviewer(again, token)
        candidate = json.loads((bundle / "single_reviewer_gt_candidate.jsonl").read_text().splitlines()[0])
        self.assertEqual(candidate["annotation_status"], STATUS)
        self.assertEqual(candidate["verification_status"], "pending_verification")
        self.assertFalse(candidate["cross_verified"])
        blind = again.start_blind_rereview(token, "POS_01")
        self.assertEqual(blind["payload"]["events"], [])
        self.assertEqual(again.review(token, submitted["id"])["status"], "SUBMITTED")
        export_single_reviewer(again, token)  # Original submitted round remains exportable.
        with again.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM users").fetchone()[0], 1)
        with self.assertRaises(PermissionError):
            again.historical(token, "POS_01")

    def test_regeneration_retains_existing_reviews_and_hash_pin_blocks_change(self):
        review = self.repository.start_review(self.tokens["first"], "POS_01")
        pin = {"video_id": "NEG_01", "relative_path": self.repository.video("NEG_01")["relative_path"], "sha256": "0" * 64}
        (self.root / "data/video_catalog_local_v2.json").write_text(json.dumps({"videos": [pin]}))
        refreshed = ReviewRepository(self.settings)
        self.assertEqual(refreshed.review(self.tokens["first"], review["id"])["id"], review["id"])
        # Historical expected SHA is retained; changed bytes still fail the reader's source guard.
        self.negative.write_bytes(b"changed source")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.videos.inspect(refreshed.video("NEG_01"))
