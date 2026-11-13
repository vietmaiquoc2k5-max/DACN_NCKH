"""Draft persistence, actual intervals, new events and concurrent editing."""
from __future__ import annotations

import threading
import multiprocessing
from pathlib import Path

from gt_studio_fixtures import StudioCase
from ground_truth_studio.models import validate_events
from ground_truth_studio.review_repository import ConflictError, ReviewRepository
from ground_truth_studio.config import Settings


def _process_write(root, private, token, review_id, event, queue):
    repository = ReviewRepository(Settings(Path(root), Path(private)))
    review = repository.review(token, review_id)
    event.wait(timeout=10)
    try:
        repository.save(token, review_id, 0, review["payload"], None)
        queue.put("saved")
    except ConflictError:
        queue.put("conflict")


class TestStudioAnnotations(StudioCase):
    def test_onset_offset_and_class_validation(self):
        event = self.event()
        for changed in (dict(event, start_sec=1., end_sec=.5), dict(event, start_sec=float("nan")), dict(event, label="AI_FIRE")):
            with self.assertRaises(ValueError):
                validate_events([changed], 2.)
        with self.assertRaises(ValueError):
            validate_events([dict(event, frame_start=None)], 2., submit=True)
        for malformed in ({}, [None], [dict(event, label=[])], [dict(event, start_sec=None)], [dict(event, start_sec=True)]):
            with self.assertRaises(ValueError):
                validate_events(malformed, 2.)

    def test_duplicate_interval_with_different_ids_is_rejected(self):
        first, second = self.event(), self.event()
        with self.assertRaisesRegex(ValueError, "Duplicate class"):
            validate_events([first, second], 2.)

    def test_new_fire_smoke_events_and_resume_after_restart(self):
        token = self.tokens["second"]
        review = self.repository.start_review(token, "POS_01")
        self.assertEqual(review["payload"]["events"], [])
        events = [self.event(label="FIRE"), self.event(label="SMOKE", start=1, end=10)]
        payload = dict(review["payload"], events=events)
        saved = self.annotations.save(token, review["id"], 0, payload)
        reopened = ReviewRepository(self.settings).review(token, review["id"])
        self.assertEqual(reopened["payload"]["events"], events)
        self.assertEqual(saved["status"], "DRAFT")
        self.assertEqual(reopened["revision"], 1)

    def test_submit_requires_explicit_view_and_evidence_and_locks(self):
        token = self.tokens["second"]
        review = self.repository.start_review(token, "POS_01")
        with self.assertRaises(ValueError):
            self.annotations.save(token, review["id"], 0, review["payload"], submit=True)
        submitted = self.submit("second", events=[self.event()])
        self.assertEqual(submitted["status"], "SUBMITTED")
        self.assertNotEqual(submitted["status"], "VERIFIED")
        with self.assertRaises(ConflictError):
            self.annotations.save(token, review["id"], submitted["revision"], submitted["payload"])

    def test_negative_fire_and_contradictory_negative(self):
        event = self.event("NEG_01")
        review = self.repository.start_review(self.tokens["first"], "NEG_01")
        with self.assertRaises(ValueError):
            self.annotations.save(self.tokens["first"], review["id"], 0, dict(review["payload"], events=[event], decision="VERIFIED_NEGATIVE"))
        result = self.submit("first", "NEG_01", [event], "CONTAINS_FIRE")
        self.assertEqual(result["payload"]["decision"], "CONTAINS_FIRE")

    def test_two_process_style_writers_use_compare_and_swap(self):
        token = self.tokens["first"]
        review = self.repository.start_review(token, "POS_01")
        second = ReviewRepository(self.settings)
        barrier = threading.Barrier(2)
        outcomes = []
        def write(repository):
            barrier.wait(timeout=5)
            try:
                repository.save(token, review["id"], 0, review["payload"], None)
                outcomes.append("saved")
            except ConflictError:
                outcomes.append("conflict")
        threads = [threading.Thread(target=write, args=(repository,)) for repository in (self.repository, second)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertCountEqual(outcomes, ["saved", "conflict"])

    def test_two_actual_processes_cannot_overwrite_each_other(self):
        token = self.tokens["first"]
        review = self.repository.start_review(token, "POS_01")
        context = multiprocessing.get_context("spawn")
        event, queue = context.Event(), context.Queue()
        processes = [context.Process(target=_process_write, args=(str(self.root), str(self.private), token, review["id"], event, queue)) for _ in range(2)]
        for process in processes:
            process.start()
        event.set()
        for process in processes:
            process.join(timeout=25)
            self.assertEqual(process.exitcode, 0)
        self.assertCountEqual([queue.get(timeout=5), queue.get(timeout=5)], ["saved", "conflict"])
        queue.close()
