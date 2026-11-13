"""Missing events and time/class disagreement require deliberate human decisions."""
from __future__ import annotations

from gt_studio_fixtures import StudioCase
from ground_truth_studio.adjudication_service import AdjudicationService
from ground_truth_studio.models import compare_events


class TestStudioAdjudication(StudioCase):
    def test_onset_class_and_missing_event_are_disputed_without_averaging(self):
        first = [self.event()]
        shifted = [self.event(start=8, end=15)]
        for second in (shifted, [self.event(label="SMOKE")], []):
            result = compare_events(first, second)
            self.assertEqual(result["status"], "DISPUTED")
            self.assertTrue(result["unmatched_first"])
        self.assertEqual(compare_events(first, [self.event(start=3, end=9)])["status"], "AGREEMENT")

    def test_adjudication_requires_both_submissions_reason_evidence_and_omissions(self):
        service = AdjudicationService(self.repository)
        self.submit("first", events=[self.event()])
        with self.assertRaises(PermissionError):
            service.comparison(self.tokens["judge"], "POS_01")
        self.submit("second", events=[self.event()])
        for reason, evidence in (("", "reference"), ("reason", "")):
            with self.assertRaises(ValueError):
                service.decide(self.tokens["judge"], "POS_01", "ACCEPT", reason, evidence, [], {})
        with self.assertRaisesRegex(ValueError, "missing historical event"):
            service.decide(self.tokens["judge"], "POS_01", "ACCEPT", "synthetic reason", "synthetic evidence", [], {})
        context = service.comparison(self.tokens["judge"], "POS_01")
        omissions = {row["historical_event_id"]: "Synthetic test: reviewer explicitly did not observe event" for row in context["historical"]}
        service.decide(self.tokens["judge"], "POS_01", "ACCEPT", "synthetic reason", "synthetic evidence", [], omissions)

    def test_require_review_retains_old_locked_records_and_new_drafts(self):
        first = self.submit("first", events=[self.event()])
        second = self.submit("second", events=[self.event()])
        service = AdjudicationService(self.repository)
        service.decide(self.tokens["judge"], "POS_01", "REQUIRE_REVIEW", "Synthetic review request", "synthetic evidence", [], {})
        reopened = self.repository.start_review(self.tokens["first"], "POS_01")
        self.assertNotEqual(reopened["id"], first["id"])
        self.assertEqual(self.repository.review(self.tokens["first"], first["id"])["status"], "SUBMITTED")
        self.assertFalse(reopened["payload"]["attest_full_view"])
        with self.assertRaises(PermissionError):
            service.comparison(self.tokens["judge"], "POS_01")
