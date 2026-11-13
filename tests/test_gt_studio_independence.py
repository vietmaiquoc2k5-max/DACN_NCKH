"""Credential and backend enforcement of blind independent review."""
from __future__ import annotations

import sqlite3

from gt_studio_fixtures import StudioCase


class TestStudioIndependence(StudioCase):
    def test_reviewer02_cannot_read_first_history_or_adjudication(self):
        first = self.submit("first", events=[self.event()])
        second = self.tokens["second"]
        for call in (lambda: self.repository.review(second, first["id"]),
                     lambda: self.repository.historical(second, "POS_01"),
                     lambda: self.repository.submitted_pair(second, "POS_01"),
                     lambda: self.repository.own_csv_rows(second, "positive_adjudication.csv")):
            with self.assertRaises(PermissionError):
                call()
        self.submit("second", events=[self.event()])
        with self.assertRaises(PermissionError):
            self.repository.historical(second, "POS_01")

    def test_one_person_cannot_register_two_names_or_switch_roles(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.repository.provision("alias", "synthetic-person-first", "annotator_02", "synthetic-password-123", "TEST_FIXTURE_ONLY")
        with self.assertRaises(PermissionError):
            self.repository.start_review(self.tokens["judge"], "POS_01")
        with self.assertRaises(PermissionError):
            self.repository.login("first' OR 1=1 --", "bad-password")

    def test_anonymous_or_forged_identity_is_denied(self):
        for token in ("first", "", "adjudicator"):
            with self.assertRaises(PermissionError):
                self.repository.start_review(token, "POS_01")
