"""Synthetic independent rounds preserve prior work and require real submissions."""
import json
from gt_studio_fixtures import StudioCase
from ground_truth_studio.independent_review import IndependentReview


class TestIndependentRounds(StudioCase):
    def test_joint_prior_retained_and_new_round_empty(self):
        old = self.submit("first")
        service = IndependentReview(self.repository)
        new = service.start(self.tokens["first"], "POS_01", joint_prior=True)
        self.assertNotEqual(old["id"], new["id"])
        self.assertFalse(new["payload"]["events"])
        self.assertEqual(self.repository.review(self.tokens["first"],old["id"])["status"],"SUBMITTED")
        self.assertEqual(service.start(self.tokens["first"],"POS_01")["id"],new["id"])
        with self.assertRaises(PermissionError):
            self.repository.review(self.tokens["second"], new["id"])

    def test_prior_pair_cannot_be_exported_as_independent(self):
        self.submit("first"); self.submit("second")
        with self.assertRaises(PermissionError):
            IndependentReview(self.repository).comparison(self.tokens["first"],"POS_01")

    def test_two_rounds_attestation_export_pending_adjudication(self):
        service=IndependentReview(self.repository)
        for who in ("first","second"):
            token=self.tokens[who]
            row=service.start(token,"POS_01")
            payload=dict(row["payload"],events=[self.event()],attest_full_view=True,evidence_ref="SYNTHETIC_ONLY")
            with self.assertRaisesRegex(ValueError,"independent-view"):
                self.annotations.save(token,row["id"],0,payload,submit=True)
            payload["attest_independent"]=True
            self.annotations.save(token,row["id"],0,payload,submit=True)
            if who=="first":
                with self.assertRaises(PermissionError): service.comparison(token,"POS_01")
        folder=service.export(self.tokens["first"],"POS_01")
        result=json.loads((folder/"dual_review.json").read_text())
        self.assertFalse(result["official"])
        self.assertEqual(result["status"],"INDEPENDENT_DUAL_REVIEW_PENDING_ADJUDICATION")

    def test_adjudicator_cannot_mix_new_round_with_joint_prior(self):
        self.submit('second')
        row=IndependentReview(self.repository).start(self.tokens['first'],'POS_01')
        payload=dict(row['payload'],events=[self.event()],attest_full_view=True,attest_independent=True,evidence_ref='SYNTHETIC_ONLY')
        self.annotations.save(self.tokens['first'],row['id'],0,payload,submit=True)
        with self.assertRaisesRegex(PermissionError,'mix independent'):
            self.repository.submitted_pair(self.tokens['judge'],'POS_01')
