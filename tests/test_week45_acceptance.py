"""Synthetic regression cases for source-bound human GT; no human records are made."""
from __future__ import annotations

import copy
import unittest

from src.week45_ground_truth import (duplicate_policy, local_review_issues,
                                    validate_submission)


def record() -> dict:
    """Build a synthetic validation fixture, never an exported human submission."""
    return {'video_id':'SYNTHETIC_A', 'source_sha256':'a'*64,
            'submitted_at_utc':'SYNTHETIC_TIME', 'reviewer_id':'SYNTHETIC_ONLY',
            'reviewer_person_id':'SYNTHETIC_ONLY', 'review_status':'SINGLE_HUMAN_REVIEWED_PROVISIONAL',
            'evidence_ref':'SYNTHETIC_ONLY', 'attest_full_view':True, 'uncertain_intervals':[],
            'events':[{'label':'FIRE', 'frame_start':5, 'frame_end':10,
                       'start_sec':.2, 'end_sec':.4, 'source_sha256':'a'*64,
                       'timestamp_provenance':'SOURCE_PTS', 'event_id':'SYNTHETIC_EVENT'}]}


class TestHumanGTContracts(unittest.TestCase):
    def validate(self, value: dict) -> list[str]:
        return validate_submission(value, list(range(100, 2100, 40)), '1/1000', 2.)

    def test_source_pts_accepts_nonzero_start_vfr_and_cross_class_overlap(self):
        value=record()
        self.assertEqual(self.validate(value), [])
        smoke=copy.deepcopy(value['events'][0]);smoke.update(label='SMOKE',event_id='SYNTHETIC_SMOKE')
        value['events'].append(smoke)
        self.assertEqual(self.validate(value), [])
        irregular=[100, 133, 180, 220]
        value['events']=[dict(smoke, frame_start=1, frame_end=3, start_sec=.033, end_sec=.12)]
        self.assertEqual(validate_submission(value,irregular,'1/1000',.16), [])

    def test_full_view_evidence_and_submission_provenance_required(self):
        for key, changed in [('attest_full_view',False), ('evidence_ref',''),
                             ('submitted_at_utc',None), ('reviewer_person_id',None)]:
            with self.subTest(key=key):
                value=record();value[key]=changed
                self.assertTrue(self.validate(value))

    def test_source_binding_and_source_pts_not_fps_are_required(self):
        value=record();value['events'][0]['source_sha256']='b'*64
        self.assertIn('INTERVAL_SOURCE_SHA_MISMATCH',self.validate(value))
        value=record();value['events'][0]['timestamp_provenance']='FRAME_FPS'
        self.assertIn('EVENT_TIMESTAMP_NOT_SOURCE_PTS',self.validate(value))
        value=record();value['events'][0]['start_sec']=.21
        self.assertIn('EVENT_FRAME_SOURCE_TIME_MISMATCH',self.validate(value))

    def test_bounds_and_missing_pts_fail_closed(self):
        for changes in [{'frame_end':50}, {'start_sec':float('nan')}, {'end_sec':2.1},
                        {'frame_start':True}, {'frame_end':5}]:
            with self.subTest(changes=changes):
                value=record();value['events'][0].update(changes)
                self.assertTrue(self.validate(value))
        self.assertIn('INVALID_SOURCE_TIMELINE',validate_submission(record(),[],'1/1000',2.))

    def test_overlaps_and_duplicate_events_require_review(self):
        value=record();value['events'].append(copy.deepcopy(value['events'][0]))
        errors=self.validate(value)
        self.assertIn('DUPLICATE_EVENT_INTERVAL',errors)
        self.assertIn('MISSING_OR_DUPLICATE_EVENT_ID',errors)
        self.assertIn('SAME_CLASS_OVERLAP_REQUIRES_REVIEW',errors)

    def test_uncertainty_is_not_promoted_and_no_event_is_not_negative_exposure(self):
        value=record();value['uncertain_intervals']=[dict(value['events'][0],label='UNCERTAIN',event_id='UNCERTAIN_FIXTURE')]
        self.assertIn('UNCERTAIN_INTERVALS_NOT_RESOLVED',self.validate(value))
        value['events']=[];value['uncertain_intervals']=[]
        self.assertEqual(self.validate(value), [])
        self.assertNotIn('verified_negative',value)

    def test_duplicate_conflicts_quarantine_all_without_modifying_human_intervals(self):
        a=record();b=copy.deepcopy(a);b['video_id']='SYNTHETIC_B';b['events'][0]['start_sec']=.24
        originals=copy.deepcopy([a,b])
        excluded,ledger=duplicate_policy([b,a])
        self.assertEqual(set(excluded),{'SYNTHETIC_A','SYNTHETIC_B'})
        self.assertIsNone(ledger[0]['canonical_video_id'])
        self.assertEqual([a,b],originals)

    def test_identical_alias_uuid_does_not_count_as_another_source(self):
        a=record();b=copy.deepcopy(a);b['video_id']='SYNTHETIC_B';b['events'][0]['event_id']='DIFFERENT_PRIVATE_UUID'
        excluded,ledger=duplicate_policy([b,a])
        self.assertEqual(set(excluded),{'SYNTHETIC_B'})
        self.assertEqual(ledger[0]['canonical_video_id'],'SYNTHETIC_A')

    def test_conflicting_uncertainty_or_human_decision_never_promotes_alias_to_verified(self):
        a=record();b=copy.deepcopy(a);b['video_id']='SYNTHETIC_B'
        for changes in [{'uncertain_intervals':[dict(b['events'][0],label='UNCERTAIN')]},
                        {'decision':'UNCERTAIN'}]:
            with self.subTest(changes=changes):
                excluded,ledger=duplicate_policy([a,dict(b,**changes)])
                self.assertEqual(set(excluded),{'SYNTHETIC_A','SYNTHETIC_B'})
                self.assertIsNone(ledger[0]['canonical_video_id'])

    def test_localized_issues_never_publish_private_uuids(self):
        value=record();value['events'].append(dict(value['events'][0],event_id='PRIVATE_NEVER_PUBLISH'))
        issues=local_review_issues([value])
        self.assertEqual(len(issues),1)
        self.assertNotIn('PRIVATE_NEVER_PUBLISH',str(issues))
        self.assertNotIn('reviewer',str(issues))


if __name__=='__main__':
    unittest.main()
