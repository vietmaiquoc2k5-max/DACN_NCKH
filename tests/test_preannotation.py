"""Synthetic-only source/scene, exposure and provisional-manifest regression tests."""
from __future__ import annotations

import copy
import hashlib
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from ground_truth_studio.config import Settings
from ground_truth_studio.preannotation import split_metadata
from scripts.prepare_vn_pos_preannotation import inventory_record, validate_candidate_ids
from src.preannotation import (PROVISIONAL, canonical_hash, classify_exposure, fingerprint,
                              match_frames, ordered_chain, propose_split, scene_components,
                              similarity_candidate, validate_split)


def source(vid: str, group: str | None = None, split: str | None = None) -> dict:
    """Return metadata for a synthetic source, never a human annotation."""
    return {'video_id': vid, 'source_sha256': hashlib.sha256(vid.encode()).hexdigest(),
            'scene_group_id': group or vid, 'duration_seconds': 20.,
            'decode_status': 'NORMAL_EOF', 'legacy_splits': [split] if split else [],
            'historical_exposure_status': 'UNKNOWN_HISTORY'}


class TestContentScreening(unittest.TestCase):
    def textured(self, seed: int) -> np.ndarray:
        """Build structured synthetic texture, unrelated to real fire/smoke labels."""
        small = np.random.default_rng(seed).integers(0, 256, (24, 32, 3), dtype=np.uint8)
        return cv2.resize(small, (320, 240), interpolation=cv2.INTER_NEAREST)

    def test_reencoding_resolution_and_brightness_screen_is_not_exact_proof(self):
        originals = [self.textured(seed) for seed in [9, 19, 29]]
        copies = []
        for original in originals:
            ok, encoded = cv2.imencode('.jpg', cv2.resize(original, (160, 120)), [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.assertTrue(ok)
            copies.append(cv2.imdecode(encoded, cv2.IMREAD_COLOR))
        pairs = match_frames([fingerprint(image) for image in originals], [fingerprint(image) for image in copies])
        result = similarity_candidate(pairs)
        self.assertIsNotNone(result)
        self.assertEqual(result['status'], 'SCENE_RELATION_UNCERTAIN')
        self.assertFalse(result['human_confirmed'])
        self.assertNotEqual(hashlib.sha256(originals[0].tobytes()).digest(), hashlib.sha256(copies[0].tobytes()).digest())

    def test_unrelated_texture_and_blank_frames_do_not_create_scene_evidence(self):
        self.assertEqual(match_frames([fingerprint(self.textured(1))], [fingerprint(self.textured(77))]), [])
        blank = fingerprint(np.zeros((64, 64, 3), np.uint8))
        self.assertEqual(match_frames([blank], [blank]), [])

    def test_three_ordered_frames_are_sequence_hypothesis_only(self):
        frames = [fingerprint(self.textured(seed)) for seed in [4, 8, 16, 32]]
        result = similarity_candidate(match_frames(frames, frames[1:]))
        self.assertEqual(result['ordered_chain_length'], 3)
        self.assertEqual(result['relation'], 'TEMPORAL_SEQUENCE_CANDIDATE')
        self.assertFalse(result['human_confirmed'])

    def test_chain_cannot_reuse_one_frame(self):
        self.assertEqual(ordered_chain([{'first':0,'second':i} for i in range(5)]), 1)
        self.assertEqual(ordered_chain([{'first':i,'second':5-i} for i in range(5)]), 1)


class TestGroupSplit(unittest.TestCase):
    def test_transitive_uncertain_relations_cannot_cross_splits(self):
        mapping = scene_components(['A','B','C','D'], [('A','B'),('B','C')])
        rows = [source(k, mapping[k]) for k in mapping]
        assigned = propose_split(rows)
        self.assertEqual(len({r['proposed_split'] for r in assigned if r['video_id'] in {'A','B','C'}}), 1)
        self.assertNotEqual(mapping['A'], mapping['D'])

    def test_missing_relation_identity_is_rejected(self):
        with self.assertRaises(ValueError): scene_components(['A'], [('A','MISSING')])

    def test_group_hash_and_assignment_are_order_independent(self):
        ids = ['A','B','C','D','E','F']
        self.assertEqual(scene_components(ids,[('A','B')]),scene_components(reversed(ids),[('B','A')]))
        rows = [source(k) for k in ids]
        self.assertEqual(propose_split(rows, 42), propose_split(list(reversed(rows)), 42))

    def test_legacy_train_val_test_reservations_are_preserved(self):
        result = propose_split([source('A',split='train'),source('B',split='val'),source('C',split='test')])
        self.assertEqual([r['proposed_split'] for r in result],['TRAIN_CANDIDATE','VALIDATION_CANDIDATE','INDEPENDENT_TEST_CANDIDATE'])
        self.assertEqual(result[2]['candidate_readiness'],'BLOCKED')
        self.assertFalse(any(r['independence_verified'] for r in result))

    def test_legacy_same_scene_train_test_conflict_is_quarantined(self):
        result = propose_split([source('A','shared','train'),source('B','shared','test')])
        self.assertTrue(all(r['proposed_split']=='BLOCKED_UNASSIGNED' for r in result))
        self.assertTrue(all('CONFLICT' in r['reason'] for r in result))

    def test_corrupted_source_quarantines_its_whole_group(self):
        rows = [source('A','shared','train'),source('B','shared','train')]
        rows[1]['decode_status']='TRUNCATED_STREAM'; rows[1]['duration_seconds']=None
        self.assertTrue(all(r['proposed_split']=='BLOCKED_UNASSIGNED' for r in propose_split(rows)))

    def test_external_scene_overlap_cannot_become_test(self):
        row = source('A',split='test'); row['external_scene_overlap']=True
        self.assertEqual(propose_split([row])[0]['proposed_split'],'BLOCKED_UNASSIGNED')

    def test_duplicate_source_sha_cross_split_is_rejected(self):
        rows = propose_split([source('A',split='train'),source('B',split='test')])
        rows[1]['source_sha256']=rows[0]['source_sha256']
        with self.assertRaisesRegex(ValueError,'crosses'): validate_split(rows)

    def test_scene_crossing_validation_test_is_rejected(self):
        rows = propose_split([source('A',split='val'),source('B',split='test')])
        rows[1]['scene_group_id']=rows[0]['scene_group_id']
        with self.assertRaisesRegex(ValueError,'crosses'): validate_split(rows)

    def test_predictions_cannot_be_inputs_to_split(self):
        for key in ['predictions','detections','confidence','event_recall','ground_truth_events']:
            row=source('A');row[key]=[]
            with self.assertRaisesRegex(ValueError,'Inference'):propose_split([row])

    def test_missing_source_metadata_is_not_silently_assigned(self):
        row=source('A');row['source_sha256']=None
        with self.assertRaisesRegex(ValueError,'Missing'):propose_split([row])
        with self.assertRaises(ValueError):propose_split([source('A'),source('A')])

    def test_unknown_history_and_absent_evidence_are_not_proven_unseen(self):
        self.assertEqual(classify_exposure(set(),history_available=False),'UNKNOWN_HISTORY')
        self.assertEqual(classify_exposure(set(),history_available=True),'NO_EVIDENCE_OF_PRIOR_USE')
        self.assertEqual(classify_exposure({'KNOWN_EVALUATION_EXPOSURE','KNOWN_TRAIN'},history_available=False),'KNOWN_TRAIN')
        for row in propose_split([source('A',split='test')]):
            self.assertEqual(row['split_status'],PROVISIONAL)
            self.assertEqual(row['annotation_status'],'PENDING_HUMAN_REVIEW')
            self.assertEqual(row['independence_status'],'TEST_INDEPENDENCE_NOT_VERIFIED')

    def test_automatic_gt_or_independence_promotion_is_rejected(self):
        for key,value in [('annotation_status','VERIFIED'),('independence_verified',True),('split_status','LOCKED_INDEPENDENT_TEST')]:
            rows=propose_split([source('A')]);rows[0][key]=value
            with self.assertRaises(ValueError):validate_split(rows)


class TestInventoryAndManifest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'repo';self.root.mkdir()
        self.settings=Settings(self.root,Path(self.temp.name)/'private')
        (self.root/'data').mkdir()

    def test_exactly_expected_80_identities_and_mismatches(self):
        rows=[{'video_id':f'VN_POS_{i:02d}'} for i in range(1,81)]
        validate_candidate_ids(rows)
        for bad in [rows[:-1],rows+[rows[0]],rows[:-1]+[{'video_id':'VN_POS_81'}]]:
            with self.assertRaisesRegex(ValueError,'exactly'):validate_candidate_ids(bad)

    def test_missing_source_and_sha_mismatch_are_excluded(self):
        entry={'video_id':'VN_POS_01','relative_path':'source.mp4','sha256':'0'*64}
        row=inventory_record(self.root,entry,{},{});self.assertFalse(row['source_available']);self.assertIsNone(row['source_sha256']);self.assertFalse(row['technical_valid'])
        (self.root/'source.mp4').write_bytes(b'only a synthetic byte fixture')
        row=inventory_record(self.root,entry,{},{});self.assertFalse(row['source_sha_verified']);self.assertIsNone(row['duration_seconds'])

    def test_metadata_duration_cannot_become_verified_exposure(self):
        (self.root/'source.mp4').write_bytes(b'synthetic source bytes')
        digest=hashlib.sha256((self.root/'source.mp4').read_bytes()).hexdigest()
        entry={'video_id':'VN_POS_01','relative_path':'source.mp4','sha256':digest}
        audit={'source_sha256':digest,'technical_eligible':False,'decode_status':'TRUNCATED_STREAM','duration_seconds':9.}
        pixels={'source_sha256':digest,'metadata':{'format':{'duration':'300'},'streams':[{'codec_type':'video','r_frame_rate':'30/1','avg_frame_rate':'25/1'}]}}
        row=inventory_record(self.root,entry,audit,pixels)
        self.assertEqual(row['duration_container_seconds'],300);self.assertIsNone(row['duration_seconds']);self.assertEqual(row['fps_nominal'],30);self.assertEqual(row['fps_average_metadata'],25)

    def publish(self):
        catalog=self.root/'data/video_catalog_local_v2.json';catalog.write_text('{}')
        rows=propose_split([source('VN_POS_01',split='test')]);rows[0]['model_predictions']=['MUST_NOT_LEAK_TO_UI']
        value={'status':PROVISIONAL,'input_catalog_sha256':hashlib.sha256(catalog.read_bytes()).hexdigest(),'videos':rows}
        value['manifest_sha256']=canonical_hash(value)
        path=self.root/'data/vn_pos_preannotation_split_manifest.json';path.write_text(json.dumps(value))
        (self.root/'data/vn_pos_split_provenance.json').write_text(json.dumps({'manifest_json_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
        return path

    def test_studio_only_receives_split_allowlist_no_predictions_or_gt(self):
        self.publish();row=split_metadata(self.settings)['VN_POS_01']
        self.assertNotIn('model_predictions',row);self.assertNotIn('events',row)
        self.assertEqual(row['proposed_split'],'INDEPENDENT_TEST_CANDIDATE')

    def test_manifest_tampering_and_catalog_change_fail_closed(self):
        path=self.publish();original=path.read_bytes();path.write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError,'checksum'):split_metadata(self.settings)
        path.write_bytes(original);(self.root/'data/video_catalog_local_v2.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'catalog'):split_metadata(self.settings)

    def test_without_manifest_existing_studio_remains_available(self):
        self.assertEqual(split_metadata(self.settings),{})

    def test_windows_spawn_import_cannot_initialize_production_repository(self):
        app=Path(__file__).resolve().parent.parent/'ground_truth_studio/app.py'
        with patch('ground_truth_studio.review_repository.ReviewRepository',side_effect=AssertionError('No repository may open on spawn import')) as repository:
            runpy.run_path(str(app),run_name='__mp_main__')
            repository.assert_not_called()


if __name__=='__main__':
    unittest.main()
