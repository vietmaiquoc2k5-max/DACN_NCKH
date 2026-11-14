"""Synthetic-only corpus audit regressions; never create human annotations."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scripts import audit_week5_dataset_candidate as candidate_audit


class TestDatasetCandidateAudit(unittest.TestCase):
    def run_case(self, invalid: bool=False) -> dict:
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/'dataset'
            images=source/'images/train/images'
            labels=source/'images/train/labels'
            images.mkdir(parents=True);labels.mkdir()
            pixels=np.zeros((16,16,3),np.uint8)
            cv2.imwrite(str(images/'synthetic.rf.a.jpg'),pixels)
            (labels/'synthetic.rf.a.txt').write_text('0 .5 .5 0 .1' if invalid else '0 .5 .5 .2 .2')
            cv2.imwrite(str(source/'images/unlabelled.jpg'),pixels)
            (root/'data_legacy.yaml').write_text('path: dataset\ntrain: images/train/images\nval: images/train/images\nnames: {0: fire, 1: smoke}\n')
            (root/'data_dfire.yaml').write_text('path: datasets/dfire\ntrain: train/images\nval: val/images\nnames: {0: smoke, 1: fire}\n')
            output=root/'runs/benchmarks/week5_domain_adaptation/synthetic'
            with patch.object(candidate_audit,'ROOT',root), patch.object(candidate_audit.subprocess,'check_output',return_value='SYNTHETIC_HEAD'):
                result=candidate_audit.audit(source,output)
            self.assertEqual(json.loads((output/'candidate_audit.json').read_text())['images'],2)
            return result

    def test_unlabelled_image_never_becomes_negative_or_training_approval(self):
        result=self.run_case()
        self.assertEqual(result['images_without_label'],1)
        self.assertFalse(result['missing_labels_treated_as_negative'])
        self.assertFalse(result['local_files_authenticated_against_author_manifest'])
        self.assertEqual(result['training_status'],'BLOCKED')
        self.assertTrue(result['original_files_unchanged'])

    def test_split_alias_and_reversed_mapping_are_explicit(self):
        result=self.run_case()
        self.assertTrue(result['legacy_train_equals_validation'])
        self.assertFalse(result['class_mapping_matches_baseline'])
        self.assertEqual(result['corpus_identity_status'],'NOT VERIFIED')

    def test_invalid_box_is_reported_without_repair(self):
        result=self.run_case(invalid=True)
        self.assertEqual(len(result['invalid_label_files']),1)
        self.assertFalse(result['labels_remapped_or_created'])


if __name__=='__main__':unittest.main()
