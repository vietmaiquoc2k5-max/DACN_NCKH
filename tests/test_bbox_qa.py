"""Synthetic-only bbox decisions; no production human approvals are generated."""
import json
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

from PIL import Image
from ground_truth_studio.bbox_service import BBoxReviewService,validate_boxes
from ground_truth_studio.integrity_service import sha256


class TestBBoxQA(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.image=self.root/'image.jpg';self.label=self.root/'original.txt'
        Image.new('RGB',(32,24),'gray').save(self.image);self.label.write_text('0 .5 .5 .2 .2\n')
        self.row={'image_sha256':sha256(self.image),'image_path':'image.jpg','label_path':'original.txt',
            'label_sha256':sha256(self.label),'aliases':[],'reserved_split':'TRAIN_CANDIDATE'}
        self.repository=Mock(settings=SimpleNamespace(repo=self.root,private=self.root/'private'))
        self.repository.actor.return_value={'id':'SYNTHETIC_ONLY','role':'annotator_01'}
        with patch('ground_truth_studio.bbox_service.load_queue',return_value=[self.row]):
            self.service=BBoxReviewService(self.repository)
        self.box=[dict(class_id=0,x=.5,y=.5,w=.2,h=.2)]

    def submit(self,**changes):
        params=dict(token='SYNTHETIC_TOKEN',identity=self.row['image_sha256'],boxes=self.box,
                    decision='APPROVE',evidence='SYNTHETIC_TEST_ONLY',attest=True)
        params.update(changes);return self.service.submit(**params)

    def test_authentication_and_signoff_are_required(self):
        for changes in [{'attest':False},{'evidence':''},{'decision':'AUTO_APPROVE'}]:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):self.submit(**changes)
        self.repository.actor.side_effect=PermissionError('Not authorized')
        with self.assertRaises(PermissionError):self.submit()

    def test_empty_label_needs_explicit_negative_and_never_grants_lineage(self):
        with self.assertRaisesRegex(ValueError,'NEGATIVE_APPROVAL'):self.submit(boxes=[])
        record=self.submit(boxes=[],negative=True)
        self.assertTrue(record['negative_frame_approved']);self.assertFalse(record['approved_for_training'])
        self.assertIsNone(record['source_video_sha256'])

    def test_label_version_is_new_and_rejection_supersedes_approval_in_export(self):
        before=self.label.read_bytes();a=self.submit();b=self.submit(decision='REJECT')
        self.assertNotEqual(a['version_id'],b['version_id'])
        self.assertEqual(self.label.read_bytes(),before)
        exported=json.loads((self.service.export('TOKEN')/'bbox_export.json').read_text())
        self.assertEqual(exported['human_approved_count'],0);self.assertEqual(exported['rows'],[])

    def test_approved_export_hash_and_own_identity_are_bound(self):
        record=self.submit()
        bundle=self.service.export('TOKEN');data=json.loads((bundle/'bbox_export.json').read_text())
        self.assertEqual(data['human_approved_count'],1);self.assertEqual(data['approved_for_training'],0)
        self.assertEqual(sha256(bundle/(record['image_sha256']+'.txt')),record['label_sha256'])
        self.repository.actor.return_value={'id':'OTHER_SYNTHETIC_ONLY','role':'annotator_02'}
        self.assertEqual(self.service.history('TOKEN'),[])

    def test_hash_changes_and_invalid_box_are_blocked(self):
        self.image.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'IMAGE_HASH_MISMATCH'):self.submit()
        for box in [dict(class_id=2,x=.5,y=.5,w=.1,h=.1),dict(class_id=0,x=0,y=.5,w=.2,h=.1),
                    dict(class_id=0,x=float('nan'),y=.5,w=.1,h=.1)]:
            with self.subTest(box=box):
                with self.assertRaises(ValueError):validate_boxes([box])

    def test_corrupt_and_missing_label_are_distinct_from_negative_approval(self):
        self.label.unlink()
        row,_,boxes=self.service.inspect('TOKEN',self.row['image_sha256'])
        self.assertEqual(boxes,[])
        with self.assertRaises(ValueError):self.submit(boxes=[])
        self.image.write_bytes(b'invalid jpeg');key=sha256(self.image)
        self.service.queue[key]=dict(self.row,image_sha256=key)
        with self.assertRaisesRegex(ValueError,'CORRUPT_IMAGE'):self.service.inspect('TOKEN',key)

    def test_tampered_approved_version_is_not_exported(self):
        record=self.submit();Path(record['label_version_path']).write_text('tampered')
        with self.assertRaisesRegex(ValueError,'APPROVED_LABEL_HASH'):self.service.export('TOKEN')

    def test_training_gate_requires_explicit_negative_even_with_other_signoff(self):
        from test_domain_adaptation import image_row
        from src.domain_adaptation import validate_dataset_rows
        row=image_row(1);row['bbox_count']=0
        with self.assertRaisesRegex(ValueError,'EMPTY_LABEL_REQUIRES_NEGATIVE'):
            validate_dataset_rows([row])
        row['bbox_review']['negative_frame_approved']=True
        validate_dataset_rows([row])

    def test_bbox_page_renders_actual_synthetic_pixels_and_controls(self):
        from streamlit.testing.v1 import AppTest
        script='''from ground_truth_studio.bbox_page import bbox_page
from unittest.mock import Mock
from types import SimpleNamespace
if __name__ == "__main__":
    repo=Mock(settings=SimpleNamespace(repo=None,private=None))
    bbox_page(repo,"SYNTHETIC_TOKEN")
'''
        main=sys.modules['__main__'];original_file=getattr(main,'__file__',None)
        try:
            with patch('ground_truth_studio.bbox_page.BBoxReviewService',return_value=self.service):
                self.row['image_name']='SYNTHETIC_IMAGE';self.service.queue[self.row['image_sha256']]=self.row
                app=AppTest.from_string(script,default_timeout=30).run()
                self.assertFalse(app.exception)
                self.assertTrue(any('Phê duyệt' in button.label for button in app.button))
                self.assertEqual(len(app.get('image')),1)
                app.text_input(key='bbox_search').set_value('NO_SUCH_SYNTHETIC_IMAGE').run()
                self.assertFalse(app.exception)
                self.assertFalse(any(s.label=='Ảnh cần review' for s in app.selectbox))
                app.checkbox(key='bbox_empty').check().run()
                self.assertEqual(app.text_input(key='bbox_search').value,'NO_SUCH_SYNTHETIC_IMAGE')
                self.assertFalse(self.service.private.exists())
        finally:
            if original_file is not None:main.__file__=original_file


if __name__=='__main__':unittest.main()
