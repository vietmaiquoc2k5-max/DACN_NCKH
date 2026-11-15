"""Synthetic safety/lineage/approval tests; never write into real Studio storage."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

from ground_truth_studio.bbox_export_adapter import export_approvals, validate_lineage
from ground_truth_studio.bbox_queue import describe_queue, filter_queue, paginate
from ground_truth_studio.bbox_snapshot import committed_reviews, snapshot_studio
from ground_truth_studio.integrity_service import sha256
from scripts.audit_week5_readiness import bbox_errors
from src.domain_adaptation import configure_parameters, freeze_backbone_statistics, state_digest
from src.frame_lineage import match_derived_image, resolve_matches, verify_video_frames
from test_domain_adaptation import TinyModel


class TestWeek5Readiness(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()

    def synthetic_review(self, index=0):
        private=self.root/'private';folder=private/'bbox_qa/versions'/str(index)
        folder.mkdir(parents=True)
        image=self.root/f'image{index}.jpg'
        image.write_bytes(cv2.imencode('.jpg',np.full((16,16,3),index,dtype=np.uint8))[1].tobytes())
        label=folder/'labels.txt';label.write_text('0 .5 .5 .4 .4\n')
        row={'version_id':str(index),'image_sha256':sha256(image),'image_path':image.name,
             'label_sha256':sha256(label),'label_version_path':str(label),'reviewer_id':'SYNTHETIC',
             'reviewed_at_utc':'2026-01-01T00:00:00Z','decision':'APPROVE','status':'SUBMITTED_BBOX_REVIEW',
             'class_mapping':['smoke','fire'],'attest_frame_review':True,'evidence_ref':'SYNTHETIC_ONLY',
             'boxes':[{'class_id':0,'x':.5,'y':.5,'w':.4,'h':.4}]}
        (folder/'review.json').write_text(json.dumps(row))
        (folder/'review.sha256').write_text(sha256(folder/'review.json'))
        return row

    def test_34_immutable_reviews_backup_while_sqlite_writer_is_active(self):
        rows=[self.synthetic_review(i) for i in range(34)]
        private=self.root/'private';dbpath=private/'ground_truth_studio.sqlite3'
        with closing(sqlite3.connect(dbpath)) as db:
            db.execute('PRAGMA journal_mode=WAL');db.execute('CREATE TABLE events(id INTEGER)')
        ready=threading.Event();stop=threading.Event();errors=[]
        def writer():
            try:
                with closing(sqlite3.connect(dbpath)) as db:
                    while not stop.is_set():
                        db.execute('INSERT INTO events VALUES(1)');db.commit();ready.set();stop.wait(.005)
            except Exception as error:errors.append(error)
        thread=threading.Thread(target=writer);thread.start();self.assertTrue(ready.wait(5))
        try:receipt=snapshot_studio(private,self.root/'snapshot')
        finally:stop.set();thread.join(5)
        self.assertFalse(errors);self.assertEqual(receipt['bbox_versions'],34)
        self.assertEqual(rows,sorted(committed_reviews(private),key=lambda r:int(r['version_id'])))
        for name,digest in receipt['checksums'].items():
            self.assertEqual(sha256(self.root/'snapshot'/name),digest)
        self.assertEqual(len(committed_reviews(self.root/'snapshot',original_private=private)),34)

    def test_inflight_uncommitted_version_does_not_break_reader(self):
        row=self.synthetic_review();folder=self.root/'private/bbox_qa/versions/inflight'
        folder.mkdir();(folder/'review.json').write_text('{')
        self.assertEqual(committed_reviews(self.root/'private'),[row])
        (folder/'review.sha256').write_text('invalid')
        with self.assertRaisesRegex(ValueError,'METADATA_HASH'):committed_reviews(self.root/'private')

    def queue(self):
        (self.root/'empty.txt').write_text('')
        (self.root/'positive.txt').write_text('0 .5 .5 .2 .2')
        return {str(i):{'image_sha256':str(i),'image_path':f'{i}.jpg','image_name':f'VN_{i}.jpg',
                       'label_path':'empty.txt' if i<129 else 'positive.txt',
                       'video_id':f'VIDEO_{i%2}','priority':i<129,'empty_label':str(i<129),
                       'aliases':[{'image_path':f'alias_{i}.jpg'}]} for i in range(1922)}

    def test_all_129_searchable_and_filter_counts_do_not_duplicate(self):
        rows=describe_queue(self.queue(),{},self.root)
        filtered=filter_queue(rows,suspicious=True,empty=True,priority=True)
        self.assertEqual(len(filtered),129)
        for row in filtered:
            found=filter_queue(rows,query=row['image_id'])
            self.assertIn(row,found)
        self.assertEqual(len(filter_queue(rows,query='VIDEO_1')),961)
        self.assertEqual(len(filter_queue(rows,query='alias_12.jpg')),1)

    def test_pagination_covers_all_ids_once_and_never_mutates_history(self):
        history={'0':{'decision':'APPROVE'},'1':{'decision':'REJECT'}};before=json.dumps(history)
        rows=describe_queue(self.queue(),history,self.root)
        pending=filter_queue(rows,status='PENDING');seen=[]
        _,_,pages=paginate(pending,1,25)
        for page in range(1,pages+1):seen.extend(paginate(pending,page,25)[0])
        self.assertEqual(len(seen),1920);self.assertEqual(len({r['image_sha256'] for r in seen}),1920)
        self.assertEqual(json.dumps(history),before)
        self.assertEqual(paginate(pending,999,25)[1],pages)
        self.assertEqual(len(filter_queue(rows,status='APPROVED')),1)

    def test_empty_label_and_missing_source_do_not_autoapprove(self):
        rows=describe_queue(self.queue(),{},self.root)
        self.assertEqual(len(filter_queue(rows,investigation=True)),1922)
        self.assertTrue(all(r['review_status']=='PENDING' for r in rows))
        self.assertIn('SOURCE_NOT_FOUND',rows[0]['investigation_reasons'])

    def test_exact_transform_reproduction_and_changed_image(self):
        rng=np.random.default_rng(42);frame=rng.integers(0,255,(40,32,3),dtype=np.uint8)
        path=self.root/'frame.jpg';cv2.imwrite(str(path),frame,[cv2.IMWRITE_JPEG_QUALITY,95])
        self.assertEqual(match_derived_image(frame,path)['jpeg_quality'],95)
        cv2.imwrite(str(path),np.zeros_like(frame),[cv2.IMWRITE_JPEG_QUALITY,95])
        self.assertIsNone(match_derived_image(frame,path))

    def test_ambiguous_source_is_never_selected(self):
        a={'source_video_sha256':'a','source_frame_index':1,'status':'VERIFIED_DERIVED_IMAGE'}
        b=dict(a,source_video_sha256='b')
        self.assertEqual(resolve_matches([a,a]),a)
        self.assertEqual(resolve_matches([a,b])['status'],'AMBIGUOUS_SOURCE')

    def test_missing_and_changed_source_block_before_decoder(self):
        (self.root/'scripts').mkdir();(self.root/'scripts/process_new_vn_batches.py').write_text('# synthetic')
        row={'image_sha256':'x','image_path':'x.jpg','source_frame_index_declared':'0','current_candidate_source_sha256':'bad'}
        source=self.root/'source.avi'
        self.assertEqual(verify_video_frames(self.root,source,[row],'unused')[0]['status'],'SOURCE_NOT_FOUND')
        source.write_bytes(b'changed')
        self.assertEqual(verify_video_frames(self.root,source,[row],'unused')[0]['status'],'SOURCE_HASH_MISMATCH')

    def test_bbox_issue_classification_never_edits_labels(self):
        path=self.root/'labels.txt';body='2 .5 .5 .1 .1\n0 nan .5 .2 .2\n1 .5 .5 0 .2\n0 0 .5 .2 .2\n'
        path.write_text(body);reasons={r['reason'] for r in bbox_errors(path)}
        self.assertEqual(reasons,{'CLASS_ID','NON_FINITE','ZERO_OR_NEGATIVE_AREA','OUT_OF_BOUNDS'})
        self.assertEqual(path.read_text(),body)

    def test_approval_pending_lineage_export_preserves_originals(self):
        record=self.synthetic_review();original=Path(record['label_version_path']).read_bytes()
        queue={record['image_sha256']:{'reserved_split':'TRAIN_CANDIDATE'}}
        result=export_approvals(self.root,[record],queue,self.root/'export')
        self.assertEqual(result['human_approved_count'],1);self.assertEqual(result['rows'],[])
        self.assertEqual(result['excluded_rows'][0]['status'],'APPROVED_PENDING_LINEAGE')
        self.assertEqual(Path(record['label_version_path']).read_bytes(),original)

    def proof(self,record):
        source=self.root/'video.avi';source.write_bytes(b'SYNTHETIC_SOURCE_ONLY')
        proof={'status':'VERIFIED_DERIVED_IMAGE','image_sha256':record['image_sha256'],
               'source_video_path':source.name,'source_video_sha256':sha256(source),'source_frame_index':0,
               'source_pts_ticks':0,'time_base':'1/25','match_evidence':'EXACT_ENCODED_IMAGE_SHA256',
               'transformation':{'jpeg_quality':95}}
        directory=self.root/'lineage';directory.mkdir()
        (directory/(record['image_sha256']+'.json')).write_text(json.dumps(proof))
        return proof,directory

    def test_valid_approval_lineage_split_adapter_can_emit_real_training_schema(self):
        record=self.synthetic_review();proof,directory=self.proof(record)
        parent=self.root/'parent.csv';parent.write_text('SYNTHETIC_PARENT')
        split=self.root/'split.json';split.write_text(json.dumps({'split_locked':True,'leakage_review_complete':True,
            'parent_manifest_path':parent.name,'parent_manifest_sha256':sha256(parent),
            'rows':[{'image_sha256':record['image_sha256'],'source_video_sha256':proof['source_video_sha256'],
                     'scene_group_id':'SYNTHETIC_SCENE','split':'train','provenance_verified':True,'provenance_evidence_ref':'SYNTHETIC'}]}))
        queue={record['image_sha256']:{'reserved_split':'TRAIN_CANDIDATE'}}
        result=export_approvals(self.root,[record],queue,self.root/'eligible',lineage_dir=directory,split_manifest=split)
        self.assertEqual(result['approved_for_training'],1)
        self.assertEqual(result['rows'][0]['split'],'train')
        self.assertNotIn('source_video_sha256',record)
        queue[record['image_sha256']]['reserved_split']='INDEPENDENT_TEST_CANDIDATE'
        blocked=export_approvals(self.root,[record],queue,self.root/'reserved',lineage_dir=directory,split_manifest=split)
        self.assertEqual(blocked['rows'],[])

    def test_changed_source_and_no_integer_pts_reject_lineage(self):
        record=self.synthetic_review();proof,_=self.proof(record)
        validate_lineage(self.root,proof,record)
        with self.assertRaisesRegex(ValueError,'PTS_REQUIRED'):validate_lineage(self.root,dict(proof,source_pts_ticks=None),record)
        (self.root/'video.avi').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SOURCE_HASH_MISMATCH'):validate_lineage(self.root,proof,record)

    def test_replay_strategy_freezes_backbone_and_bn_buffers_during_optimizer_steps(self):
        torch.manual_seed(42);model=TinyModel();policy=configure_parameters(model,'replay_20_percent')
        prefixes=tuple(policy['backbone_prefixes']);self.assertTrue(prefixes)
        before=state_digest(model,prefixes);head=state_digest(model,('model.1.',))
        optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
        for _ in range(2):
            model.train();freeze_backbone_statistics(model,prefixes);optimizer.zero_grad()
            model(torch.randn(2,3,8,8)).square().mean().backward();optimizer.step()
        self.assertEqual(before,state_digest(model,prefixes));self.assertNotEqual(head,state_digest(model,('model.1.',)))


if __name__=='__main__':unittest.main()
