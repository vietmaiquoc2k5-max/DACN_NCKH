"""Synthetic-only Week 5 contracts; these tests are not adaptation experiments."""
from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ultralytics.utils.metrics import Metric

from scripts.run_week5_domain_adaptation import validate_budget, validate_inputs

from src.domain_adaptation import (backbone_prefixes, configure_parameters,
                                   extract_map, freeze_backbone_statistics,
                                   map_delta, read_yolo_boxes, replay_selection,
                                   restrict_optimizer, select_adapted_model,
                                   state_digest, validate_class_mapping,
                                   validate_dataset_rows, validate_resume,
                                   verify_file_hash, verify_video_comparison)


class TinyModel(torch.nn.Module):
    """Small synthetic network with a backbone BN buffer and trainable head."""
    def __init__(self):
        super().__init__()
        self.yaml={'backbone':[['synthetic',1]]}
        self.model=torch.nn.Sequential(
            torch.nn.Sequential(torch.nn.Conv2d(3,4,1),torch.nn.BatchNorm2d(4),torch.nn.ReLU()),
            torch.nn.Conv2d(4,2,1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


def image_row(index: int, split: str='train') -> dict:
    digest=hashlib.sha256(str(index).encode()).hexdigest()
    row={'split':split,'image_sha256':digest,'source_video_sha256':digest,
         'label_sha256':'b'*64,'scene_group_id':'SCENE_'+digest,'lineage_verified':True}
    row['bbox_review']={'status':'SUBMITTED_BBOX_REVIEW','attest_frame_review':True,
                        'reviewer_id':'SYNTHETIC_ONLY','reviewed_at_utc':'SYNTHETIC_ONLY',
                        'evidence_ref':'SYNTHETIC_ONLY',
                        **{k:row[k] for k in ['image_sha256','label_sha256','source_video_sha256']}}
    return row


class TestDatasetContracts(unittest.TestCase):
    def test_event_only_and_pending_manifests_block_training_before_any_image_load(self):
        for strategy in ['full_finetune','freeze_backbone','replay_20_percent']:
            with self.subTest(strategy=strategy):
                with self.assertRaisesRegex(ValueError,'DATASET_NOT_READY'):
                    validate_inputs({'class_mapping':['smoke','fire'],
                        'status':'BLOCKED_NOT_A_TRAINABLE_DATASET','rows':[]},strategy)
        with self.assertRaisesRegex(ValueError,'GROUP_SPLIT'):
            validate_inputs({'class_mapping':['smoke','fire'],'status':'READY_FOR_TRAINING',
                             'rows':[]},'full_finetune')

    def test_resources_cannot_be_estimated_without_actual_bound_pilot(self):
        with self.assertRaisesRegex(ValueError,'PILOT_DATA_OR_CHECKPOINT'):
            validate_budget({},'a'*64,'b'*64,30)
        budget={'dataset_sha256':'a'*64,'baseline_sha256':'b'*64,
                'measured_epoch_seconds':None,'max_training_hours':3}
        with self.assertRaisesRegex(ValueError,'TIMED_PILOT'):
            validate_budget(budget,'a'*64,'b'*64,30)

    def test_boolean_lineage_without_native_source_receipt_is_not_training_approval(self):
        rows=[dict(image_row(1),corpus='VN'),dict(image_row(2,'val'),corpus='VN')]
        manifest={'class_mapping':['smoke','fire'],'status':'READY_FOR_TRAINING',
                  'split_locked':True,'leakage_review_complete':True,'rows':rows}
        with self.assertRaisesRegex(ValueError,'AUTHENTICATED_NATIVE_FRAME_LINEAGE'):
            validate_inputs(manifest,'full_finetune')

    def test_class_mapping_never_silently_reverses_legacy_classes(self):
        validate_class_mapping(['smoke','fire'])
        validate_class_mapping({'0':'SMOKE','1':'FIRE'})
        with self.assertRaisesRegex(ValueError,'CLASS_MAPPING'):validate_class_mapping(['fire','smoke'])

    def test_label_format_finite_bounds_and_empty_not_human_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'synthetic.txt'
            for bad in ['2 .5 .5 .2 .2','0 nan .5 .2 .2','0 0 .5 .4 .2',
                        '1 .5 .5 0 .2','1 .5 .5 .2','1 .95 .5 .2 .2']:
                with self.subTest(bad=bad):
                    path.write_text(bad,encoding='utf-8')
                    with self.assertRaisesRegex(ValueError,'INVALID_YOLO'):read_yolo_boxes(path)
            path.write_text('0 .5 .5 1 1\n1 .25 .25 .1 .1',encoding='utf-8')
            self.assertEqual(len(read_yolo_boxes(path)),2)
            path.write_text('',encoding='utf-8');self.assertEqual(read_yolo_boxes(path),[])
        row=image_row(1);row.pop('bbox_review')
        with self.assertRaisesRegex(ValueError,'BBOX_GROUND_TRUTH_REQUIRED'):validate_dataset_rows([row])

    def test_vn_and_dfire_group_leakage_cannot_cross_splits(self):
        for field in ['image_sha256','source_video_sha256','scene_group_id']:
            with self.subTest(field=field):
                rows=[image_row(1),image_row(2,'val')]
                rows[1][field]=rows[0][field]
                with self.assertRaisesRegex(ValueError,'SPLIT_LEAKAGE'):
                    validate_dataset_rows(rows,require_human=False)

    def test_event_gt_legacy_boolean_and_uncertain_do_not_approve_bboxes(self):
        row=image_row(1)
        validate_dataset_rows([row])
        for change in [{'lineage_verified':False},{'bbox_review':{'is_reviewed':True}},
                       {'bbox_review':{'status':'JOINT_HUMAN_CONSENSUS'}},{'uncertain':True}]:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):validate_dataset_rows([dict(row,**change)])
        row['bbox_review']['label_sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'BBOX_GROUND_TRUTH_REQUIRED'):validate_dataset_rows([row])
        with self.assertRaisesRegex(ValueError,'BBOX_GROUND_TRUTH_REQUIRED'):validate_dataset_rows([])

    def test_checkpoint_and_label_bytes_are_authenticated(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'synthetic.bin';path.write_bytes(b'original')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            verify_file_hash(path,digest)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'FILE_SHA256_MISMATCH'):verify_file_hash(path,digest)


class TestTrainingMechanics(unittest.TestCase):
    def test_full_finetune_covers_all_learnable_parameters_and_optimizer(self):
        model=TinyModel();policy=configure_parameters(model,'full_finetune')
        self.assertEqual(policy['trainable_parameters'],policy['total_parameters'])
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        restrict_optimizer(optimizer,model)
        before=state_digest(model,('model.',))
        model(torch.ones(2,3,8,8)).square().mean().backward();optimizer.step()
        self.assertNotEqual(before,state_digest(model,('model.',)))

    def test_real_optimizer_step_keeps_backbone_parameters_and_bn_buffers_fixed(self):
        torch.manual_seed(42)
        model=TinyModel();policy=configure_parameters(model,'freeze_backbone')
        prefixes=backbone_prefixes(model)
        before=state_digest(model,prefixes);head=state_digest(model,('model.1.',))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01)
        restrict_optimizer(optimizer,model)
        self.assertTrue(all(not p.requires_grad for n,p in model.named_parameters() if n.startswith(prefixes)))
        self.assertEqual(sum(p.numel() for g in optimizer.param_groups for p in g['params']),policy['trainable_parameters'])
        for _ in range(3):
            model.train();freeze_backbone_statistics(model,prefixes)
            optimizer.zero_grad();model(torch.randn(2,3,8,8)).square().mean().backward();optimizer.step()
        self.assertEqual(before,state_digest(model,prefixes))
        self.assertNotEqual(head,state_digest(model,('model.1.',)))

    def test_fixed_dfl_projection_is_not_a_finetune_parameter(self):
        model=TinyModel();model.model[1]=torch.nn.ModuleDict({'dfl':torch.nn.Linear(2,1,bias=False),'learnable':torch.nn.Linear(2,1)})
        policy=configure_parameters(model,'full_finetune')
        self.assertIn('model.1.dfl.weight',policy['frozen_parameter_names'])
        self.assertIn('model.1.learnable.weight',policy['trainable_parameter_names'])

    def test_optimizer_missing_head_coverage_is_rejected(self):
        model=TinyModel();configure_parameters(model,'full_finetune')
        optimizer=torch.optim.SGD(model.model[0].parameters(),lr=.1)
        with self.assertRaisesRegex(ValueError,'OPTIMIZER_COVERAGE'):restrict_optimizer(optimizer,model)

    def test_replay_is_deterministic_train_only_unique_and_reports_rounding(self):
        vn=[image_row(i) for i in range(5)];dfire=[image_row(i) for i in range(10,22)]
        a=replay_selection(vn,dfire,42);b=replay_selection(vn,list(reversed(dfire)),42)
        self.assertEqual(a,b);self.assertEqual(a['dfire_presentations'],2)
        self.assertAlmostEqual(a['actual_replay_fraction'],2/7)
        self.assertAlmostEqual(a['rounding_difference_from_point2'],2/7-.2)
        self.assertFalse(a['replacement'])
        self.assertNotEqual(a['selected'],replay_selection(vn,dfire,123)['selected'])
        self.assertEqual(replay_selection(vn[:4],dfire,42)['actual_replay_fraction'],.2)

    def test_replay_refuses_validation_overlap_and_inadequate_corpus(self):
        vn=[image_row(i) for i in range(8)]
        for dfire in [[image_row(10,'val')],[image_row(0),image_row(20)],[image_row(10)]]:
            with self.subTest(dfire=dfire):
                with self.assertRaises(ValueError):replay_selection(vn,dfire,42)

    def test_resume_requires_real_last_checkpoint_and_identical_bindings(self):
        expected={'strategy':'freeze_backbone','seed':42,'dataset_sha256':'a'*64,
                  'baseline_sha256':'b'*64,'config_sha256':'c'*64}
        receipt={**expected,'status':'INTERRUPTED','last_checkpoint_sha256':'d'*64}
        validate_resume(receipt,expected)
        for field in expected:
            broken=dict(receipt);broken[field]='CHANGED'
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError,'RESUME_BINDING'):validate_resume(broken,expected)
        with self.assertRaisesRegex(ValueError,'NO_REAL_RESUMABLE'):validate_resume(dict(receipt,last_checkpoint_sha256=None),expected)


class TestMetricsAndSelection(unittest.TestCase):
    def test_actual_ultralytics_map_aggregates_and_per_class_not_hardcoded_class_order(self):
        box=Metric();box.all_ap=np.array([[.8]*10,[.4]*10]);box.ap_class_index=np.array([1,0])
        result=extract_map(SimpleNamespace(box=box))
        self.assertAlmostEqual(result['map50'],.6);self.assertAlmostEqual(result['map50_95'],.6)
        self.assertAlmostEqual(result['per_class']['FIRE']['map50'],.8)
        self.assertAlmostEqual(result['per_class']['SMOKE']['map50_95'],.4)

    def test_missing_class_and_empty_metric_are_null_not_fabricated(self):
        box=Metric();box.all_ap=np.array([[.8]*10]);box.ap_class_index=np.array([1])
        result=extract_map(SimpleNamespace(box=box))
        self.assertIsNone(result['per_class']['SMOKE']['map50'])
        box.ap_class_index=[]
        self.assertIsNone(extract_map(SimpleNamespace(box=box))['map50'])
        self.assertIsNone(map_delta(None,.5));self.assertIsNone(map_delta(.6,None))
        self.assertAlmostEqual(map_delta(.6,.7),-.1)

    def test_video_comparison_requires_same_gt_scope_sampling_and_temporal_configuration(self):
        baseline={'scope':'validation','checkpoint_sha256':'a'*64,
                  **{k:'BOUND' for k in ['video_scope_sha256','gt_sha256','sampling_protocol_sha256',
                                        'temporal_config_sha256','evaluator_sha256']}}
        adapted=dict(baseline,checkpoint_sha256='b'*64)
        verify_video_comparison(baseline,adapted)
        for field in ['scope','video_scope_sha256','gt_sha256','sampling_protocol_sha256','temporal_config_sha256','evaluator_sha256']:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):verify_video_comparison(baseline,dict(adapted,**{field:'CHANGED'}))

    def candidate(self) -> dict:
        return {'scope':'validation','status':'MEASURED','vn_map50_95':.6,'dfire_map50_95':.5,
                'video_recall':.9,'event_recall':.9,'far_per_hour':2.,'delay_sec':3.,
                'baseline_video_recall':.9,'baseline_event_recall':.9,'baseline_far_per_hour':3.,
                'runtime_feasible':True,'checkpoint_sha256':'a'*64}

    def test_no_model_selected_without_complete_measured_validation_and_video_evidence(self):
        self.assertIsNone(select_adapted_model([])['selected'])
        row=self.candidate()
        for changes in [{'dfire_map50_95':None},{'status':'NOT_TRAINED'},{'event_recall':.8},
                        {'video_recall':.8},{'far_per_hour':4.},{'delay_sec':11.},{'runtime_feasible':False}]:
            with self.subTest(changes=changes):
                self.assertIsNone(select_adapted_model([dict(row,**changes)])['selected'])
        with self.assertRaisesRegex(ValueError,'TEST_RESULTS_CANNOT_SELECT'):select_adapted_model([dict(row,scope='test')])

    def test_selection_is_predeclared_and_deterministic_on_validation_only(self):
        a=self.candidate();b=dict(a,checkpoint_sha256='b'*64,dfire_map50_95=.55)
        self.assertEqual(select_adapted_model([a,b])['selected'],b)
        self.assertEqual(select_adapted_model([b,a])['selected'],b)


if __name__=='__main__':
    unittest.main()
