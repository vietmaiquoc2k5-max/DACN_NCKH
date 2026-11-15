"""Build versioned, fail-closed split reservations and independent readiness gates."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ground_truth_studio.integrity_service import sha256
from scripts.audit_week5_readiness import PARENT, QUEUE, read_csv, write_json


def build(out: Path) -> dict:
    """Preserve all reservations and expose unresolved scopes without a trainable YAML."""
    reconciliation=json.loads((out/'vn_queue_reconciliation.json').read_text(encoding='utf-8'))
    candidates=reconciliation['rows'];near=read_csv(ROOT/QUEUE/'cross_split_pixel_candidates.csv')
    by_sha=defaultdict(list)
    for row in candidates:by_sha[row['image_sha256']].append(row)
    exact=[{'image_sha256':key,'members':[r['image_path'] for r in members],
            'splits':sorted({r['legacy_split'] for r in members})} for key,members in by_sha.items() if len(members)>1]
    scopes={};disposition={r['image_path']:r for r in read_csv(PARENT/'quarantine_dispositions.csv')}
    map_split={'TRAIN_CANDIDATE':'train','TRAIN_LEGACY_RESERVATION':'train',
               'VALIDATION_CANDIDATE':'val','VAL_LEGACY_RESERVATION':'val',
               'INDEPENDENT_TEST_CANDIDATE':'test','TEST_LEGACY_RESERVATION':'test'}
    for split in ('train','val','test','quarantine'):
        rows=[dict(r,eligible_for_training=False,eligible_for_evaluation=False,
                   blockers=disposition.get(r['image_path'],{}).get('reasons','LINEAGE_SCENE_APPROVAL_REQUIRED'))
              for r in candidates if map_split.get(r['reserved_split'],'quarantine')==split]
        payload={'status':'BLOCKED_CANDIDATE_RESERVATIONS_ONLY','split':split,'rows':rows,
                 'parent_manifest_sha256':sha256(ROOT/QUEUE/'vn_image_candidates.csv'),
                 'reason':'Preserve previous exposure/reservations; approval, lineage and scene gates remain separate',
                 'test_moved_to_train':False}
        write_json(out/f'vn_{split}_manifest.json',payload);scopes['VN_'+split]=len(rows)
    for split in ('train','val','test'):
        parent=PARENT/f'dfire_candidate_{split}.csv';rows=read_csv(parent)
        write_json(out/f'dfire_{split}_manifest.json',{'status':'BLOCKED_SCENE_AND_BASELINE_EXPOSURE_UNRESOLVED',
            'split':split,'rows':rows,'parent_manifest_sha256':sha256(parent),
            'reason':'Retain prior conservative quarantine; no test-to-train movement',
            'eligible_for_training':False,'eligible_for_evaluation':False})
        scopes['DFIRE_'+split]=len(rows)
    protocol=json.loads((ROOT/'configs/research/week5_execution_protocol_v2.json').read_text())
    checkpoint=ROOT/protocol['checkpoint'];checkpoint_ok=sha256(checkpoint)==protocol['checkpoint_sha256']
    gates={'DFIRE_READY':False,'VN_BBOX_READY':False,'SPLIT_READY':False,
           'BASELINE_CHECKPOINT_READY':checkpoint_ok,'RESOURCE_PILOT_READY':False,'WEEK5_TRAINING_READY':False}
    blockers=['DFIRE_CAMERA_SCENE_GROUPS_UNRESOLVED','BASELINE_TRAIN_MEMBERSHIP_NOT_AUTHENTICATED',
              'VN_APPROVED_LINEAGE_AND_INDEPENDENT_SPLITS_NOT_COMPLETE','VN_CROSS_SPLIT_NEAR_DUPLICATES_UNRESOLVED',
              'NO_VALID_SUBSET_FOR_RESOURCE_PILOT']
    write_json(out/'readiness_gates.json',{'gates':gates,'blockers':blockers,'scope_counts':scopes,
        'baseline_checkpoint_sha256':sha256(checkpoint),'training_runs':0,'completed_epochs':0,
        'baseline_vn_map':None,'baseline_dfire_map':None,'adaptation_metrics':None,'resource_pilot':None,
        'yaml_created':False,'status':'BLOCKED'})
    write_json(out/'dataset_manifest.json',{'status':'BLOCKED_NOT_A_TRAINABLE_DATASET','class_mapping':['smoke','fire'],
        'rows':[],'readiness_gates':gates,'blockers':blockers,'split_locked':False,'leakage_review_complete':False,
        'parent_manifest_sha256':sha256(PARENT/'dataset_manifest.json')})
    write_json(out/'split_reconciliation.json',{'vn_exact_duplicate_groups':exact,
        'vn_exact_duplicate_extra_paths':sum(len(r['members'])-1 for r in exact),
        'vn_exact_cross_split_groups':sum(len(r['splits'])>1 for r in exact),
        'vn_near_candidate_pairs':near,'scope_counts':scopes,
        'source_camera_scene_confirmation':'NOT_COMPLETE','ready_clean_subset':None,
        'prior_dfire_quarantined':len(read_csv(PARENT/'dfire_quarantined.csv'))})
    return gates


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(parser.parse_args().output)))
