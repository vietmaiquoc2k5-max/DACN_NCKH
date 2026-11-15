"""Incremental live-review audit; reuse receipts and preserve test reservations."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ground_truth_studio.integrity_service import sha256
from scripts.audit_week5_readiness import read_csv,write_json,QUEUE


def baseline_membership(checkpoint: Path) -> dict:
    """Inspect trusted local checkpoint metadata without inferring missing membership."""
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    args=payload.get('train_args',{})
    return {'checkpoint_sha256':sha256(checkpoint),'epoch':payload.get('epoch'),
        'date':payload.get('date'),'version':payload.get('version'),
        'train_args_data':args.get('data'),'checkpoint_keys':sorted(payload.keys()),
        'status':'TRAINING_MEMBERSHIP_NOT_AUTHENTICATED',
        'reason':'Dataset path/training metrics do not bind the image bytes presented during training',
        'independent_evaluation_claimed':False,
        'fallback_protocol':{'status':'PROPOSED_NOT_EXECUTED','initialization':'New explicitly recorded pretrained checkpoint or random initialization',
            'required':['Locked source/scene-disjoint train/val/test with per-image SHA',
                        'Train new baseline only on locked train; select checkpoint on validation',
                        'Use same resulting baseline for A/B/C; keep reserved test sealed'],
            'reuse_historical_test_for_training':False}}


def scene_pairs(out: Path, contacts: Path) -> dict:
    """Resolve only exact decoded pixel equality; similarity remains a human queue."""
    contacts.mkdir(parents=True,exist_ok=False)
    pairs=read_csv(ROOT/QUEUE/'cross_split_pixel_candidates.csv');results=[]
    for index,row in enumerate(pairs):
        a,b=(cv2.imread(str(ROOT/row[key])) for key in ['image_a','image_b'])
        if a is None or b is None:raise ValueError('PAIR_IMAGE_MISSING')
        equal=a.shape==b.shape and np.array_equal(a,b)
        canvas=np.full((250,800,3),240,np.uint8)
        for i,(image,key) in enumerate([(a,'image_a'),(b,'image_b')]):
            h,w=image.shape[:2];scale=min(400/w,220/h);small=cv2.resize(image,(int(w*scale),int(h*scale)))
            canvas[:small.shape[0],i*400:i*400+small.shape[1]]=small
            cv2.putText(canvas,Path(row[key]).name,(i*400+2,240),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),1)
        path=contacts/f'VN_{index:03d}.jpg';cv2.imwrite(str(path),canvas)
        results.append(dict(row,classification='EXACT_DECODED_PIXEL_OVERLAP' if equal else 'UNRESOLVED',
            image_a_sha256=sha256(ROOT/row['image_a']),image_b_sha256=sha256(ROOT/row['image_b']),
            contact_sheet=path.relative_to(ROOT).as_posix(),contact_sheet_sha256=sha256(path),
            human_decision=None,priority='CROSS_SPLIT_EVALUATION_RISK',test_reservation_preserved=True))
    write_json(out/'vn_scene_pairs.json',results)
    parent=ROOT/'runs/benchmarks/week5_domain_adaptation/readiness_20261115_ed406cca_v2'
    groups=json.loads((parent/'dfire_group_review.json').read_text())
    queue=[{'group_id':g['group_id'],'members':g['members'],'contact_sheets':g['contact_sheets'],
            'decision':'UNRESOLVED','priority':'CROSS_SPLIT_EVALUATION_RISK',
            'required_evidence':'same footage/camera or unrelated decision with reviewer/time/evidence; retain reservations'} for g in groups]
    write_json(out/'human_scene_decision_queue.json',{'DFIRE':queue,'VN':results,
        'parent_dfire_groups_sha256':sha256(parent/'dfire_group_review.json'),
        'decisions_not_auto_approved':True})
    summary={'VN_pairs':len(results),'VN_classifications':dict(Counter(r['classification'] for r in results)),
        'DFIRE_groups_reused':len(groups),'DFIRE_automatic_exact_pixel_resolution':0,
        'DFIRE_unresolved':len(groups),'scene_independence_proven':False,
        'prior_visual_findings':json.loads((parent/'dfire_visual_observations.json').read_text()),
        'no_test_to_train_movement':True}
    write_json(out/'scene_resolution.json',summary)
    return summary


if __name__=='__main__':
    out=ROOT/'.tools/week5_unblock_2e4b73a5/audit';out.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    write_json(out/'baseline_membership.json',baseline_membership(ROOT/'runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt'))
    print(json.dumps(scene_pairs(out,ROOT/'.tools/week5_unblock_2e4b73a5/vn_contacts')))
