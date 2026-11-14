"""Audit author split lists and create blocked, grouped bbox candidate manifests."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from ground_truth_studio.bbox_service import load_queue,QUEUE
from ground_truth_studio.integrity_service import sha256


def prepare(raw: Path,out: Path) -> dict:
    """Authenticate files without inventing absent images, labels or human approvals."""
    if not out.is_relative_to(ROOT/'runs/benchmarks/week5_domain_adaptation'):
        raise ValueError('Unsafe output directory')
    out.mkdir(parents=True,exist_ok=False)
    files=sorted(p for p in raw.rglob('*') if p.is_file()) if raw.exists() else []
    inventory={p.relative_to(raw).as_posix():sha256(p) for p in files}
    images=[p for p in files if p.suffix.lower() in {'.jpg','.png','.jpeg','.bmp'}]
    labels=[p for p in files if p.suffix.lower()=='.txt' and p.parent.name=='labels']
    lists={p.name:[s.strip() for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
           for p in files if p.parent==raw and p.suffix.lower()=='.txt'}
    train=set(lists.get('dfire_train.txt',[]));test=set(lists.get('dfire_test.txt',[]))
    legacy=ROOT/'runs/benchmarks/week5_domain_adaptation/dataset_candidate_20261114_73034262'
    legacy_files=list(csv.DictReader((legacy/'source_files.csv').open(encoding='utf-8')))
    changed=[r['relative_path'] for r in legacy_files if sha256(ROOT/'dataset'/r['relative_path'])!=r['sha256']]
    if changed:raise ValueError('LEGACY_SOURCE_HASH_MISMATCH')
    legacy_names={Path(r['relative_path']).name for r in legacy_files if r['relative_path'].endswith('.jpg')}
    rows=load_queue(ROOT)
    for row in rows:
        if sha256(ROOT/row['image_path'])!=row['image_sha256']:raise ValueError('QUEUE_IMAGE_HASH_MISMATCH')
        if row.get('label_path') and sha256(ROOT/row['label_path'])!=row['label_sha256']:raise ValueError('QUEUE_LABEL_HASH_MISMATCH')
    candidates=list(csv.DictReader((ROOT/QUEUE/'vn_image_candidates.csv').open(encoding='utf-8')))
    fields=['image_name','image_path','label_path','image_sha256','label_sha256','video_id','reserved_split',
            'current_candidate_source_sha256','scene_group_candidate','frame_source_lineage','approved_for_training']
    for name,prefixes in [('candidate_train',{'TRAIN_CANDIDATE','TRAIN_LEGACY_RESERVATION'}),
                          ('candidate_val',{'VALIDATION_CANDIDATE','VAL_LEGACY_RESERVATION'}),
                          ('reserved_test',{'INDEPENDENT_TEST_CANDIDATE','TEST_LEGACY_RESERVATION'}),
                          ('quarantined',{'BLOCKED_UNASSIGNED'})]:
        with (out/(name+'.csv')).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
            writer.writerows(r for r in candidates if r['reserved_split'] in prefixes)
    summary={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_url':'https://github.com/gaia-solutions-on-demand/DFireDataset',
        'source_url_basis':'USER_CONFIRMED; local split files not content-authenticated against author archive',
        'archive':None,'archive_sha256':None,'author_version':None,'raw_path':str(raw),
        'raw_directory_exists':raw.exists(),'raw_files':len(files),'raw_images':len(images),'raw_labels':len(labels),
        'raw_sha256':inventory,'provided_list_counts':{k:len(v) for k,v in lists.items()},
        'provided_train_test_name_overlap':sorted(train&test),
        'legacy_filename_overlap_with_author_lists':len(legacy_names&(train|test)),
        'legacy_files_rehashed_unchanged':len(legacy_files),'legacy_identity':'DFIRE_IDENTITY_NOT_VERIFIED',
        'DFIRE_READY':False,'DFIRE_STATUS':'BLOCKED_DATASET_MISSING' if not images or not labels else 'DFIRE_IDENTITY_NOT_VERIFIED',
        'SPLIT_READY':False,'VN_BBOX_READY':False,'WEEK5_TRAINING_READY':False,
        'queue_unique_images':len(rows),'priority_images':sum(r['priority'] for r in rows),
        'approved_images':0,'pending_images':len(rows),'approved_training_rows':0,
        'split_reservations':dict(Counter(r['reserved_split'] for r in candidates)),
        'unresolved_cross_split_pixel_pairs':75,'semantic_mapping_verified':False,
        'lineage_status':'LINEAGE_NOT_VERIFIED','checkpoint_sha256':sha256(ROOT/'runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt'),
        'resource_pilot':None,'baseline_map':None,'training_executed':False,
        'yaml_created':False,'labels_remapped':False,'approved_export_created':False,
        'blockers':['DFIRE_IMAGES_LABELS_MISSING','SEMANTIC_MAPPING_NOT_VERIFIED','GROUP_RELATIONS_UNRESOLVED',
                    'NO_HUMAN_BBOX_APPROVALS','LINEAGE_NOT_VERIFIED','NO_MEASURED_RESOURCE_PILOT']}
    (out/'readiness_gates.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'dataset_manifest.json').write_text(json.dumps({'status':'BLOCKED_NOT_A_TRAINABLE_DATASET','rows':[],
        'class_mapping':{'0':'smoke','1':'fire'},'split_locked':False,'leakage_review_complete':False,
        'prior_manifest_sha256':sha256(ROOT/'data/vn_pos_preannotation_split_manifest.json'),
        'blockers':summary['blockers']},indent=2),encoding='utf-8')
    assert all(sha256(raw/name)==value for name,value in inventory.items())
    print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=Path(r'D:\workspace\datasets\DFire_official_raw'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();prepare(args.raw.resolve(),args.output_dir.resolve())
