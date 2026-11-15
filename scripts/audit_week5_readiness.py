"""Reconcile current files and human review snapshots without changing source data."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ground_truth_studio.bbox_service import load_queue, QUEUE
from ground_truth_studio.bbox_snapshot import committed_reviews
from ground_truth_studio.config import Settings
from ground_truth_studio.integrity_service import sha256
from src.domain_adaptation import read_yolo_boxes

PARENT = ROOT / 'runs/benchmarks/week5_domain_adaptation/recovery_20261114_5df62d8a'


def read_csv(path: Path) -> list[dict]:
    """Read UTF-8 manifests, preserving original string fields."""
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: object) -> None:
    """Write a new audit artifact; caller owns the fresh output directory."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def bbox_errors(path: Path) -> list[dict]:
    """Classify each failing line; never clamp or edit original annotations."""
    errors = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        reason = None
        try:
            values = list(map(float, line.split()))
            if len(values) != 5:
                reason = 'FIELD_COUNT'
            elif not all(math.isfinite(v) for v in values):
                reason = 'NON_FINITE'
            else:
                c,x,y,w,h = values
                if c not in (0,1): reason = 'CLASS_ID'
                elif w <= 0 or h <= 0: reason = 'ZERO_OR_NEGATIVE_AREA'
                elif w > 1 or h > 1 or min(x-w/2,y-h/2) < -1e-6 or max(x+w/2,y+h/2) > 1+1e-6:
                    reason = 'OUT_OF_BOUNDS'
        except ValueError:
            reason = 'NON_NUMERIC'
        if reason:
            errors.append({'line':number, 'text':line, 'reason':reason})
    return errors


def reconcile(out: Path, private: Path) -> dict:
    """Compute actual set intersections and statuses, without exposing reviewer IDs."""
    candidates = read_csv(ROOT / QUEUE / 'vn_image_candidates.csv')
    general = read_csv(ROOT / QUEUE / 'bbox_annotation_queue.csv')
    priority = read_csv(ROOT / QUEUE / 'empty_label_review_129.csv')
    queue = {r['image_sha256']:r for r in load_queue(ROOT)}
    history = committed_reviews(private, original_private=Settings.load().private)
    latest = {r['image_sha256']:r for r in history}
    by_image_reviewers = defaultdict(dict)
    for review in history:
        by_image_reviewers[review['image_sha256']][review['reviewer_id']] = review
    dispositions = {r['image_sha256']:r for r in read_csv(PARENT / 'quarantine_dispositions.csv')}
    rows = []
    for row in candidates:
        item = dict(row)
        key = row['image_sha256']
        image, label = ROOT / row['image_path'], ROOT / row['label_path']
        image_ok = image.is_file() and sha256(image) == key
        label_ok = label.is_file() and sha256(label) == row['label_sha256']
        in_queue = key in queue
        decisions = {r['decision'] for r in by_image_reviewers[key].values()}
        state = 'CONFLICTING_REVIEWERS' if len(decisions)>1 else {'APPROVE':'APPROVED','REJECT':'REJECTED'}.get(next(iter(decisions),''),'PENDING')
        item.update(image_exists=image.exists(), image_hash_valid=image_ok, label_hash_valid=label_ok,
            label_current_empty=label_ok and not label.read_text(encoding='utf-8').strip(),
            queue_membership=in_queue, catalog_membership='CSV_BACKED_BBOX_CATALOG' if in_queue else 'NOT_IN_BBOX_QUEUE',
            queue_exclusion_reason='' if in_queue else 'OUTSIDE_EXISTING_SELECTED_BBOX_SCOPE; SEE_RESERVATION',
            source_exists=bool(row['source_path_candidate']) and (ROOT/row['source_path_candidate']).is_file(),
            priority=key in {r['image_sha256'] for r in priority}, current_human_status=state,
            stored_schema_status=latest.get(key,{}).get('status','NO_COMMITTED_VERSION'),
            disposition_reasons=dispositions.get(key,{}).get('reasons'),
            ui_initial_filter='ALL; explicitly searchable including Needs data investigation',
            ui_prior_filter='only own pending; no priority/empty/video search controls',
            ui_session_filter_state='NOT_CAPTURED; live browser state is not in backend storage')
        rows.append(item)
    pids={r['image_sha256'] for r in priority}; gids={r['image_sha256'] for r in general}
    suspicious=[r for r in rows if r['image_sha256'] in pids]
    summary={'candidate_image_ids':len(rows),'candidate_unique_sha':len({r['image_sha256'] for r in rows}),
        'general_queue_ids':len(general),'priority_ids':len(priority),'priority_unique_sha':len(pids),
        'queue_overlap_unique_sha':len(pids & gids),'deduplicated_queue':len(queue),
        'outside_queue':sum(not r['queue_membership'] for r in rows),
        'candidate_reservations':dict(Counter(r['reserved_split'] for r in rows)),
        'queue_reservations':dict(Counter(r['reserved_split'] for r in rows if r['queue_membership'])),
        'queue_human_statuses':dict(Counter(r['current_human_status'] for r in rows if r['queue_membership'])),
        'suspicious_human_statuses':dict(Counter(r['current_human_status'] for r in suspicious)),
        'missing_or_changed_images':sum(not r['image_hash_valid'] for r in rows),
        'missing_or_changed_labels':sum(not r['label_hash_valid'] for r in rows),
        'source_missing':sum(not r['source_exists'] for r in rows),
        'suspicious_found_in_queue':sum(r['queue_membership'] for r in suspicious),
        'review_versions':len(history),'review_unique_images':len(latest),
        'drafts':'No persistent bbox DRAFT state in current schema; incomplete unpublished directories not counted',
        'parent_manifest_sha256':sha256(ROOT/QUEUE/'vn_image_candidates.csv')}
    write_json(out/'vn_queue_reconciliation.json',{'summary':summary,'rows':rows})
    write_json(out/'suspicious_empty_labels.json',{'summary':summary,'rows':suspicious})
    print(json.dumps(summary),flush=True)
    return summary


def audit_dfire(out: Path, contacts: Path) -> dict:
    """Rehash all files, decode images, classify bbox errors and render suspect groups."""
    out=out.resolve();contacts=contacts.resolve()
    corpus=ROOT/'datasets/dfire_author_linked_kaggle_v1'
    source_files=read_csv(PARENT/'dfire_file_sha256.csv')
    mismatches=[]
    for number,row in enumerate(source_files,1):
        path=corpus/row['path']
        if not path.is_file() or sha256(path)!=row['sha256']:
            mismatches.append(row['path'])
        if number%5000==0:print(json.dumps({'dfire_files_hashed':number}),flush=True)
    if mismatches:
        write_json(out/'dfire_hash_mismatches.json',mismatches)
        raise ValueError('DFIRE_SOURCE_CHANGED')
    images=read_csv(PARENT/'dfire_image_manifest.csv'); invalid=[]; corrupt=[]
    for number,row in enumerate(images,1):
        if cv2.imread(str(ROOT/row['image_path'])) is None:corrupt.append(row['image_path'])
        errors=bbox_errors(ROOT/row['label_path'])
        if errors:invalid.append({'image_path':row['image_path'],'label_path':row['label_path'],
                                  'label_sha256':row['label_sha256'],'errors':errors})
        if number%5000==0:print(json.dumps({'dfire_images_decoded':number}),flush=True)
    write_json(out/'dfire_bbox_issues.json',invalid)
    write_json(out/'dfire_content_recheck.json',{'files_hashed':len(source_files),
        'images_decoded':len(images),'invalid_bbox_images':len(invalid),'corrupt_images':corrupt,
        'file_manifest_sha256':sha256(PARENT/'dfire_file_sha256.csv')})
    prior=json.loads((PARENT/'dfire_content_audit.json').read_text(encoding='utf-8'))
    groups=[];contacts.mkdir(parents=True,exist_ok=False)
    for number, group in enumerate(prior['cross_split_perceptual_candidate_groups']):
        members=group['images']; pixels=[cv2.imread(str(ROOT/p)) for p in members]
        exact_pixels=all(p is not None and p.shape==pixels[0].shape and np.array_equal(p,pixels[0]) for p in pixels)
        status='REENCODED_DUPLICATE_PIXEL_IDENTICAL' if exact_pixels else 'UNRESOLVED'
        sheets=[]
        for offset in range(0,len(members),12):
            chunk=members[offset:offset+12]; canvas=np.full((math.ceil(len(chunk)/3)*210,960,3),240,np.uint8)
            for j,path in enumerate(chunk):
                im=pixels[offset+j];h,w=im.shape[:2];scale=min(320/w,180/h)
                small=cv2.resize(im,(max(1,int(w*scale)),max(1,int(h*scale))))
                y,x=(j//3)*210,(j%3)*320
                canvas[y:y+small.shape[0],x:x+small.shape[1]]=small
                cv2.putText(canvas,Path(path).parts[-3]+'/'+Path(path).name,(x+3,y+198),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),1)
            target=contacts/f'group_{number:04d}_{offset//12:03d}.jpg'
            if not cv2.imwrite(str(target),canvas):raise OSError('CONTACT_SHEET_WRITE_FAILED')
            sheets.append({'path':target.relative_to(ROOT).as_posix(),'sha256':sha256(target)})
        groups.append({'group_id':f'DFIRE_{number:04d}','classification':status,'members':members,
                       'splits':group['splits'],'contact_sheets':sheets,'human_resolution':None})
    summary={'files_hash_verified':len(source_files),'image_count':len(images),'label_count':len(images),
        'valid_label_images':len(images)-len(invalid),'invalid_bbox_images':len(invalid),
        'error_line_categories':dict(Counter(e['reason'] for r in invalid for e in r['errors'])),
        'corrupt_images':corrupt,'suspected_groups':len(groups),
        'group_classifications':dict(Counter(g['classification'] for g in groups)),
        'exact_sha_duplicate_groups':sum(n>1 for n in Counter(r['image_sha256'] for r in images).values()),
        'source_mapping':['smoke','fire'],'source_files_changed':False,
        'parent_manifest_sha256':sha256(PARENT/'dfire_image_manifest.csv'),
        'camera_scene_independence':'UNRESOLVED; prior Camera0001 val/test evidence retained',
        'baseline_training_membership':'UNVERIFIED','split_ready':False}
    write_json(out/'dfire_bbox_issues.json',invalid)
    write_json(out/'dfire_group_review.json',groups)
    write_json(out/'dfire_reaudit.json',summary)
    print(json.dumps(summary),flush=True)
    return summary


def main() -> None:
    """Create a new immutable audit version; private snapshots stay outside Git."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--private-snapshot',type=Path,required=True)
    parser.add_argument('--contacts',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(2)
    reconcile(args.output,args.private_snapshot)
    audit_dfire(args.output,args.contacts)


if __name__=='__main__': main()
