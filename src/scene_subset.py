"""Evidence-bound scene decisions and deterministic candidate subset selection."""
from __future__ import annotations

from typing import Any


def validate_scene_decision(decision: dict, member_shas: set[str]) -> None:
    """Require explicit source evidence, rather than perceptual similarity alone."""
    if decision.get('classification') not in {'CONFIRMED_OVERLAP','CONFIRMED_INDEPENDENT'}:
        raise ValueError('SCENE_DECISION_UNRESOLVED')
    if set(decision.get('member_sha256',[])) != member_shas:
        raise ValueError('SCENE_MEMBER_BINDING_MISMATCH')
    if not decision.get('reviewer_id') or not decision.get('reviewed_at_utc'):
        raise ValueError('SCENE_REVIEW_SIGNOFF_REQUIRED')
    evidence=decision.get('evidence',[])
    accepted={'SOURCE_CAMERA_METADATA','SOURCE_FOOTAGE_METADATA','VISUAL_SCENE_REVIEW'}
    if not evidence or any(e.get('kind') not in accepted or not e.get('sha256') or not e.get('path') for e in evidence):
        raise ValueError('SCENE_EVIDENCE_REQUIRED; perceptual hashes alone are insufficient')


def build_subset(rows: list[dict], independent_pairs: set[frozenset[str]],
                 blocked_sources: set[str]) -> dict[str,Any]:
    """Select only fixed approved/lineage/source-qualified rows with proven split separation.

    Rows retain their original reservation. Every opposing source pair needs evidence;
    unknown relation blocks both sides, rather than inventing independent scene IDs.
    """
    pool=[];excluded=[]
    for row in sorted(rows,key=lambda r:r['image_sha256']):
        reasons=[]
        if row.get('split') not in {'train','val'}:reasons.append('RESERVED_TEST_OR_UNASSIGNED')
        if row.get('human_approved') is not True:reasons.append('HUMAN_APPROVAL_REQUIRED')
        if row.get('lineage_verified') is not True:reasons.append('LINEAGE_REQUIRED')
        if row.get('source_video_sha256') in blocked_sources:reasons.append('SOURCE_QUARANTINED')
        if not row.get('source_video_sha256'):reasons.append('SOURCE_IDENTITY_REQUIRED')
        if reasons:excluded.append({'image_sha256':row['image_sha256'],'reasons':reasons})
        else:pool.append(row)
    train={r['source_video_sha256'] for r in pool if r['split']=='train'}
    val={r['source_video_sha256'] for r in pool if r['split']=='val'}
    unresolved={frozenset((a,b)) for a in train for b in val
                if a==b or frozenset((a,b)) not in independent_pairs}
    # A valid smaller subset need not wait for every unrelated source pair.
    # Choose the evidenced seed with greatest row coverage, then SHA tie-break.
    coverage={source:sum(r['source_video_sha256']==source for r in pool) for source in train|val}
    seeds=sorted(((a,b) for a in train for b in val if a!=b and frozenset((a,b)) in independent_pairs),
                 key=lambda pair:(-sum(coverage[s] for s in pair),pair))
    selected_train=set();selected_val=set()
    if seeds:
        a,b=seeds[0];selected_train.add(a);selected_val.add(b)
        for source in sorted(train-{a},key=lambda s:(-coverage[s],s)):
            if all(source!=other and frozenset((source,other)) in independent_pairs for other in selected_val):
                selected_train.add(source)
        for source in sorted(val-{b},key=lambda s:(-coverage[s],s)):
            if all(source!=other and frozenset((source,other)) in independent_pairs for other in selected_train):
                selected_val.add(source)
    selected=[]
    for row in pool:
        accepted=selected_train if row['split']=='train' else selected_val
        if row['source_video_sha256'] not in accepted:
            excluded.append({'image_sha256':row['image_sha256'],'reasons':['SCENE_INDEPENDENCE_NOT_VERIFIED']})
        else:selected.append(row)
    ready={'train','val'}<={r['split'] for r in selected}
    return {'status':'SUBSET_DATA_CANDIDATE' if ready else 'BLOCKED_NO_PROVEN_OPPOSING_SCENES',
            'rows':selected,'excluded':excluded,'unresolved_source_pairs':[sorted(p) for p in sorted(unresolved,key=lambda x:sorted(x))],
            'train_count':sum(r['split']=='train' for r in selected),'val_count':sum(r['split']=='val' for r in selected),
            'test_moved_to_train':False,'selection_policy':'Approval + exact lineage + reservation + scene evidence; no model metrics'}


def training_membership(rows: list[dict]) -> list[dict]:
    """Record exact planned training bytes; test/validation cannot become baseline train."""
    if not rows or any(r.get('split')!='train' for r in rows):
        raise ValueError('BASELINE_MEMBERSHIP_TRAIN_ONLY')
    fields=('image_path','image_sha256','label_path','label_sha256','source_video_sha256','scene_group_id')
    if any(any(not r.get(k) for k in fields) for r in rows):raise ValueError('MEMBERSHIP_HASH_AND_PROVENANCE_REQUIRED')
    if len({r['image_sha256'] for r in rows})!=len(rows):raise ValueError('MEMBERSHIP_DUPLICATE')
    return [{k:r[k] for k in fields} for r in sorted(rows,key=lambda r:r['image_sha256'])]
