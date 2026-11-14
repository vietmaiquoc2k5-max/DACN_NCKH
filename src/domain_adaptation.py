"""Fail-closed dataset, parameter, replay and metric contracts for Week 5."""
from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

STRATEGIES={'full_finetune','freeze_backbone','replay_20_percent'}


def validate_class_mapping(names: Any) -> None:
    """Require smoke=0/fire=1 without silently remapping legacy annotations."""
    mapping=dict(enumerate(names)) if isinstance(names,list) else {int(k):v for k,v in names.items()}
    if {k:str(v).upper() for k,v in mapping.items()}!={0:'SMOKE',1:'FIRE'}:
        raise ValueError('CLASS_MAPPING_MISMATCH')


def read_yolo_boxes(path: Path) -> list[tuple[float,...]]:
    """Check YOLO bbox syntax, finite coordinates and six-decimal edge rounding.

    Args:
        path: Existing label file. Empty is syntactically valid, not human approval.

    Returns:
        Parsed boxes without editing or promoting the annotation.
    """
    rows=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            cls,x,y,w,h=map(float,line.split())
            if cls not in (0,1) or not all(math.isfinite(v) for v in [x,y,w,h]) or not (0<w<=1 and 0<h<=1):raise ValueError()
            if min(x-w/2,y-h/2)<-1e-6 or max(x+w/2,y+h/2)>1+1e-6:raise ValueError()
        except ValueError as error:
            raise ValueError('INVALID_YOLO_BBOX_OR_BOUNDS: '+path.name) from error
        rows.append((cls,x,y,w,h))
    return rows


def validate_dataset_rows(rows: list[dict[str,Any]], *, require_human: bool=True) -> None:
    """Reject image/source/scene leakage and event-to-bbox promotion.

    Args:
        rows: Image records with immutable source/label identity and split.
        require_human: Require explicit per-frame bbox signoff with hash bindings.
    """
    if not rows:raise ValueError('BBOX_GROUND_TRUTH_REQUIRED')
    groups: dict[tuple[str,str],set[str]]=defaultdict(set)
    for row in rows:
        if row['split'] not in {'train','val','test'}:raise ValueError('INVALID_DATASET_SPLIT')
        source_field='source_video_sha256' if require_human or row.get('source_video_sha256') else 'source_group_id'
        for field in ['image_sha256',source_field,'scene_group_id']:
            if not row.get(field):raise ValueError('MISSING_DATASET_LINEAGE')
            groups[(field,row[field])].add(row['split'])
        if require_human:
            if row.get('lineage_verified') is not True:
                raise ValueError('FRAME_SOURCE_LINEAGE_NOT_VERIFIED')
            evidence=row.get('bbox_review') or {}
            if not (evidence.get('status')=='SUBMITTED_BBOX_REVIEW' and evidence.get('attest_frame_review') is True and evidence.get('reviewer_id') and evidence.get('reviewed_at_utc') and evidence.get('evidence_ref') and evidence.get('image_sha256')==row['image_sha256'] and evidence.get('label_sha256')==row.get('label_sha256') and evidence.get('source_video_sha256')==row['source_video_sha256']):
                raise ValueError('BBOX_GROUND_TRUTH_REQUIRED')
            if row.get('uncertain') is True:raise ValueError('UNCERTAIN_FRAME_NOT_TRAINING_GT')
            if row.get('bbox_count')==0 and evidence.get('negative_frame_approved') is not True:
                raise ValueError('EMPTY_LABEL_REQUIRES_NEGATIVE_APPROVAL')
    if any(len(splits)>1 for splits in groups.values()):raise ValueError('IMAGE_SOURCE_OR_SCENE_SPLIT_LEAKAGE')


def backbone_prefixes(model: torch.nn.Module) -> tuple[str,...]:
    """Read actual model YAML topology, never assume freeze=10 means backbone."""
    count=len(model.yaml['backbone'])
    if count<=0:raise ValueError('BACKBONE_TOPOLOGY_UNAVAILABLE')
    return tuple(f'model.{index}.' for index in range(count))


def configure_parameters(model: torch.nn.Module,strategy: str) -> dict[str,Any]:
    """Set learnable parameter coverage, preserving the fixed DFL projection."""
    if strategy not in STRATEGIES:raise ValueError('UNKNOWN_ADAPTATION_STRATEGY')
    prefixes=backbone_prefixes(model) if strategy=='freeze_backbone' else ()
    frozen=[];trainable=[]
    for name,parameter in model.named_parameters():
        active=not (name.startswith(prefixes) or '.dfl.' in name)
        parameter.requires_grad_(active)
        if not active:parameter.grad=None
        (trainable if active else frozen).append(name)
    if not trainable:raise ValueError('NO_TRAINABLE_PARAMETERS')
    freeze_backbone_statistics(model,prefixes)
    return {'strategy':strategy,'backbone_prefixes':list(prefixes),'frozen_parameter_names':frozen,
            'trainable_parameter_names':trainable,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'total_parameters':sum(p.numel() for p in model.parameters()),'fixed_projection_policy':'Ultralytics DFL is fixed for all strategies'}


def freeze_backbone_statistics(model: torch.nn.Module,prefixes: tuple[str,...]) -> None:
    """Keep frozen backbone BatchNorm buffers fixed after each model.train() call."""
    for name,module in model.named_modules():
        if any(name==prefix.rstrip('.') or name.startswith(prefix) for prefix in prefixes):module.eval()


def restrict_optimizer(optimizer: torch.optim.Optimizer,model: torch.nn.Module) -> None:
    """Exclude frozen tensors and optimizer state while retaining scheduler group slots."""
    owned={id(p) for p in model.parameters()}
    for group in optimizer.param_groups:
        if any(id(p) not in owned for p in group['params']):raise ValueError('FOREIGN_OPTIMIZER_PARAMETERS')
        removed=[p for p in group['params'] if not p.requires_grad]
        group['params']=[p for p in group['params'] if p.requires_grad]
        for parameter in removed:optimizer.state.pop(parameter,None)
    if not any(group['params'] for group in optimizer.param_groups):raise ValueError('EMPTY_TRAINING_OPTIMIZER')
    actual=[id(p) for group in optimizer.param_groups for p in group['params']]
    expected={id(p) for p in model.parameters() if p.requires_grad}
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('INCOMPLETE_OR_DUPLICATE_OPTIMIZER_COVERAGE')


def state_digest(model: torch.nn.Module,prefixes: tuple[str,...]) -> str:
    """Hash frozen parameters and buffers, including BatchNorm statistics."""
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        if name.startswith(prefixes):
            value=tensor.detach().cpu().contiguous()
            digest.update((name+str(value.dtype)+str(tuple(value.shape))).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def replay_selection(vn_train: list[dict[str,Any]],dfire: list[dict[str,Any]],seed: int) -> dict[str,Any]:
    """Choose fixed-per-run, without-replacement D-Fire train-only replay IDs."""
    if not vn_train or any(row['split']!='train' for row in vn_train):raise ValueError('INVALID_VN_TRAIN_REPLAY_SCOPE')
    if any(row['split']!='train' for row in dfire):raise ValueError('DFIRE_REPLAY_MUST_BE_TRAIN_ONLY')
    identities=[row['image_sha256'] for row in vn_train+dfire]
    if len(identities)!=len(set(identities)):raise ValueError('REPLAY_CONTENT_OVERLAP_OR_DUPLICATION')
    count=math.ceil(len(vn_train)/4)
    if count>len(dfire):raise ValueError('DFIRE_TRAIN_TOO_SMALL_FOR_WITHOUT_REPLACEMENT_REPLAY')
    chosen=random.Random(seed).sample(sorted(dfire,key=lambda row:row['image_sha256']),count)
    fraction=count/(len(vn_train)+count)
    return {'selected':chosen,'seed':seed,'vn_presentations':len(vn_train),'dfire_presentations':count,
            'actual_replay_fraction':fraction,'rounding_difference_from_point2':fraction-.2,
            'replacement':False,'repetition_policy':'SAME_SELECTION_EACH_EPOCH; EACH_SELECTED_IMAGE_ONCE_PER_EPOCH',
            'definition':'ceil(V/4) D-Fire train image presentations in V+ceil(V/4) combined epoch'}


def validate_resume(receipt: dict[str,Any],expected: dict[str,Any]) -> None:
    """Require a real resumable checkpoint and unchanged strategy/data/config/source."""
    for key in ['strategy','seed','dataset_sha256','baseline_sha256','config_sha256']:
        if receipt.get(key)!=expected.get(key):raise ValueError('TRAINING_RESUME_BINDING_MISMATCH')
    if receipt.get('status') not in {'RUNNING','INTERRUPTED'} or not receipt.get('last_checkpoint_sha256'):
        raise ValueError('NO_REAL_RESUMABLE_CHECKPOINT')


def extract_map(metrics: Any) -> dict[str,Any]:
    """Read actual Ultralytics AP aggregates; absent classes remain unavailable."""
    box=metrics.box
    indices=list(box.ap_class_index)
    per_class={name:{'map50':None,'map50_95':None} for name in ['SMOKE','FIRE']}
    for position,index in enumerate(indices):
        if int(index) not in (0,1):raise ValueError('EVALUATION_CLASS_MAPPING_MISMATCH')
        per_class[{0:'SMOKE',1:'FIRE'}[int(index)]]={'map50':float(box.ap50[position]),'map50_95':float(box.ap[position])}
    if not indices:
        return {'map50':None,'map50_95':None,'per_class':per_class,'status':'NO_EVALUABLE_GT_CLASSES'}
    values=[float(box.map50),float(box.map)]+[v for row in per_class.values() for v in row.values() if v is not None]
    if any(not math.isfinite(v) or not 0<=v<=1 for v in values):
        raise ValueError('NONFINITE_OR_INVALID_MAP')
    return {'map50':float(box.map50),'map50_95':float(box.map),'per_class':per_class,'status':'MEASURED'}


def map_delta(adapted: float | None,baseline: float | None) -> float | None:
    """Preserve missing evidence rather than turning it into zero improvement."""
    return None if adapted is None or baseline is None else adapted-baseline


def select_adapted_model(candidates: list[dict[str,Any]]) -> dict[str,Any]:
    """Select only measured validation results with paired video/retention evidence."""
    required=['vn_map50_95','dfire_map50_95','video_recall','event_recall','far_per_hour','delay_sec',
              'baseline_video_recall','baseline_event_recall','baseline_far_per_hour','checkpoint_sha256']
    ready=[]
    for row in candidates:
        if row.get('scope')!='validation':raise ValueError('TEST_RESULTS_CANNOT_SELECT_MODEL')
        if row.get('status')!='MEASURED' or any(row.get(k) is None for k in required):continue
        if row.get('runtime_feasible') is not True:continue
        if any(not math.isfinite(row[k]) for k in required if k!='checkpoint_sha256'):continue
        if row['video_recall']<row['baseline_video_recall']-.05 or row['event_recall']<row['baseline_event_recall']-.05 or row['delay_sec']>10 or row['far_per_hour']>row['baseline_far_per_hour']:continue
        ready.append(row)
    if not ready:return {'status':'NO_ELIGIBLE_MEASURED_ADAPTED_MODEL','selected':None}
    ranked=sorted(ready,key=lambda r:(-r['vn_map50_95'],-r['dfire_map50_95'],-r['video_recall'],r['far_per_hour'],r['delay_sec'],r['checkpoint_sha256']))
    return {'status':'VALIDATION_SELECTED_NOT_TEST_ACCEPTANCE','selected':ranked[0]}


def verify_file_hash(path: Path, expected: str) -> None:
    """Reject changed or missing immutable model/data bytes before execution."""
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=expected:raise ValueError('FILE_SHA256_MISMATCH: '+path.name)


def verify_video_comparison(baseline: dict[str,Any],adapted: dict[str,Any]) -> None:
    """Require paired validation scope and unchanged event/sampling definitions."""
    keys=['video_scope_sha256','gt_sha256','sampling_protocol_sha256','temporal_config_sha256','evaluator_sha256']
    if baseline.get('scope')!='validation' or adapted.get('scope')!='validation':
        raise ValueError('VIDEO_MODEL_SELECTION_REQUIRES_VALIDATION')
    if any(not baseline.get(key) or baseline[key]!=adapted.get(key) for key in keys):
        raise ValueError('UNPAIRED_VIDEO_COMPARISON')
    if not baseline.get('checkpoint_sha256') or not adapted.get('checkpoint_sha256'):
        raise ValueError('VIDEO_CHECKPOINT_BINDING_MISSING')
