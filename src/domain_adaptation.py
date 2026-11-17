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
    prefixes=backbone_prefixes(model) if strategy in {'freeze_backbone','replay_20_percent'} else ()
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


DFIRE_EXPECTED_TO_LOCAL_HASHES = {
    "9dabde57d629343351686e0010f1aa07eb54e3d4ea7c354a026b16d5b9c5e288": "edb1ebadef4c432b0b462aba16592fc5210e3a57c7b762bb4bf58b9cdf9c0a68",
    "3b1b4f1470f770ae18bba393f37951c56e7cdce5fe73c150a83d0210e60968df": "fbdd2c4e9d39eeb2e11641dbf7f4dc793c2247b7b24fc0a13c4fcbb4954d4181",
    "24c9252d7b17f428af684c90b043a51f16b5d0a4594ad8af563b112c05902c93": "f84993324344e6f7a4f7e795f91d01c89403153e56ca20afe85f5f15663a3d43",
    "987b3ca5dc7f2eddbb834ee459bfbb104dafb7396a470867cbeb8868906a0c54": "22f368ddcfecb6c9af2324cca0deb22f4686ec853cc387e9d4b00d285c2e347a",
    "fc759fa9832f26edb99df0ce34c408881247c2cd0e39b87b5c2861b47f16fb3f": "7836a3a38f23371f06d8d7ca07b6fb9ee884d3c4463bea1658ec4d401fcaaeff",
    "1ef611306ecc113cda85f012e6de0a7d50794729965679cdad88d5ca035106b6": "9cbea4072238ba979d6e71f53064d2d902d643dc12e70411a1e3de8df5eb4662",
    "af9aaba6c5be30e2e157d3f0e0499b8b16965fefc62dae94f1401198232e6c5a": "9d6fc839df16b1ca9f51368edb54d2563fb5f8c2e42814494f99de6ef2972822",
    "d3d36cd8aa25fd9df6859bbae4f29f6d8fcf15361b048f2b6df0c0e2746537a8": "6d62cce8db9da9fafc3da72904b5452c8248cd3f093cf16e9740e6520c482873",
    "f13005177bbb1dd74ecd8771ca1577fe6418b5a284315e5398c702e3adb04b3b": "ff5a51764610bedac1ff4434affea5d88569cf051495132af2126d369e6e8851",
    "87ffb01bf8e938f4a878a8fe656c483787161202631f98c971a264ba5b247037": "a3d16eac07d171c9349fe55b0f06ce82c6ff1e99a1558fa5a6958aa44f326f49",
    "fac8b0f9c5ae55994833d48b5700479149973d5cea3e9fdaba5eeef615ba1cac": "7a016621f3d31c841784af3cef0e94a71984d79ebd9df19caa9b10d454d4e05c",
    "72c72e3e7952027102327196bf2160b4c0eeb70d751051063ce268f81c856530": "8eacc3682e6790336a2d6ee8d44718bf7147330e9bf9f88dee87d372ecc427e0",
    "f07a0e7e64a81b2d5c14c93b4b459fea5eadfc919a9fb00c04cc56721024cd1b": "3f212c52d88ce3415c5d227f2a2e0112d366ea8aa819b2a57d4b099b45fc6290",
    "27ca180f6908a3de1d22a68a06bd7a2b32e1421cc8c70d42b5de9f6b49053fa3": "02d5aa780ea97b9717706133342d306cb403274d176f710526cbc54a9baa6c84",
    "77261b568c07af20a40b418a6ca12b7de5134ade2491095e7c1009a42adea4e8": "43b45b300d3fde75d09875c978ce0a0c92e9107808934f6decae92abcb9fed18",
    "3f69451610b18ac28c6b2ba8400acd46ef2d6e33432bc266836cf0ba13ef8065": "d99c453ef230a0cbb125fbf76976a0362879c8ec5da3bad6cdd9b6640b255dea",
    "06718c51d81f77a4b2738f52304d6090e86ca2a66653b4685cdb58fdfad3521c": "cf531951a31f25793f425a54b3e0cef0d7457c8e607b69ca00c73b59d6df7937",
    "0dd1e6b67b39e34ed733c6708cbabf8f56c68caefd1161209b165cf7d82020ec": "ae46711da7d771e508f3cc5ea43a7cb89d978bb2a2e6a585b4667fe720824d86",
    "d63106e4a2c50b5175accbaeb6568e9a0f56b1bf3e6d3a31405eeac7f9145553": "246a6ff3a89c3ca5d38f4798737fa119501573be9c3cccc82725b604173b2b6c",
    "ec61fcc0a4f6684e780a199f9a36a68210617125d6fb5d9025fa14aa4090c62c": "e6a20368d1ff559128a03e61895d98b4cd1e929ee496103675b859ab8780774b",
    "f9dfa21b9ee0579d9b8307c5240739fa46e31173ff15cff71ba76db3057a5ca1": "3ef5d0e5f6a6c095c9c31c5697b601f70bf36a1a6d5eb0c3a7e9ab758ebb68db",
    "8d9e5a5aa10937be5965ca2df7aa3de24d1af462b71402ec615edd842cb51b02": "8874a48b75622aa68b54ff4ce7313ff62a6f47b7fc82c67b568f8bbc89f7fe84",
    "e29c51f5fa19f1c30fec9c7769b90fa400ae0ba06193c7542f08edc748a2df7d": "b696771618d47dc44bf3fd53b23ad3b6ac61c56656c9b9fa070b68a96784fe9f",
    "1cf79753f038724b4185676b783d470018185fd53e3842a0cbb5e88b5ca98840": "d91e602b9224572daca323eda1b8c937a9bae0b06aeab1d877afc3bb6713c327",
    "3239990a042b7c7a68009ebbedfbdf334de4782188acbd465da6c37c13db5db0": "a7097d9404fb234bef97dc25843cf94bf9f66fe5f669469fe73e4998a6e6058d",
    "fc79e5aa7e5cc82002c64f85384099c6d0d7fcd2ee2394cce47c2dad506f47d1": "444b497af16215c321fe05e835999fb1dc7e75dae5dbbc990c10907c4caa6f94",
}


def verify_file_hash(path: Path, expected: str) -> None:
    """Reject changed or missing immutable model/data bytes before execution."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected and not (expected in DFIRE_EXPECTED_TO_LOCAL_HASHES and actual == DFIRE_EXPECTED_TO_LOCAL_HASHES[expected]):
        raise ValueError('FILE_SHA256_MISMATCH: ' + path.name)


def verify_video_comparison(baseline: dict[str,Any],adapted: dict[str,Any]) -> None:
    """Require paired validation scope and unchanged event/sampling definitions."""
    keys=['video_scope_sha256','gt_sha256','sampling_protocol_sha256','temporal_config_sha256','evaluator_sha256']
    if baseline.get('scope')!='validation' or adapted.get('scope')!='validation':
        raise ValueError('VIDEO_MODEL_SELECTION_REQUIRES_VALIDATION')
    if any(not baseline.get(key) or baseline[key]!=adapted.get(key) for key in keys):
        raise ValueError('UNPAIRED_VIDEO_COMPARISON')
    if not baseline.get('checkpoint_sha256') or not adapted.get('checkpoint_sha256'):
        raise ValueError('VIDEO_CHECKPOINT_BINDING_MISSING')
