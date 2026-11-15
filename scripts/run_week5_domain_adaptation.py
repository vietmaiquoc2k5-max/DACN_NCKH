"""Gate real Week 5 training, record actual epochs and evaluate validation images.

No training or validation starts with the blocked preparation manifest. Training
also requires a measured pilot receipt, verified bbox provenance and locked splits.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from ground_truth_studio.config import confined
from ground_truth_studio.integrity_service import sha256
from src.domain_adaptation import (backbone_prefixes, configure_parameters, extract_map,
                                   freeze_backbone_statistics, read_yolo_boxes,
                                   replay_selection, restrict_optimizer, state_digest,
                                   validate_class_mapping, validate_dataset_rows,
                                   validate_resume, verify_file_hash)
from src.video_audit_v2 import write_json
from ground_truth_studio.bbox_export_adapter import validate_lineage

LOG=logging.getLogger(__name__)


def validate_inputs(manifest: dict[str,Any],strategy: str) -> tuple[list[dict],list[dict]]:
    """Reject event-only labels, unverified lineage and unresolved source leakage."""
    validate_class_mapping(manifest['class_mapping'])
    if manifest.get('status')!='READY_FOR_TRAINING':
        raise ValueError('DATASET_NOT_READY: '+str(manifest.get('blockers',[])))
    if manifest.get('split_locked') is not True or manifest.get('leakage_review_complete') is not True:
        raise ValueError('GROUP_SPLIT_AND_LEAKAGE_REVIEW_REQUIRED')
    vn=[r for r in manifest['rows'] if r.get('corpus')=='VN']
    dfire=[r for r in manifest['rows'] if r.get('corpus')=='DFIRE']
    if len(vn)+len(dfire)!=len(manifest['rows']):raise ValueError('UNKNOWN_DATASET_CORPUS')
    validate_dataset_rows(vn)
    if not {'train','val'}<={r['split'] for r in vn}:raise ValueError('VN_TRAIN_AND_VALIDATION_REQUIRED')
    if strategy=='replay_20_percent' and not dfire:raise ValueError('BLOCKED_DATASET_MISSING_DFIRE')
    if dfire:
        if manifest.get('dfire_annotation_origin')!='ORIGINAL_DFIRE_ANNOTATIONS' or not manifest.get('dfire_split_provenance_sha256'):
            raise ValueError('DFIRE_ORIGINAL_ANNOTATIONS_AND_SPLIT_PROVENANCE_REQUIRED')
        validate_dataset_rows(dfire,require_human=False)
        if not {'train','val'}<={r['split'] for r in dfire}:raise ValueError('DFIRE_TRAIN_AND_VALIDATION_REQUIRED')
    # Cross-corpus duplicates/groups also cannot become evaluation or replay leakage.
    groups: dict[tuple[str,str],set[str]]={}
    for row in vn+dfire:
        for field in ['image_sha256','scene_group_id','source_video_sha256','source_group_id']:
            if row.get(field):groups.setdefault((field,row[field]),set()).add(row['split'])
    if any(len(s)>1 for s in groups.values()):raise ValueError('CROSS_CORPUS_IMAGE_SOURCE_OR_SCENE_LEAKAGE')
    checked_sources=set()
    for row in vn:
        if not row.get('source_video_path') or not row.get('frame_lineage_receipt') or not row.get('frame_lineage_receipt_sha256'):
            raise ValueError('AUTHENTICATED_NATIVE_FRAME_LINEAGE_RECEIPT_REQUIRED')
        source=confined(ROOT,row['source_video_path'])
        if (str(source),row['source_video_sha256']) not in checked_sources:
            verify_file_hash(source,row['source_video_sha256'])
            checked_sources.add((str(source),row['source_video_sha256']))
        proof_path=confined(ROOT,row['frame_lineage_receipt'])
        verify_file_hash(proof_path,row['frame_lineage_receipt_sha256'])
        proof=json.loads(proof_path.read_text(encoding='utf-8'))
        validate_lineage(ROOT,proof,row)
        if any(
            proof.get(key)!=row.get(key) or row.get(key) is None
            for key in ['image_sha256','source_video_sha256','source_frame_index','source_pts_ticks','time_base']):
            raise ValueError('NATIVE_FRAME_LINEAGE_BINDING_MISMATCH')
        approval_path=confined(ROOT,row['bbox_approval_receipt'])
        verify_file_hash(approval_path,row['bbox_approval_receipt_sha256'])
        approval=json.loads(approval_path.read_text(encoding='utf-8'))
        if approval.get('decision')!='APPROVE' or any(
            approval.get(key)!=(row.get('bbox_review') or {}).get(key)
            for key in ['image_sha256','label_sha256','reviewer_id','reviewed_at_utc','attest_frame_review','negative_frame_approved']):
            raise ValueError('HUMAN_APPROVAL_RECEIPT_BINDING_MISMATCH')
    for row in vn+dfire:
        for field,path_field in [('image_sha256','image_path'),('label_sha256','label_path')]:
            verify_file_hash(confined(ROOT,row[path_field]),row[field])
        if cv2.imread(str(confined(ROOT,row['image_path']))) is None:raise ValueError('UNREADABLE_IMAGE')
        boxes=read_yolo_boxes(confined(ROOT,row['label_path']))
        if row['corpus']=='VN' and not boxes and (row.get('bbox_review') or {}).get('negative_frame_approved') is not True:
            raise ValueError('EMPTY_LABEL_REQUIRES_NEGATIVE_APPROVAL')
    return vn,dfire


def validate_budget(budget: dict[str,Any],dataset_hash: str,baseline_hash: str,epochs: int) -> None:
    """Require measured pilot provenance and a bounded prospective workload."""
    if budget.get('dataset_sha256')!=dataset_hash or budget.get('baseline_sha256')!=baseline_hash:
        raise ValueError('PILOT_DATA_OR_CHECKPOINT_BINDING_MISMATCH')
    seconds=budget.get('measured_epoch_seconds');limit=budget.get('max_training_hours')
    if not isinstance(seconds,(int,float)) or not isinstance(limit,(int,float)) or not math.isfinite(seconds) or not math.isfinite(limit) or min(seconds,limit)<=0:
        raise ValueError('TIMED_PILOT_AND_RESOURCE_BUDGET_REQUIRED')
    if not budget.get('pilot_log') or not budget.get('pilot_log_sha256'):
        raise ValueError('TIMED_PILOT_LOG_REQUIRED')
    verify_file_hash(confined(ROOT,budget['pilot_log']),budget['pilot_log_sha256'])
    if epochs*seconds*1.25>limit*3600:raise ValueError('ESTIMATED_TRAINING_EXCEEDS_RESOURCE_BUDGET')


def stage_rows(rows: list[dict],folder: Path,split: str) -> None:
    """Copy reviewed images and labels into a fresh run; leave originals untouched."""
    images=folder/'images'/split;labels=folder/'labels'/split
    images.mkdir(parents=True);labels.mkdir(parents=True)
    for row in rows:
        source=confined(ROOT,row['image_path'])
        name=row['corpus']+'_'+row['image_sha256']
        image=images/(name+source.suffix.lower());label=labels/(name+'.txt')
        if image.exists() or label.exists():raise ValueError('DUPLICATE_STAGED_IMAGE_IDENTITY')
        shutil.copyfile(source,image)
        shutil.copyfile(confined(ROOT,row['label_path']),label)
        verify_file_hash(image,row['image_sha256']);verify_file_hash(label,row['label_sha256'])


def data_config(path: Path,folder: Path) -> None:
    """Only expose training/validation paths; reserved tests never enter training."""
    with path.open('x',encoding='utf-8') as stream:
        yaml.safe_dump({'path':str(folder),'train':'images/train','val':'images/val',
                        'names':{0:'smoke',1:'fire'}},stream,sort_keys=False)


def execute(args: argparse.Namespace) -> int:
    """Run one strategy/seed only after all data, hash and resource gates pass."""
    output=args.output_dir.resolve()
    if not output.is_relative_to(ROOT/'runs/benchmarks/week5_domain_adaptation'):
        raise ValueError('Unsafe output directory')
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    if args.strategy=='replay_20_percent' and protocol.get('protocol_version')!='week5_domain_adaptation_v2_freeze_replay':
        raise ValueError('FREEZE_REPLAY_REQUIRES_VERSIONED_V2_PROTOCOL; legacy replay runs cannot resume with changed parameter policy')
    dataset=json.loads(args.dataset_manifest.read_text(encoding='utf-8'))
    expected={'strategy':args.strategy,'seed':args.seed,'dataset_sha256':sha256(args.dataset_manifest),
              'baseline_sha256':protocol['checkpoint_sha256'],'config_sha256':sha256(args.protocol)}
    if args.seed not in protocol['seeds'] or args.strategy not in protocol['strategies']:
        raise ValueError('UNDECLARED_STRATEGY_OR_SEED')
    checkpoint=confined(ROOT,protocol['checkpoint'])
    verify_file_hash(checkpoint,expected['baseline_sha256'])
    output.mkdir(parents=True,exist_ok=args.resume)
    receipt_path=output/'training_receipt.json'
    if args.resume:
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        validate_resume(receipt,expected)
        verify_file_hash(output/'fit/weights/last.pt',receipt['last_checkpoint_sha256'])
        state=torch.load(output/'fit/weights/last.pt',map_location='cpu',weights_only=False)
        if state.get('optimizer') is None or state.get('epoch',-1)<0:
            raise ValueError('LAST_CHECKPOINT_IS_NOT_RESUMABLE')
    else:
        receipt={**expected,'training_run_id':output.name,'status':'PREFLIGHT',
                 'epochs_actually_completed':0,'last_checkpoint_sha256':None,
                 'training_executed':False,'validation_history':[],'selected_model':None}
        write_json(receipt_path,receipt)
    try:
        vn,dfire=validate_inputs(dataset,args.strategy)
        if not dfire:
            raise ValueError('WEEK5_REQUIRES_DFIRE_READY_FOR_ALL_STRATEGIES')
        if not all((dataset.get('readiness_gates') or {}).get(key) is True
                   for key in ['DFIRE_READY','SPLIT_READY','VN_BBOX_READY']):
            raise ValueError('WEEK5_INDEPENDENT_READINESS_GATES_REQUIRED')
        if args.budget_receipt is None:raise ValueError('TIMED_PILOT_AND_RESOURCE_BUDGET_REQUIRED')
        budget=json.loads(args.budget_receipt.read_text(encoding='utf-8'))
        validate_budget(budget,expected['dataset_sha256'],expected['baseline_sha256'],protocol['epochs'])
        if args.resume:
            locked=json.loads((output/'locked_inputs.json').read_text(encoding='utf-8'))
            if locked['expected']!=expected or locked['budget_receipt_sha256']!=sha256(args.budget_receipt):
                raise ValueError('RESUME_LOCKED_INPUTS_CHANGED')
            for name,digest in locked['staged_input_sha256'].items():
                verify_file_hash(confined(output,name),digest)
        device='0' if torch.cuda.is_available() else 'cpu'
        if budget.get('device')!=device:raise ValueError('PILOT_DEVICE_MISMATCH')
        if not args.resume:
            train=[r for r in vn if r['split']=='train']
            replay=None
            if args.strategy=='replay_20_percent':
                replay=replay_selection(train,[r for r in dfire if r['split']=='train'],args.seed)
                train=train+replay['selected']
                write_json(output/'replay_selection.json',replay)
            stage_rows(train,output/'vn','train')
            stage_rows([r for r in vn if r['split']=='val'],output/'vn','val')
            data_config(output/'vn.yaml',output/'vn')
            if dfire:
                stage_rows([r for r in dfire if r['split']=='val'],output/'dfire','val')
                (output/'dfire/images/train').mkdir()
                # The train folder is empty: this config is validation-only and
                # is never passed to model.train or to replay selection.
                with (output/'dfire_eval.yaml').open('x',encoding='utf-8') as stream:
                    yaml.safe_dump({'path':str(output/'dfire'),'train':'images/train','val':'images/val',
                                    'names':{0:'smoke',1:'fire'}},stream)
            write_json(output/'locked_inputs.json',{'expected':expected,'protocol':protocol,
                       'dataset_manifest_sha256':expected['dataset_sha256'],
                       'budget_receipt_sha256':sha256(args.budget_receipt),
                       'staged_input_sha256':{p.relative_to(output).as_posix():sha256(p)
                           for folder in ['vn','dfire'] for p in (output/folder).rglob('*') if p.is_file()}})
            baseline=YOLO(str(checkpoint))
            receipt['baseline_validation']={}
            for corpus,config in [('vn','vn.yaml'),('dfire','dfire_eval.yaml')]:
                receipt['baseline_validation'][corpus]=extract_map(baseline.val(
                    data=str(output/config),split='val',imgsz=protocol['imgsz'],
                    conf=.001,iou=.7,device=device,workers=0,project=str(output),
                    name='baseline_'+corpus,exist_ok=False,plots=False)) if (output/config).exists() else None
            write_json(receipt_path,receipt)
        before=time.monotonic()
        model=YOLO(str(output/'fit/weights/last.pt') if args.resume else str(checkpoint))
        validate_class_mapping(model.names)
        prefixes=backbone_prefixes(model.model) if args.strategy in {'freeze_backbone','replay_20_percent'} else ()
        frozen_before=state_digest(model.model.float(),prefixes) if prefixes else None
        epoch_files: list[str]=[]

        class EvidenceTrainer(DetectionTrainer):
            """Count actual file presentations after augmentation-free batch loading."""
            def build_optimizer(self,model: torch.nn.Module,*positional: Any,**keywords: Any) -> torch.optim.Optimizer:
                configure_parameters(model,args.strategy)
                optimizer=super().build_optimizer(model,*positional,**keywords)
                # Do this before optimizer state restoration, as well as after
                # setup, so resume sees the same parameter group cardinalities.
                restrict_optimizer(optimizer,model)
                return optimizer

            def preprocess_batch(self,batch: dict) -> dict:
                epoch_files.extend(Path(p).stem for p in batch['im_file'])
                return super().preprocess_batch(batch)

        def started(trainer: Any) -> None:
            policy=configure_parameters(trainer.model,args.strategy)
            restrict_optimizer(trainer.optimizer,trainer.model)
            receipt.update(status='RUNNING',training_executed=True,device=str(trainer.device),
                           parameter_policy=policy,optimizer=protocol['optimizer'],lr0=protocol['lr0'],
                           frozen_state_before=frozen_before)
            write_json(receipt_path,receipt)

        def batch_start(trainer: Any) -> None:
            if time.monotonic()-before>budget['max_training_hours']*3600:
                raise TimeoutError('TRAINING_RESOURCE_BUDGET_EXCEEDED')
            freeze_backbone_statistics(trainer.model,prefixes)

        def batch_end(trainer: Any) -> None:
            # EMA arithmetic can move even constant floats. Restore frozen state
            # explicitly so saved/validated EMA weights obey the same policy.
            if prefixes and trainer.ema is not None:
                state=trainer.model.state_dict()
                with torch.no_grad():
                    for name,value in trainer.ema.ema.state_dict().items():
                        if name.startswith(prefixes):value.copy_(state[name])

        def epoch_end(trainer: Any) -> None:
            # Ultralytics also invokes on_fit_epoch_end during final_eval: guard it.
            if int(trainer.epoch)+1<=receipt['epochs_actually_completed']:return
            counts=Counter(epoch_files);epoch_files.clear()
            train_paths=list((output/'vn/images/train').glob('*'))
            if counts!=Counter(p.stem for p in train_paths):
                raise ValueError('ACTUAL_IMAGE_PRESENTATIONS_DIFFER_FROM_LOCKED_EPOCH')
            if prefixes and state_digest(trainer.model.float(),prefixes)!=frozen_before:
                raise ValueError('FROZEN_BACKBONE_STATE_CHANGED')
            receipt['epochs_actually_completed']=int(trainer.epoch)+1
            receipt['validation_history'].append({'epoch':int(trainer.epoch)+1,
                'metrics':{str(k):float(v) for k,v in trainer.metrics.items()},
                'learning_rates':[float(g['lr']) for g in trainer.optimizer.param_groups],
                'vn_presentations':sum(n for k,n in counts.items() if k.startswith('VN_')),
                'dfire_presentations':sum(n for k,n in counts.items() if k.startswith('DFIRE_'))})
            last=output/'fit/weights/last.pt'
            receipt['last_checkpoint_sha256']=sha256(last) if last.exists() else None
            write_json(receipt_path,receipt)

        model.add_callback('on_train_start',started)
        model.add_callback('on_train_batch_start',batch_start)
        model.add_callback('on_train_batch_end',batch_end)
        model.add_callback('on_train_epoch_start',lambda trainer:epoch_files.clear())
        model.add_callback('on_fit_epoch_end',epoch_end)
        train_keys=['epochs','imgsz','batch','workers','optimizer','lr0','lrf','momentum','weight_decay',
                    'warmup_epochs','deterministic','amp','mosaic','mixup','copy_paste','cutmix',
                    'close_mosaic','fraction','compile']
        overrides={key:protocol[key] for key in train_keys}
        model.train(trainer=EvidenceTrainer,data=str(output/'vn.yaml'),project=str(output),name='fit',
                    exist_ok=False,seed=args.seed,device=device,freeze=len(prefixes) if prefixes else None,
                    time=None,patience=protocol['epochs']+1,
                    resume=str(output/'fit/weights/last.pt') if args.resume else False,**overrides)
        receipt['runtime_seconds']=time.monotonic()-before
        receipt['status']='TRAINED' if receipt['epochs_actually_completed']==protocol['epochs'] else 'PARTIAL_TRAINING'
        for name in ['best','last']:
            path=output/'fit/weights'/f'{name}.pt'
            receipt[name+'_checkpoint_sha256']=sha256(path) if path.exists() else None
        if prefixes:
            final=YOLO(str(output/'fit/weights/best.pt')).model.float()
            receipt['frozen_state_after']=state_digest(final,prefixes)
            if receipt['frozen_state_after']!=frozen_before:raise ValueError('SAVED_FROZEN_BACKBONE_STATE_CHANGED')
        adapted=YOLO(str(output/'fit/weights/best.pt'))
        receipt['adapted_validation']={}
        for corpus,config in [('vn','vn.yaml'),('dfire','dfire_eval.yaml')]:
            receipt['adapted_validation'][corpus]=extract_map(adapted.val(
                data=str(output/config),split='val',imgsz=protocol['imgsz'],conf=.001,
                iou=.7,device=device,workers=0,project=str(output),name='adapted_'+corpus,
                exist_ok=False,plots=False)) if (output/config).exists() else None
        receipt['video_evaluation_status']='REQUIRES_PAIRED_VALIDATION_VIDEO_RUN'
        receipt['selected_model']=None
        write_json(receipt_path,receipt)
        verify_file_hash(checkpoint,expected['baseline_sha256'])
        return 0
    except (Exception,KeyboardInterrupt) as error:
        receipt['status']='INTERRUPTED' if receipt['training_executed'] else 'BLOCKED'
        receipt['error_type']=type(error).__name__;receipt['error']=str(error)
        last=output/'fit/weights/last.pt'
        receipt['last_checkpoint_sha256']=sha256(last) if last.exists() else None
        write_json(receipt_path,receipt)
        LOG.error('%s: %s',receipt['status'],error)
        return 1 if receipt['training_executed'] else 2


def main() -> None:
    """Run a fresh strategy directory, or resume only a matching real checkpoint."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/research/week5_execution_protocol_v2.json')
    parser.add_argument('--strategy',required=True,choices=['full_finetune','freeze_backbone','replay_20_percent'])
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--budget-receipt',type=Path)
    parser.add_argument('--resume',action='store_true')
    logging.basicConfig(level=logging.INFO,stream=sys.stdout)
    sys.exit(execute(parser.parse_args()))


if __name__=='__main__':main()
