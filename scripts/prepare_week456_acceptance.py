"""Read-only evidence/readiness audit; no training, inference or GT promotion."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from ground_truth_studio.config import Settings,confined
from ground_truth_studio.integrity_service import sha256,frozen,catalog
from src.research_readiness import split_leaks,replay_plan


def main() -> None:
    """Collect actual evidence and write reproducible plans to a fresh directory."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--joint-export',type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'.tools'):
        parser.error('Use a fresh output directory beneath .tools')
    out.mkdir(parents=True)
    def save(name,value):
        (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    source=args.joint_export.resolve()
    hashes=json.loads((source/'file_sha256.json').read_text(encoding='utf-8'))
    assert all(sha256(confined(source,name))==digest for name,digest in hashes.items())
    reviews=json.loads((source/'single_review_records.json').read_text(encoding='utf-8'))
    # Verify every submitted source, including excluded technical inputs; never
    # treat a semantic claim as decode acceptance or independent verification.
    for row in reviews:
        assert sha256(confined(ROOT,row['relative_path']))==row['source_sha256']
    save('joint_consensus_prior.json',{'status':'JOINT_CONSENSUS_PRIOR','export_id':source.name,
        'files':hashes,'review_ids':[r['review_id'] for r in reviews],'submitted_records':len(reviews),
        'kind_counts':{k:sum(r['kind']==k for r in reviews) for k in {r['kind'] for r in reviews}},
        'independent':False,'declaration_basis':'User explicitly reports two people jointly watched and agreed',
        'identity_provenance':'Retained in original private ground_truth_provenance.json; no second identity invented'})
    v2=ROOT/'runs/benchmarks/week4_decode_audit_v2/run_20261113'
    artifact=json.loads((v2/'artifact_sha256.json').read_text())
    assert all(sha256(confined(v2,n))==h for n,h in artifact.items())
    summary=json.loads((v2/'summary.json').read_text())
    assert summary['eligible_scope_sha256']=='cc77669f0d5b4695fb7f1f043a6a446ba5ca949ecf46a7cc3f455e667cd89a78'
    frozen_hash,_=frozen(Settings.load())
    with (ROOT/'datasets/vn_dataset/split_manifest.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    leaks=split_leaks(rows)
    image_hash_splits={}
    missing=[]; bad_labels=[]
    for row in rows:
        path=ROOT/'datasets'/row['relative_path'].replace('\\','/')
        if not path.is_file():
            missing.append(row['image_name']);continue
        image_hash_splits.setdefault(sha256(path),set()).add(row['split'])
        label=ROOT/'datasets/vn_dataset/labels'/row['split']/(path.stem+'.txt')
        if not label.is_file(): bad_labels.append(row['image_name']);continue
        for line in label.read_text().splitlines():
            try:
                cls,x,y,w,h=map(float,line.split())
                assert cls in (0,1) and all(0<=v<=1 for v in (x,y,w,h)) and w>0 and h>0
            except (ValueError,AssertionError): bad_labels.append(row['image_name'])
    dfire=ROOT/'datasets/dfire'
    dfire_counts={split:len(list((dfire/split/'images').glob('*'))) for split in ['train','val','test']}
    checkpoint=ROOT/'runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt'
    checkpoint_hash=sha256(checkpoint)
    import torch
    from ultralytics import YOLO
    model=YOLO(str(checkpoint))
    backbone=len(model.model.yaml['backbone'])
    prefixes=tuple(f'model.{i}.' for i in range(backbone))
    freeze_names=[n for n,_ in model.model.named_parameters() if n.startswith(prefixes)]
    strategies=[]
    for strategy in ['full_finetune','freeze_backbone','replay_20_percent']:
        strategies.append({'strategy':strategy,'checkpoint':checkpoint.relative_to(ROOT).as_posix(),
            'checkpoint_sha256':checkpoint_hash,'epochs':30,'seeds':[42,123,456],'imgsz':640,
            'data':'data_vn.yaml','class_mapping':model.names,'freeze_module_indices':list(range(backbone)) if strategy=='freeze_backbone' else [],
            'replay':{'unit':'20% combined epoch image presentations; ceil(VN_train/4) D-Fire train only','status':'BLOCKED_DFIRE_MISSING'} if strategy=='replay_20_percent' else None,
            'status':'PLAN_ONLY_NOT_TRAINED'})
    save('week5_readiness.json',{'status':'BLOCKED','vn_manifest_rows':len(rows),'vn_video_split_leaks':leaks,
        'cross_split_duplicate_hashes':sum(len(s)>1 for s in image_hash_splits.values()),'missing_images':missing,
        'invalid_or_missing_labels':sorted(set(bad_labels)),'dfire_split_counts':dfire_counts,
        'checkpoint_sha256':checkpoint_hash,'class_mapping':model.names,'freeze_parameter_names':freeze_names,
        'strategies':strategies,'training_started':False,'metrics':None})
    plan={'confidence':[.25,.35,.5],'k_n':[[2,8],[3,8],[4,10]],'variation_threshold':[0,.02],
          'cooldown_sec':[5,12],'merge_gap_sec':[2],'zones':['disabled','predeclared_camera_config']}
    save('validation_sweep_plan.json',{'status':'BLOCKED_VALIDATION_EVIDENCE','grid':plan,
         'candidate_count':3*3*2*2*1*2,'validation_scope':None,'test_scope':None,'locked':False,
         'selection_objective':'max event F1 subject to predeclared recall, delay and FAR constraints; lock before independent test',
         'metrics':None})
    matrix=[]
    base={'imgsz':640,'conf':.35,'k':3,'n':8}
    for axis,values in [('imgsz',[416,640,960]),('conf',[.25,.35,.5]),('k_n',[[2,8],[3,8],[4,10]])]:
        for value in values:matrix.append({'axis':axis,'value':value,'baseline':base,'day_night':'HUMAN_METADATA_REQUIRED','status':'PLANNED'})
    save('week6_plan.json',{'status':'PLANNED','ablation_one_factor_matrix':matrix,
        'backends':{m:'INSTALLED_NOT_BENCHMARKED' if importlib.util.find_spec(m) else 'NOT_TESTED_ENVIRONMENT_UNAVAILABLE' for m in ['torch','onnxruntime','openvino','tensorrt']},
        'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cpu':platform.processor(),'fps':None,'latency':None,'verified_error_cases':0,
        'benchmark_protocol':{'warmup_frames':50,'measured_frames':1000,'batch_size':1,'camera_counts':[1,4,9],
                              'record':['throughput_fps','inference_latency','end_to_end_latency','RAM','VRAM','versions','resolution','source_scope']}})
    for name,fields in {'parameter_sweep.csv':['config_hash','split','recall','f1','far','delay','ci_low','ci_high'],
        'per_video_events.csv':['video_id','pipeline','tp','fp','fn','delay','eligible_hours'],
        'false_alarm_events.csv':['video_id','pipeline','event_id','start_sec'],
        'missed_events.csv':['video_id','ground_truth_event_id'],
        'event_matching.csv':['video_id','gt_event_id','pred_event_id','match'],
        'verified_error_cases.csv':['case_id','video_id','source_sha256','frame_index','gt_review_id','prediction_ref','verified_by','cause']}.items():
        with (out/name).open('w',newline='',encoding='utf-8') as stream:csv.writer(stream).writerow(fields)
    metrics=['Event Recall','Event Precision','Event F1','Mean Detection Delay','False Alarm Events','Valid Negative Hours','FAR/hour','Poisson CI95%']
    save('baseline_comparison.json',{'status':'BLOCKED_GT_SPLIT_AND_INFERENCE_EVIDENCE','metrics':{m:{'A':None,'B':None} for m in metrics},
                                   'technical_candidate_hours':summary['technical_valid_negative_candidate_hours'],'official':False})
    (out/'comparison.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="750" height="100"><rect width="750" height="100" fill="#eee"/><text x="20" y="50">BLOCKED: no independently validated A/B measurements; metrics = null</text></svg>')
    inputs=['src/temporal_verifier.py','src/event_eval.py','src/bytetrack_adapter.py','src/temporal_features.py','src/camera_manager.py','src/video_audit.py','src/video_audit_v2.py','src/inference_frame_evidence.py','src/week4_acceptance.py','scripts/export_week4_frame_jsonl.py','scripts/ablation_validation_recall.py','scripts/recover_negative_decode_v2.py','data/negative_videos_manifest.json','data/video_catalog_local_v2.json','data/video_dataset_list.csv']
    save('provenance.json',{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'source_sha256':{n:sha256(ROOT/n) for n in inputs},'frozen_sha256':frozen_hash,'technical_scope_sha256':summary['eligible_scope_sha256'],
         'checkpoint_sha256':checkpoint_hash,'sampling_protocol_proposed':'FULL_DURATION_SAMPLED_5FPS; source PTS indexed native stride, not native-frame inference',
         'inference_started':False,'inference_blockers':['Independent human GT','Explicit validation/test video scopes','Locked validation configuration'],
         'operating_point':'PROVISIONAL','scientific_week4':'PARTIAL'})
    save('file_sha256.json',{p.name:sha256(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'records_verified':len(reviews),'technical_hours':summary['technical_valid_negative_candidate_hours'],
        'vn_split_leaks':leaks,'missing_images':len(missing),'bad_labels':len(set(bad_labels)),'dfire':dfire_counts,'freeze_params':len(freeze_names)}))


if __name__=='__main__':main()
