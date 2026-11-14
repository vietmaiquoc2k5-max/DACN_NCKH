"""Reauthenticate historical Week 4 evidence without rerunning its detector or writing it."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from ground_truth_studio.config import confined
from ground_truth_studio.integrity_service import sha256
from scripts.evaluate_week4_final_scientific import measure,predict
from src.scientific_inference import coverage,sample_indices
from src.video_audit_v2 import write_json


def verify(old: Path,out: Path) -> dict:
    """Verify immutable frame/source evidence and reproduce the recorded evaluator.

    Args:
        old: Existing historical run; all access is read-only.
        out: Current closure output directory, distinct from the historical run.

    Returns:
        Checksums, real replay comparisons and the explicitly exploratory status.
    """
    if old==out or not out.is_relative_to(ROOT/'runs/benchmarks/week4_acceptance_closure'):
        raise ValueError('Output must be a new closure directory')
    if (out/'existing_week4_evidence_verification.json').exists():raise FileExistsError('Do not overwrite verification evidence')
    out.mkdir(parents=True,exist_ok=True)
    inventory=json.loads((old/'artifact_sha256.json').read_text(encoding='utf-8'))
    mismatches=[name for name,digest in inventory.items() if sha256(confined(old,name))!=digest]
    if mismatches:raise ValueError('Historical artifact hashes changed: '+','.join(mismatches))
    manifest=json.loads((old/'run_manifest.json').read_text(encoding='utf-8'))
    config=json.loads((old/'inference_config.json').read_text(encoding='utf-8'))
    scope=json.loads((old/'source_scope.json').read_text(encoding='utf-8'))
    protocol=json.loads((old/'experiment_protocol_lock.json').read_text(encoding='utf-8'))
    code={}
    for name,digest in protocol['code_sha256'].items():
        actual=sha256(ROOT/name)
        status='EXACT_RUNTIME_BYTES' if actual==digest else 'NOT_MATCHED'
        archived=None
        if status=='NOT_MATCHED':
            import hashlib
            for commit in ['6942fa567683ed9fdc3ef455188db1a74cef2c7f','730342623b5946f0eb9d25241eb4b8fc20488e79']:
                blob=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
                if hashlib.sha256(blob).hexdigest()==digest and blob.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n'):
                    status='AUTHENTICATED_GIT_RUNTIME_BYTES_CURRENT_EOL_ONLY_DIFFERENCE';archived=commit;break
        if status=='NOT_MATCHED':raise ValueError('Historical implementation cannot be authenticated: '+name)
        code[name]={'status':status,'expected_sha256':digest,'current_sha256':actual,'archived_commit':archived}
    if sha256(old/'source_scope.json')!=manifest['scope_sha256'] or sha256(old/'inference_config.json')!=manifest['config_sha256']:
        raise ValueError('Historical scope/config binding failed')
    if sha256(ROOT/manifest['checkpoint'])!=manifest['checkpoint_sha256']:
        raise ValueError('Baseline checkpoint changed')
    receipts={r['video_id']:r for r in json.loads((old/'video_completion.json').read_text(encoding='utf-8'))}
    all_rows={};counts=[]
    for entry in scope['videos']:
        vid=entry['video_id'];path=confined(ROOT,entry['relative_path'])
        if sha256(path)!=entry['source_sha256']:raise ValueError('Historical source changed: '+vid)
        folder=old/'videos'/vid
        if sha256(folder/'frame_level.jsonl')!=receipts[vid]['frame_jsonl_sha256']:raise ValueError('Frame JSONL hash differs')
        pts=json.loads((old/'pts'/(vid+'.json')).read_text(encoding='utf-8'));ticks=pts['pts'];base=Fraction(pts['time_base'])
        indices=sample_indices(ticks,pts['time_base'])
        binding={'run_id':old.name,'video_id':vid,'camera_id':'BENCH_'+vid,'source_sha256':entry['source_sha256'],
                 'checkpoint_sha256':manifest['checkpoint_sha256'],'config_sha256':manifest['config_sha256'],
                 'sampling_protocol':config['sampling_protocol']}
        rows=[]
        with (folder/'frame_level.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);index=row['frame_index']
                if row['timestamp_sec']!=float((ticks[index]-ticks[0])*base) or row['source_pts_sec']!=float(ticks[index]*base):raise ValueError('Frame seconds/ticks differ')
                if row['detector_input_config']!=config:raise ValueError('Frame input config differs')
                rows.append(row)
        checked=coverage(rows,indices,ticks,pts['time_base'],binding)
        if receipts[vid]['native_decoded_frames']!=len(ticks):raise ValueError('Native frame count differs')
        counts.append(checked)
        # Store JSON records only, never source images. Replay does not call NN.
        all_rows[vid]=rows
    gt=json.loads((old/'positive_gt_events.json').read_text(encoding='utf-8'))
    candidates=json.loads((old/'validation_results.json').read_text(encoding='utf-8'))
    validation=[e for e in scope['videos'] if e['split']=='validation']
    replayed=[]
    for candidate in candidates:
        predictions={'A':[],'B':[]}
        for entry in validation:
            a,b=predict(all_rows[entry['video_id']],candidate['config'],entry['duration_seconds'])
            predictions['A'].extend(a);predictions['B'].extend(b)
        results={name:measure(validation,gt,pred) for name,pred in predictions.items()}
        if any(results[name]['metrics']!=candidate[name]['metrics'] or results[name]['counts']!=candidate[name]['counts'] for name in ['A','B']):
            raise ValueError('Validation replay differs')
        replayed.append(candidate['config_sha256'])
    point=json.loads((old/'operating_point_provisional.json').read_text(encoding='utf-8'))
    holdout=[e for e in scope['videos'] if e['split']=='exploratory_holdout'];predictions={'A':[],'B':[]}
    for entry in holdout:
        a,b=predict(all_rows[entry['video_id']],point['config'],entry['duration_seconds']);predictions['A'].extend(a);predictions['B'].extend(b)
    comparison=json.loads((old/'baseline_vs_temporal.json').read_text(encoding='utf-8'))
    measured={name:measure(holdout,gt,pred) for name,pred in predictions.items()}
    for name in ['A','B']:
        if measured[name]!=comparison['metrics'][name]:raise ValueError('Exploratory replay differs')
        m=measured[name];count=m['counts']['negative_false_alarms'];hours=m['metrics']['negative_video_hours']
        if not math.isclose(m['metrics']['false_alarms_per_hour'],count/hours,rel_tol=1e-12):raise ValueError('FAR scope arithmetic fails')
    # Preservation is checked again after replay: no historical artifact writes.
    if any(sha256(confined(old,name))!=digest for name,digest in inventory.items()):raise ValueError('Historical artifacts modified during verification')
    result={'complete':True,'historical_artifacts_verified':len(inventory),'historical_artifacts_modified':False,
            'source_videos_verified':len(counts),'frames_verified':sum(r['processed_frames'] for r in counts),
            'source_pts_ticks_and_seconds_verified':True,'validation_candidates_replayed':len(replayed),
            'detector_inference_rerun':False,'code_provenance':code,'operating_point_status':point['status'],
            'independent_test_valid':False,'scope':'EXPLORATORY_HOLDOUT_NOT_INDEPENDENT_TEST',
            'A':{'metrics':measured['A']['metrics'],'counts':measured['A']['counts']},
            'B':{'metrics':measured['B']['metrics'],'counts':measured['B']['counts']},
            'roi':'NORMALIZED_RECTANGLE_INTEGRATION_ONLY; ZONES_DISABLED_IN_MEASURED_BENCHMARK',
            'scientific_status':'PARTIAL'}
    write_json(out/'existing_week4_evidence_verification.json',result)
    print('HISTORICAL_WEEK4_VERIFIED',len(counts),'VIDEOS',result['frames_verified'],'FRAMES',len(replayed),'CANDIDATES','NO_NN_RERUN',flush=True)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--historical-run',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();verify(args.historical_run.resolve(),args.output_dir.resolve())


if __name__=='__main__':main()
