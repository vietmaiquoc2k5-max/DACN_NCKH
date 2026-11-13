"""Measured validation replay and exploratory holdout; never counterfeit test independence."""
from __future__ import annotations
import argparse,csv,hashlib,itertools,json,sys,time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.temporal_verifier import TemporalVerifier,DetectionBox
from src.event_eval import Event,evaluate,merge_frame_alarms_to_events,exact_poisson_ci
from ground_truth_studio.integrity_service import sha256
from scripts.run_week4_final_scientific import save


def predict(rows,config,duration):
    """Replay the real temporal engine on stored real tracks/features and source time."""
    verifier=TemporalVerifier(window_size_n=config['n'],confirm_frames_k=config['k'],suspect_frames_k=1,
        cooldown_sec=config['cooldown'],feature_threshold=config['variation'],exclusion_zones=[])
    raw=[];alerts=[]
    for row in rows:
        ts=row['timestamp_sec']
        if ts>=duration:continue
        identity={'video_id':row['video_id'],'camera_id':row['camera_id'],'timestamp_sec':ts}
        for d in row['detections']:
            if d['confidence']>=config['confidence']:
                raw.append({**identity,'label':d['class_name'],'confidence':d['confidence']})
        boxes=[];features={}
        for d in row['tracked_by_confidence'][str(config['confidence'])]:
            if d['confidence']<config['confidence']:continue
            boxes.append(DetectionBox(d['class_id'],d['class_name'],d['confidence'],tuple(d['bbox_xyxy']),tuple(d['norm_xywh']),d['track_id']))
            features[d['track_id']]=SimpleNamespace(**d['features'])
        result=verifier.update(boxes,tracks_assigned=True,feature_map=features,timestamp_sec=ts)
        for alert in result['triggered_alerts']:alerts.append({**identity,'label':alert['label'],'confidence':alert['confidence']})
    return [asdict(e) for e in merge_frame_alarms_to_events(raw,config['merge_gap'])],[asdict(e) for e in merge_frame_alarms_to_events(alerts,config['merge_gap'])]


def measure(entries,gt,predictions):
    positive={e['video_id'] for e in entries if e['kind']=='positive'};negative={e['video_id'] for e in entries if e['kind']=='negative'}
    hours=sum(e['duration_seconds'] for e in entries if e['kind']=='negative')/3600
    ids=positive|negative
    result=evaluate([Event(**g) for g in gt if g['video_id'] in positive],[Event(**p) for p in predictions if p['video_id'] in ids],positive,negative,hours or None,allow_missing_negative=True)
    count=result['counts']['negative_false_alarms']
    result['metrics']['poisson_ci_95']=list(exact_poisson_ci(count,hours)) if hours>0 else None
    delays=sorted(m['delay_sec'] for m in result['matches'])
    result['metrics']['median_delay_sec']=delays[len(delays)//2] if delays else None
    result['metrics']['p95_delay_sec']=delays[min(len(delays)-1,int(.95*len(delays)))] if delays else None
    result['independent_test_valid']=False;result['scientific_acceptance']='PARTIAL'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True,type=Path);args=parser.parse_args();out=args.run_dir.resolve()
    assert out.is_relative_to(ROOT/'runs/benchmarks/week4_final_scientific')
    receipt=json.loads((out/'integrity_receipt.json').read_text());assert receipt['complete']
    scope=json.loads((out/'source_scope.json').read_text());entries=scope['videos'];gt=json.loads((out/'positive_gt_events.json').read_text())
    assert receipt['scope_sha256']==sha256(out/'source_scope.json')
    configs=[dict(confidence=c,k=k,n=n,variation=f,cooldown=cd,merge_gap=2.0) for c,(k,n),f,cd in itertools.product([.25,.35,.5],[(2,8),(3,8),(4,10)],[0,.02],[5.,12.])]
    objective={'source':'existing event_eval.grid_search constraints','recall_drop_max':.05,'far_reduction_min':.5,'mean_delay_max_sec':10,
        'ranking':['maximize event_f1','minimize FAR','minimize delay','config_hash'],'no_test_tuning':True,
        'matching':'src.event_eval.evaluate default camera/class one-to-one onset +/-10s; delay=max(0,pred-GT)',
        'episodes':'same video/camera/class gap<=2s; B points are actual temporal confirmed alert times',
        'exclusion_zones':'disabled, no benchmark zones predeclared','grid':configs}
    assert not (out/'operating_point_provisional.json').exists(),'Do not overwrite an evaluated selection; choose a fresh run/evaluation'
    save(out/'selection_protocol.json',objective)
    validation=[e for e in entries if e['split']=='validation'];holdout=[e for e in entries if e['split']=='exploratory_holdout']
    predictions={i:{'A':[],'B':[]} for i in range(len(configs))}
    for entry in validation:
        folder=out/'videos'/entry['video_id'];video_receipt=json.loads((folder/'video_completion.json').read_text())
        assert video_receipt['frame_jsonl_sha256']==sha256(folder/'frame_level.jsonl')
        rows=[json.loads(line) for line in (folder/'frame_level.jsonl').open()]
        for i,config in enumerate(configs):
            a,b=predict(rows,config,entry['duration_seconds']);predictions[i]['A'].extend(a);predictions[i]['B'].extend(b)
        print('VALIDATION_REPLAY',entry['video_id'],flush=True)
    candidates=[]
    for i,config in enumerate(configs):
        a=measure(validation,gt,predictions[i]['A']);b=measure(validation,gt,predictions[i]['B']);am,bm=a['metrics'],b['metrics']
        reduction=(1-bm['false_alarms_per_hour']/am['false_alarms_per_hour']) if am['false_alarms_per_hour'] else None
        feasible=(bm['event_instance_recall']>=am['event_instance_recall']-.05 and bm['mean_detection_delay_sec'] is not None and bm['mean_detection_delay_sec']<=10 and reduction is not None and reduction>=.5)
        candidates.append({'config_id':i,'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),'config':config,
            'validation_scope_hash':scope['validation_scope_hash'],'A':a,'B':b,'far_reduction':reduction,'feasible':feasible,'status':'MEASURED_VALIDATION'})
    ranked=sorted(candidates,key=lambda r:(-r['B']['metrics']['event_f1'],r['B']['metrics']['false_alarms_per_hour'],r['B']['metrics']['mean_detection_delay_sec'] if r['B']['metrics']['mean_detection_delay_sec'] is not None else float('inf'),r['config_sha256']))
    feasible=[r for r in ranked if r['feasible']];selected=(feasible or ranked)[0]
    save(out/'validation_results.json',candidates)
    with (out/'parameter_sweep.csv').open('w',newline='') as stream:
        fields=['config_id','config_sha256','validation_scope_hash','confidence','k','n','variation','cooldown','merge_gap','event_recall','event_precision','event_f1','mean_delay','false_alarm_events','negative_hours','far_per_hour','poisson_ci_95','status','feasible'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in candidates:
            m=r['B']['metrics'];writer.writerow({**r['config'],**{k:r[k] for k in ['config_id','config_sha256','validation_scope_hash','status','feasible']},'event_recall':m['event_instance_recall'],'event_precision':m['event_precision'],'event_f1':m['event_f1'],'mean_delay':m['mean_detection_delay_sec'],'false_alarm_events':r['B']['counts']['negative_false_alarms'],'negative_hours':m['negative_video_hours'],'far_per_hour':m['false_alarms_per_hour'],'poisson_ci_95':m['poisson_ci_95']})
    point={'status':'PROVISIONAL','config':selected['config'],'config_sha256':selected['config_sha256'],'validation_scope_hash':scope['validation_scope_hash'],
        'selection':'FEASIBLE_VALIDATION_MAX_F1' if feasible else 'NO_FEASIBLE_OPERATING_POINT; diagnostic max-F1 only',
        'validation_result':selected,'selection_timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'test_not_yet_used_declaration':False,'reason':'Historical test/negative data already used; no independent lock issued',
        'GT_PROTOCOL':'JOINT_HUMAN_CONSENSUS','checkpoint_sha256':receipt['checkpoint_sha256']}
    save(out/'operating_point_provisional.json',point);point_hash=sha256(out/'operating_point_provisional.json')
    hold_predictions={'A':[],'B':[]}
    for entry in holdout:
        folder=out/'videos'/entry['video_id'];r=json.loads((folder/'video_completion.json').read_text());assert r['frame_jsonl_sha256']==sha256(folder/'frame_level.jsonl')
        rows=[json.loads(line) for line in (folder/'frame_level.jsonl').open()]
        a,b=predict(rows,point['config'],entry['duration_seconds']);hold_predictions['A'].extend(a);hold_predictions['B'].extend(b)
    assert sha256(out/'operating_point_provisional.json')==point_hash
    measured={name:measure(holdout,gt,preds) for name,preds in hold_predictions.items()}
    save(out/'event_predictions_a.json',hold_predictions['A']);save(out/'event_predictions_b.json',hold_predictions['B'])
    save(out/'baseline_vs_temporal.json',{'scope':'EXPLORATORY_HOLDOUT_NOT_INDEPENDENT_TEST','GT_PROTOCOL':'JOINT_HUMAN_CONSENSUS','metrics':measured,'operating_point_sha256':point_hash,'official_acceptance':False})
    with (out/'baseline_vs_temporal.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['metric','A','B'])
        for name in measured['A']['metrics']:writer.writerow([name,measured['A']['metrics'][name],measured['B']['metrics'][name]])
    for filename,key in [('false_alarm_events.jsonl','unmatched_predictions'),('missed_events.jsonl','unmatched_ground_truth'),('event_matching.jsonl','matches')]:
        with (out/filename).open('w') as stream:
            for pipeline,result in measured.items():
                for row in result[key]:
                    if filename=='false_alarm_events.jsonl' and not row.get('is_negative_video'):continue
                    stream.write(json.dumps({'pipeline':pipeline,**row})+'\n')
    with (out/'per_video_results.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['video_id','pipeline','kind','predicted_events','matched_events','missed_events','negative_hours','far'])
        for entry in holdout:
            for name,events in hold_predictions.items():
                preds=[e for e in events if e['video_id']==entry['video_id']];matches=sum(r['ground_truth']['video_id']==entry['video_id'] for r in measured[name]['matches']);miss=sum(r['video_id']==entry['video_id'] for r in measured[name]['unmatched_ground_truth']);hours=entry['duration_seconds']/3600 if entry['kind']=='negative' else None
                writer.writerow([entry['video_id'],name,entry['kind'],len(preds),matches,miss,hours,len(preds)/hours if hours else None])
    save(out/'scientific_acceptance.json',{'status':'PARTIAL','reason':['INDEPENDENT_TEST_CONTAMINATION','No official validation/test lock','Training corpus hashes/scene independence unverified'],'GT_PROTOCOL':'JOINT_HUMAN_CONSENSUS','independent_review_not_required':True,'validation_candidates_executed':len(candidates),'independent_test_executed':False,'exploratory_holdout_measured':True})
    print('MEASURED_CANDIDATES',len(candidates),'FEASIBLE',len(feasible),flush=True)


if __name__=='__main__':main()
