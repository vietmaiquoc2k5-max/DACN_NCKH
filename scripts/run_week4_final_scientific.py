"""Consensus GT audit and bounded-memory, source-PTS full-duration inference."""
from __future__ import annotations
import argparse, csv, hashlib, json, logging, subprocess, sys, time
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from ground_truth_studio.config import Settings,confined
from ground_truth_studio.integrity_service import sha256
from src.scientific_inference import sample_indices,coverage

CHECKPOINT='runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt'
EXPECTED='f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6'


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)


def source_pts(entry,out):
    """Use authenticated v2 packet/frame evidence, otherwise freshly decode PTS."""
    root=ROOT/'runs/benchmarks/week4_decode_audit_v2/run_20261113/records'/entry['video_id']
    if (root/'record.json').exists():
        record=json.loads((root/'record.json').read_text())
        assert record['source_sha256']==entry['source_sha256'] and record['technical_eligible']
        timeline=root/'timeline.stdout'
        assert sha256(timeline)==record['artifact_sha256']['timeline.stdout']
        base=record['time_base']; rows=[]
        with timeline.open() as stream:
            for line in stream:
                if line.startswith('frame|'):
                    fields=dict(p.split('=',1) for p in line.strip().split('|')[1:] if '=' in p)
                    rows.append((int(fields['pts']),int(fields['duration'])))
        return base,rows,'AUTHENTICATED_V2_SOURCE_FRAME_TIMELINE'
    settings=Settings.load();path=confined(ROOT,entry['relative_path'])
    command=[settings.ffprobe,'-v','error','-threads','2','-select_streams','v:0','-show_frames','-show_streams','-show_entries','frame=best_effort_timestamp,duration,pkt_duration:stream=time_base','-of','json',str(path)]
    result=subprocess.run(command,capture_output=True,timeout=1800)
    (out/(entry['video_id']+'_pts.stderr')).write_bytes(result.stderr)
    assert result.returncode==0 and not result.stderr.strip(),'SOURCE_PTS_PROBE_ERROR'
    raw=json.loads(result.stdout);base=raw['streams'][0]['time_base'];rows=[]
    for r in raw['frames']:
        rows.append((int(r['best_effort_timestamp']),int(r.get('duration',r.get('pkt_duration',0)))))
    assert rows and all(d>0 for _,d in rows),'MISSING_EXPLICIT_FRAME_DURATION'
    return base,rows,'FRESH_FFPROBE_SOURCE_FRAME_PTS'


def prepare(out,joint):
    """Freeze a source/config scope using human consensus, never detector outcomes."""
    out.mkdir(parents=True,exist_ok=False);(out/'pts').mkdir();(out/'videos').mkdir()
    hashes=json.loads((joint/'file_sha256.json').read_text())
    assert all(sha256(confined(joint,n))==v for n,v in hashes.items())
    reviews=json.loads((joint/'single_review_records.json').read_text())
    technical=json.loads((ROOT/'runs/benchmarks/week4_decode_audit_v2/run_20261113/summary.json').read_text())
    eligible={r['video_id']:r for r in technical['eligible_scope_identity']}
    accepted=[];excluded=[];events=[];seen_ids=set();seen_sha=set()
    for review in sorted(reviews,key=lambda r:r['video_id']):
        vid=review['video_id']; reasons=[]
        entry={k:review[k] for k in ['video_id','relative_path','source_sha256']}
        if vid in seen_ids: reasons.append('DUPLICATE_REVIEW_IDENTITY')
        seen_ids.add(vid)
        if not review.get('submitted_at_utc') or not review.get('evidence_ref') or review.get('attest_full_view') is not True:reasons.append('MISSING_SUBMISSION_EVIDENCE')
        if review.get('uncertain_intervals'):reasons.append('UNCERTAIN_INTERVALS')
        if review['kind']=='negative' and (review['decision']!='VERIFIED_NEGATIVE' or review['events']):reasons.append('NOT_HUMAN_VERIFIED_NEGATIVE')
        if review['kind']=='negative' and vid not in eligible:reasons.append('TECHNICAL_AUDIT_EXCLUDED')
        if not reasons:
            path=confined(ROOT,review['relative_path'])
            if not path.is_file() or sha256(path)!=review['source_sha256']:reasons.append('SOURCE_SHA_MISMATCH')
            if review['source_sha256'] in seen_sha:reasons.append('DUPLICATE_SOURCE_CONTENT')
        if not reasons:
            try:
                base,rows,basis=source_pts(entry,out/'pts'); pts=[r[0] for r in rows]
                indices=sample_indices(pts,base);duration=float((rows[-1][0]+rows[-1][1]-rows[0][0])*Fraction(base))
                assert all(b[0]==a[0]+a[1] for a,b in zip(rows,rows[1:])),'SOURCE_TIMELINE_GAP_OR_OVERLAP'
                if review['kind']!='negative':
                    assert review['events'],'NO_POSITIVE_EVENTS'
                    spans=set()
                    for event in review['events']:
                        assert event['source_sha256']==review['source_sha256'] and event['label'] in ['FIRE','SMOKE'],'EVENT_SOURCE_OR_CLASS'
                        assert event['timestamp_provenance']=='SOURCE_PTS','ESTIMATED_EVENT_TIMESTAMP'
                        key=(event['label'],event['start_sec'],event['end_sec']);assert key not in spans,'DUPLICATE_EVENT';spans.add(key)
                        assert 0<=event['frame_start']<=event['frame_end']<len(pts),'EVENT_FRAME_BOUNDS'
                        for f,t in [(event['frame_start'],event['start_sec']),(event['frame_end'],event['end_sec'])]:
                            # Export used decimal FFprobe seconds (six digits); differencing
                            # two rounded timestamps has <=1 microsecond quantization error.
                            assert abs(float((pts[f]-pts[0])*Fraction(base))-t)<=0.0000011,'EVENT_PTS_MISMATCH'
                        assert 0<=event['start_sec']<=event['end_sec']<=duration,'EVENT_TIME_BOUNDS'
                else:duration=min(duration,float(review['duration_seconds']),eligible[vid]['duration_seconds'])
                save(out/'pts'/(vid+'.json'),{'pts':pts,'time_base':base,'durations':[r[1] for r in rows],'basis':basis})
                entry.update(kind='negative' if review['kind']=='negative' else 'positive',duration_seconds=duration,
                             native_frames=len(pts),expected_samples=len(indices),source_timestamp_original=review['source_timestamp_provenance'])
            except (AssertionError,ValueError,KeyError) as error:reasons.append(str(error))
        if reasons:excluded.append({'video_id':vid,'reasons':reasons});continue
        accepted.append(entry);seen_sha.add(review['source_sha256'])
        if entry['kind']=='positive':
            events.extend(dict(video_id=vid,camera_id='BENCH_'+vid,label=e['label'],start_sec=e['start_sec'],end_sec=e['end_sec']) for e in review['events'])
        print('GT_ACCEPTED',vid,entry['expected_samples'],flush=True)
    # Preserve prior odd-POS validation split. Negative allocation is metadata-only:
    # deterministic SHA order, small sources first, >=0.5h validation while retaining
    # >=5h exploratory negative exposure where the pool permits. All were seen before.
    negative=[e for e in accepted if e['kind']=='negative']; total=sum(e['duration_seconds'] for e in negative)
    val_neg=set();seconds=0
    for e in sorted(negative,key=lambda e:(e['duration_seconds'],e['source_sha256'])):
        if seconds>=1800:break
        if seconds+e['duration_seconds']<=max(0,total-18000):val_neg.add(e['video_id']);seconds+=e['duration_seconds']
    for e in accepted:e['split']='validation' if (e['kind']=='positive' and int(e['video_id'].split('_')[-1])%2) or e['video_id'] in val_neg else 'exploratory_holdout'
    split={'protocol':'VIDEO_GROUP_ONLY_METADATA_PREDECLARED','independent_test_valid':False,
           'contamination':'INDEPENDENT_TEST_CONTAMINATION','reason':'Historic POS evaluation/sweep and all negative full inference/FAR results were already available; no fresh independent holdout claimed',
           'training_provenance':'args.yaml declares data_dfire.yaml; original training content-hash manifest unavailable',
           'scene_independence':'NOT_VERIFIED; logical BENCH camera IDs are not real scene IDs',
           'videos':accepted,'validation_scope_hash':hashlib.sha256(json.dumps([e for e in accepted if e['split']=='validation'],sort_keys=True).encode()).hexdigest()}
    save(out/'source_scope.json',split);save(out/'positive_gt_events.json',events)
    with (out/'positive_gt_events.jsonl').open('w') as f:
        for e in events:f.write(json.dumps(e)+'\n')
    for name,data,fields in [('negative_gt_scope.csv',negative,['video_id','source_sha256','duration_seconds','split']),('gt_exclusion_report.csv',excluded,['video_id','reasons'])]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(data)
    save(out/'gt_evidence_audit.json',{'GT_PROTOCOL':'JOINT_HUMAN_CONSENSUS','review_records':len(reviews),'accepted_positive':sum(e['kind']=='positive' for e in accepted),'accepted_negative':len(negative),'negative_hours':total/3600,'excluded':excluded})
    save(out/'ground_truth_provenance.json',{'GT_PROTOCOL':'JOINT_HUMAN_CONSENSUS','export_id':joint.name,'bundle_sha256':hashes,'independent_cross_verified':False,'private_identities_published':False})
    config={'imgsz':416,'detector_conf':.10,'confidences':[.25,.35,.5],'iou':.7,'max_det':300,'rect':False,'half':False,'batch':8,'sample_hz':5,'sampling_protocol':'FULL_DURATION_SAMPLED_5FPS','sampling_rule':'first source PTS frame at/after elapsed grid; sparse grids skipped without duplicating a frame','zones':[]}
    save(out/'inference_config.json',config)
    assert sha256(ROOT/CHECKPOINT)==EXPECTED
    save(out/'run_manifest.json',{'run_id':out.name,'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),'checkpoint':CHECKPOINT,'checkpoint_sha256':EXPECTED,'config_sha256':sha256(out/'inference_config.json'),'scope_sha256':sha256(out/'source_scope.json'),'GT_PROTOCOL':'JOINT_HUMAN_CONSENSUS','status':'PREFLIGHT_COMPLETE','operating_point':'PROVISIONAL','independent_test_valid':False,'runner_sha256':sha256(Path(__file__))})


def infer_video(out,entry,model,device):
    """Decode original frames once, retaining bounded batches and every real result."""
    import cv2
    from dataclasses import asdict
    from src.config import ByteTrackSettings
    from src.bytetrack_adapter import ByteTrackAdapter
    from src.temporal_features import TemporalFeatureExtractor
    from src.temporal_verifier import DetectionBox
    manifest=json.loads((out/'run_manifest.json').read_text());config=json.loads((out/'inference_config.json').read_text())
    vid=entry['video_id'];folder=out/'videos'/vid;folder.mkdir(exist_ok=True)
    pts_ref=json.loads((out/'pts'/(vid+'.json')).read_text());pts=pts_ref['pts'];base=pts_ref['time_base'];indices=sample_indices(pts,base)
    path=confined(ROOT,entry['relative_path']);assert sha256(path)==entry['source_sha256']
    binding={'run_id':out.name,'video_id':vid,'camera_id':'BENCH_'+vid,'source_sha256':entry['source_sha256'],'checkpoint_sha256':EXPECTED,'config_sha256':manifest['config_sha256'],'sampling_protocol':config['sampling_protocol']}
    final=folder/'video_completion.json'
    if final.exists():
        old=json.loads(final.read_text()); assert old['source_sha256']==entry['source_sha256'] and old['config_sha256']==manifest['config_sha256'] and old['frame_jsonl_sha256']==sha256(folder/'frame_level.jsonl')
        coverage((json.loads(line) for line in (folder/'frame_level.jsonl').open()),indices,pts,base,binding);return old
    assert not (folder/'frame_level.jsonl').exists(),'Partial attempt retained; use a fresh run or archive incomplete attempt explicitly'
    trackers={str(c):ByteTrackAdapter(ByteTrackSettings(track_high_thresh=c,track_low_thresh=.1,new_track_thresh=c,track_buffer=15),per_class=True) for c in config['confidences']}
    features={k:TemporalFeatureExtractor(history_size=12) for k in trackers}
    cap=cv2.VideoCapture(str(path));assert cap.isOpened(); frames=[];meta=[];selected=set(indices);decoded=0;processed=0;start=time.monotonic()
    def flush(stream):
        nonlocal processed
        if not frames:return
        results=model.predict(frames,imgsz=config['imgsz'],conf=config['detector_conf'],iou=config['iou'],max_det=config['max_det'],rect=False,half=False,device=device,verbose=False)
        assert len(results)==len(frames)
        for frame,result,index in zip(frames,results,meta):
            raw=[];height,width=frame.shape[:2]
            for xy,cls,conf in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist()):
                raw.append({'class_id':int(cls),'class_name':model.names[int(cls)].upper(),'confidence':float(conf),'bbox_xyxy':[float(v) for v in xy],'track_id':None})
            ts=float((pts[index]-pts[0])*Fraction(base));tracked={}
            for key,tracker in trackers.items():
                boxes=[]
                for d in raw:
                    x1,y1,x2,y2=d['bbox_xyxy'];boxes.append(DetectionBox(d['class_id'],d['class_name'],d['confidence'],tuple(map(int,(x1,y1,x2,y2))),((x1+x2)/2/width,(y1+y2)/2/height,(x2-x1)/width,(y2-y1)/height)))
                tb=tracker.update(boxes,frame);values=[]
                for b in tb:
                    if b.track_id<=0:continue
                    feat=features[key].update(b.track_id,frame,b.xyxy,ts)
                    values.append({'class_id':b.cls_id,'class_name':b.label,'confidence':b.confidence,'bbox_xyxy':b.xyxy,'norm_xywh':b.norm_xywh,'track_id':b.track_id,'features':asdict(feat)})
                tracked[key]=values
            row={**binding,'frame_index':index,'source_pts_ticks':pts[index],'source_pts_sec':float(pts[index]*Fraction(base)),'time_base':base,'timestamp_sec':ts,'timestamp_provenance':'SOURCE_PTS','timestamp_source':'source_pts','detector_input_config':config,'processing_status':'OK','detections':raw,'tracked_by_confidence':tracked}
            stream.write(json.dumps(row)+'\n');processed+=1
        stream.flush();frames.clear();meta.clear()
    try:
        with (folder/'frame_level.jsonl').open('x') as stream:
            while cap.grab():
                if decoded in selected:
                    ok,frame=cap.retrieve();assert ok and frame is not None
                    frames.append(frame);meta.append(decoded)
                    if len(frames)>=config['batch']:flush(stream)
                decoded+=1
                if time.monotonic()-start>14400:raise TimeoutError('Video inference exceeded four hours')
            flush(stream)
    finally:cap.release()
    assert decoded==len(pts),'Native decode count differs from authenticated PTS timeline'
    assert sha256(path)==entry['source_sha256'] and sha256(ROOT/CHECKPOINT)==EXPECTED
    receipt=coverage((json.loads(line) for line in (folder/'frame_level.jsonl').open()),indices,pts,base,binding)
    receipt.update(source_sha256=entry['source_sha256'],config_sha256=manifest['config_sha256'],frame_jsonl_sha256=sha256(folder/'frame_level.jsonl'),native_decoded_frames=decoded,seconds=time.monotonic()-start)
    save(final,receipt);return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['prepare','infer'],required=True);parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--joint-export',type=Path);args=parser.parse_args()
    out=args.run_dir.resolve();assert out.is_relative_to(ROOT/'runs/benchmarks/week4_final_scientific')
    if args.phase=='prepare':prepare(out,args.joint_export);return
    import torch,numpy as np,cv2,ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(4);cv2.setNumThreads(1)
    assert sha256(ROOT/CHECKPOINT)==EXPECTED
    manifest=json.loads((out/'run_manifest.json').read_text());assert manifest['runner_sha256']==sha256(Path(__file__)) and manifest['config_sha256']==sha256(out/'inference_config.json') and manifest['scope_sha256']==sha256(out/'source_scope.json')
    model=YOLO(str(ROOT/CHECKPOINT));model.model.eval();device='cuda:0' if torch.cuda.is_available() else 'cpu'
    assert {int(k):v.lower() for k,v in model.names.items()}=={0:'smoke',1:'fire'}
    entries=json.loads((out/'source_scope.json').read_text())['videos']
    cap=cv2.VideoCapture(str(ROOT/entries[0]['relative_path']));ok,frame=cap.read();cap.release();assert ok
    smoke=model.predict(frame,imgsz=416,conf=.1,iou=.7,rect=False,half=False,device=device,verbose=False)
    save(out/'checkpoint_provenance.json',{'sha256':EXPECTED,'architecture':model.model.__class__.__name__,'names':model.names,'device':device,'torch':torch.__version__,'ultralytics':ultralytics.__version__,'smoke_test_real_source':entries[0]['video_id'],'smoke_test_detections':len(smoke[0].boxes),'smoke_test_pass':True})
    receipts=[]
    # Same locked list for both pipelines; positives first for early end-to-end checks.
    for index,entry in enumerate(sorted(entries,key=lambda e:(e['kind']!='positive',e['duration_seconds'],e['video_id'])),1):
        receipt=infer_video(out,entry,model,device);receipts.append(dict(video_id=entry['video_id'],**receipt));save(out/'video_completion.json',receipts)
        print('INFER',index,len(entries),entry['video_id'],receipt['processed_frames'],flush=True)
    save(out/'integrity_receipt.json',{'complete':True,'videos':len(receipts),'processed_frames':sum(r['processed_frames'] for r in receipts),'source_before_after_verified':True,'checkpoint_sha256':EXPECTED,'scope_sha256':sha256(out/'source_scope.json')})


if __name__=='__main__':main()
