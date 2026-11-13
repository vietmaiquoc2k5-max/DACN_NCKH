"""Independently revalidate frame records, tick/second PTS, tracks and scope hashes."""
from __future__ import annotations
import argparse,json,sys,math
from fractions import Fraction
from pathlib import Path
from typing import Any,Iterator
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from ground_truth_studio.integrity_service import sha256
from src.scientific_inference import sample_indices,coverage
from scripts.run_week4_final_scientific import save


def main() -> None:
    """Validate a completed local run and write its verification receipt."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args();out=args.run_dir.resolve()
    assert out.is_relative_to(ROOT/'runs/benchmarks/week4_final_scientific')
    manifest=json.loads((out/'run_manifest.json').read_text());config=json.loads((out/'inference_config.json').read_text())
    assert manifest['scope_sha256']==sha256(out/'source_scope.json') and manifest['config_sha256']==sha256(out/'inference_config.json')
    protocol=json.loads((out/'experiment_protocol_lock.json').read_text())
    assert all(sha256(ROOT/n)==h for n,h in protocol['code_sha256'].items()),'Predeclared experiment implementation changed'
    integrity=json.loads((out/'integrity_receipt.json').read_text());assert integrity['complete'] is True
    assert integrity['scope_sha256']==manifest['scope_sha256']
    aggregate=json.loads((out/'video_completion.json').read_text());aggregate_by_id={r['video_id']:r for r in aggregate}
    sources=json.loads((out/'source_scope.json').read_text())['videos'];summary=[]
    assert len({e['video_id'] for e in sources})==len(sources)==len(aggregate)
    assert len({e['source_sha256'] for e in sources})==len(sources),'Exact source duplicate'
    for entry in sources:
        vid=entry['video_id'];folder=out/'videos'/vid;receipt=json.loads((folder/'video_completion.json').read_text());ref=json.loads((out/'pts'/(vid+'.json')).read_text());pts=ref['pts'];base=ref['time_base'];indices=sample_indices(pts,base)
        assert aggregate_by_id[vid]=={'video_id':vid,**receipt}
        assert len(pts)==entry['native_frames']==receipt['native_decoded_frames']
        assert len(indices)==entry['expected_samples']==receipt['processed_frames']
        assert sha256(ROOT/entry['relative_path'])==entry['source_sha256']==receipt['source_sha256']
        assert receipt['config_sha256']==manifest['config_sha256']
        assert receipt['frame_jsonl_sha256']==sha256(folder/'frame_level.jsonl')
        binding={'run_id':out.name,'video_id':vid,'camera_id':'BENCH_'+vid,'source_sha256':entry['source_sha256'],'checkpoint_sha256':manifest['checkpoint_sha256'],'config_sha256':manifest['config_sha256'],'sampling_protocol':config['sampling_protocol']}
        observed={'empty_detection_frames':0,'raw_detections':0,'track_observations':{str(c):0 for c in config['confidences']}}
        def verify_detection(d: dict[str,Any]) -> None:
            """Reject invalid class, confidence or image coordinates."""
            assert d['class_id'] in (0,1) and d['class_name']=={0:'SMOKE',1:'FIRE'}[d['class_id']] and 0<=d['confidence']<=1
            box=d['bbox_xyxy'];assert len(box)==4 and all(math.isfinite(x) for x in box)
            assert box[0]<box[2] and box[1]<box[3]
        def records() -> Iterator[dict[str,Any]]:
            """Yield real observations after source-time and feature checks."""
            for line in (folder/'frame_level.jsonl').open():
                row=json.loads(line);index=row['frame_index']
                assert 0<=index<len(pts)
                assert row['source_pts_sec']==float(pts[index]*Fraction(base)) and row['timestamp_sec']==float((pts[index]-pts[0])*Fraction(base))
                assert row['timestamp_provenance']=='SOURCE_PTS'
                assert row['detector_input_config']==config
                observed['empty_detection_frames']+=not row['detections'];observed['raw_detections']+=len(row['detections'])
                for d in row['detections']:
                    verify_detection(d);assert d['track_id'] is None
                assert set(row['tracked_by_confidence'])==set(observed['track_observations'])
                for confidence,tracks in row['tracked_by_confidence'].items():
                    seen=set()
                    for d in tracks:
                        verify_detection(d)
                        assert isinstance(d['track_id'],int) and d['track_id']>0 and d['track_id'] not in seen
                        seen.add(d['track_id'])
                        assert len(d['norm_xywh'])==4 and all(math.isfinite(v) for v in d['norm_xywh'])
                        assert set(d['features'])=={'area_growth_rate','centroid_speed','roi_intensity_variance','flicker_score','smoke_expansion_score'}
                        assert all(math.isfinite(v) for v in d['features'].values())
                    observed['track_observations'][confidence]+=len(tracks)
                yield row
        result=coverage(records(),indices,pts,base,binding);result.update(video_id=vid,**observed);summary.append(result)
    checkpoint=ROOT/manifest['checkpoint'];assert sha256(checkpoint)==manifest['checkpoint_sha256']==integrity['checkpoint_sha256']
    assert len(summary)==integrity['videos'] and sum(r['processed_frames'] for r in summary)==integrity['processed_frames']
    save(out/'frame_evidence_verification.json',{'complete':True,'videos':len(summary),'processed_frames':sum(r['processed_frames'] for r in summary),'expected_frames':sum(r['expected_frames'] for r in summary),'native_decoded_frames':sum(e['native_frames'] for e in sources),'empty_detection_frames':sum(r['empty_detection_frames'] for r in summary),'pts_ticks_and_seconds_verified':True,'predeclared_code_unchanged':True,'tracking_features_verified':True,'exact_source_duplicate_count':0,'near_duplicate_and_scene_independence':'NOT_VERIFIED','details':summary})
    print('VERIFIED_VIDEOS',len(summary),'VERIFIED_FRAMES',sum(r['processed_frames'] for r in summary))


if __name__=='__main__':main()
