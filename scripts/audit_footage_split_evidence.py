"""Decode prioritized footage and screen multiple evidence types; never auto-sign independence."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ground_truth_studio.integrity_service import sha256
from src.preannotation import fingerprint


def main() -> None:
    """Decode to EOF and save sampled overview/duplicate hypotheses in a fresh directory."""
    out=ROOT/'.tools/week5_footage_evidence_20261116';out.mkdir(exist_ok=False)
    cv2.setNumThreads(1);orb=cv2.ORB_create(nfeatures=600);frames={};receipts=[]
    for video in ['VN_POS_05','VN_POS_08','VN_POS_13']:
        path=ROOT/'datasets/negative_videos'/f'{video}.mp4';before=sha256(path)
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);count=0;samples=[]
        if not cap.isOpened() or fps<=0:raise ValueError('VIDEO_OPEN_OR_FPS_FAILURE')
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                if count%max(1,round(fps*2))==0:
                    h,w=frame.shape[:2];small=cv2.resize(frame,(320,max(8,int(h*320/w))))
                    fp=fingerprint(small)['variants'][0];kp,desc=orb.detectAndCompute(small,None)
                    samples.append({'index':count,'elapsed_index_over_fps':count/fps,'pixels_sha256':hashlib.sha256(frame.tobytes()).hexdigest(),
                                    'fingerprint':fp,'image':small,'points':np.float32([k.pt for k in kp]),'descriptors':desc})
                count+=1
        finally:cap.release()
        assert sha256(path)==before
        sheets=[]
        for offset in range(0,len(samples),16):
            sheet=np.full((4*210,4*320,3),245,np.uint8)
            for j,row in enumerate(samples[offset:offset+16]):
                image=row['image'];h,w=image.shape[:2];scale=min(320/w,180/h);im=cv2.resize(image,(int(w*scale),int(h*scale)))
                y,x=(j//4)*210,(j%4)*320;sheet[y:y+im.shape[0],x:x+im.shape[1]]=im
                cv2.putText(sheet,f"{video} frame={row['index']}",(x+2,y+202),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
            target=out/f'{video}_{offset//16:02d}.jpg';cv2.imwrite(str(target),sheet)
            sheets.append({'path':target.relative_to(ROOT).as_posix(),'sha256':sha256(target)})
        frames[video]=samples
        receipts.append({'video_id':video,'source_sha256':before,'opencv_decoded_frames':count,'fps':fps,'sample_interval_seconds':2,
                         'sampled_frames':len(samples),'contact_sheets':sheets,
                         'termination':'OpenCV sequential read ended; prior independent technical receipt still required for clean EOF'})
        print(json.dumps(receipts[-1]),flush=True)
    results=[];matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    for train in ['VN_POS_05','VN_POS_13']:
        exact=[];near=[];geometry=[];tested=0
        for a in frames[train]:
            for b in frames['VN_POS_08']:
                tested+=1
                if a['pixels_sha256']==b['pixels_sha256']:exact.append([a['index'],b['index']])
                p,q=a['fingerprint'],b['fingerprint'];ph=(int(p['phash'],16)^int(q['phash'],16)).bit_count();dh=(int(p['dhash'],16)^int(q['dhash'],16)).bit_count()
                if min(p['std'],q['std'])>=10 and ph<=6 and dh<=8:near.append([a['index'],b['index'],ph,dh])
                if a['descriptors'] is None or b['descriptors'] is None or len(b['descriptors'])<2:continue
                good=[pair[0] for pair in matcher.knnMatch(a['descriptors'],b['descriptors'],k=2) if len(pair)==2 and pair[0].distance<.75*pair[1].distance]
                if len(good)>=8:
                    matrix,mask=cv2.findHomography(a['points'][[m.queryIdx for m in good]],b['points'][[m.trainIdx for m in good]],cv2.RANSAC,3.0)
                    if matrix is not None and int(mask.sum())>=8 and float(mask.mean())>=.6:geometry.append([a['index'],b['index'],int(mask.sum())])
        results.append({'train':train,'validation':'VN_POS_08','sample_pairs_tested':tested,'sample_exact_pixel_matches':exact,
                        'perceptual_hypotheses':near,'orb_ransac_hypotheses':geometry,
                        'automatic_independence_decision':'UNKNOWN','limitation':'No match in sampled evidence is not proof of absence of all reused footage or same-camera scenes'})
    (out/'footage_screen.json').write_text(json.dumps({'source_receipts':receipts,'comparisons':results},indent=2),encoding='utf-8')
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
