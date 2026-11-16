"""Build compact source review UI; import genuine decisions into a fresh subset version."""
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ground_truth_studio.integrity_service import sha256
from src.scene_subset import build_subset
from src.scene_decision_import import verify_batch

PARENT=ROOT/'runs/benchmarks/week5_domain_adaptation/final_bbox_20261116_57d1b826'
BLOCKED={'3ed5b015393ce74d1da992dece66ce4a4b7abcd4b77ce5d3d23264794c4eacbe'}


def save(path: Path, value: object) -> None:
    """Refuse overwrites of existing evidence."""
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)


def advance(bundle: Path, decisions: list[dict], actor: dict | None, target: Path,
            existing: list[dict] | None = None) -> dict:
    """Import a batch and recalculate deterministic subset; never launch training."""
    scope_path=bundle/'source_priority_report.json';scope=json.loads(scope_path.read_text(encoding='utf-8'))
    scope['scope_sha256']=sha256(scope_path)
    if decisions and actor is None:raise PermissionError('AUTHENTICATED_SCENE_REVIEWER_REQUIRED')
    verified=verify_batch(ROOT,scope,decisions,actor,existing) if decisions else []
    all_decisions=(existing or [])+verified
    manifest_path=PARENT/'candidate_provenance_manifest.json'
    if sha256(manifest_path)!=scope['parent_manifest_sha256']:raise ValueError('PARENT_MANIFEST_CHANGED')
    rows=json.loads(manifest_path.read_text(encoding='utf-8'))['rows']
    for row in rows:
        if row['human_approved'] and row['lineage_verified'] and row['split'] in ('train','val') and row['source_video_sha256'] not in BLOCKED:
            if sha256(ROOT/row['image_path'])!=row['image_sha256'] or sha256(ROOT/row['label_path'])!=row['label_sha256']:
                raise ValueError('ANNOTATION_BYTES_CHANGED')
            if sha256(ROOT/row['frame_lineage_receipt'])!=row['frame_lineage_receipt_sha256']:raise ValueError('LINEAGE_RECEIPT_CHANGED')
    pairs={frozenset(d['member_sha256']) for d in all_decisions if d['classification']=='CONFIRMED_INDEPENDENT'}
    before_pairs={frozenset(d['member_sha256']) for d in (existing or []) if d['classification']=='CONFIRMED_INDEPENDENT'}
    before=build_subset(rows,before_pairs,BLOCKED)
    result=build_subset(rows,pairs,BLOCKED)
    retained={r['source_video_sha256']:r['source_video_path'] for r in result['rows']}
    for source,path in retained.items():
        if sha256(ROOT/path)!=source:raise ValueError('RETAINED_SOURCE_SHA_CHANGED')
    for decision in all_decisions:
        for evidence in decision['evidence']:
            from ground_truth_studio.config import confined
            if sha256(confined(ROOT,evidence['path']))!=evidence['sha256']:raise ValueError('STORED_SCENE_EVIDENCE_CHANGED')
    target.mkdir(parents=True,exist_ok=False)
    save(target/'scene_decisions.json',all_decisions)
    save(target/'decision_receipt.json',{'actor_id':actor['id'] if verified else None,'status':'AUTHENTICATED_DECISIONS_IMPORTED' if verified else 'NO_DECISIONS_SUBMITTED','scope_sha256':scope['scope_sha256'],
         'batch_sha256':sha256(target/'scene_decisions.json'),'accepted_decisions':len(verified)})
    save(target/'eligible_subset_manifest.json',result)
    selected=result['rows'];ready=result['train_count']>0 and result['val_count']>0
    save(target/'training_readiness.json',{'technical_candidate_nonempty':ready,
         'status':'PILOT_CANDIDATE_PENDING_PROTOCOL_VALIDATION' if ready else 'BLOCKED',
         'research_suitability':'NOT_ESTABLISHED','DFIRE_READY':False,'WEEK5_TRAINING_READY':False,
         'pilot_executed':False,'training_runs':0,'delta_train':result['train_count']-before['train_count'],'delta_val':result['val_count']-before['val_count'],
         'reason':'Scene pair evidence is necessary but does not establish sample-size, ontology and D-Fire gates'})
    save(target/'split_leakage_audit.json',{'retained_opposing_pairs_all_evidenced':all(
        frozenset((a['source_video_sha256'],b['source_video_sha256'])) in pairs for a in selected for b in selected if a['split']=='train' and b['split']=='val'),
         'test_rows_retained':sum(r['split']=='test' for r in selected),'parent_manifest_sha256':scope['parent_manifest_sha256']})
    return result


def prepare(output: Path) -> None:
    """Execute ranking and contact sheets from approved exact-lineage candidates."""
    output=output.resolve();output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    parent=PARENT/'candidate_provenance_manifest.json';rows=json.loads(parent.read_text(encoding='utf-8'))['rows']
    groups=defaultdict(list)
    for r in rows:
        if r['human_approved'] and r['lineage_verified'] and r['split'] in ('train','val') and r['source_video_sha256'] not in BLOCKED:
            groups[r['source_video_sha256']].append(r)
    sources={key:{'path':members[0]['source_video_path'],'video_id':members[0]['source_video_id'],
                  'split':members[0]['split'],'count':len(members),
                  'composition':dict(Counter(r['class_composition'] for r in members))} for key,members in groups.items()}
    train=[s for s in sources if sources[s]['split']=='train'];val=[s for s in sources if sources[s]['split']=='val']
    ranked=[]
    for a in train:
        for b in val:
            classes=set(sources[a]['composition'])|set(sources[b]['composition'])
            coverage=int(any(c in classes for c in ('SMOKE_ONLY','BOTH')))+int(any(c in classes for c in ('FIRE_ONLY','BOTH')))+int('NEGATIVE' in classes)
            ranked.append({'member_sha256':[a,b],'class_coverage':coverage,'image_impact':sources[a]['count']+sources[b]['count'],'decision':'UNKNOWN'})
    ranked.sort(key=lambda p:(-p['class_coverage'],-p['image_impact'],p['member_sha256']))
    chosen=[];used=set()
    for pair in ranked:
        if not chosen or any(s not in used for s in pair['member_sha256']):
            chosen.append(pair);used.update(pair['member_sha256'])
        if len(chosen)==6:break
    for index,pair in enumerate(chosen):
        canvas=np.full((440,960,3),240,np.uint8);members=[]
        for column,source in enumerate(pair['member_sha256']):
            candidates=sorted(groups[source],key=lambda r:r['source_frame_index'])
            for n,row in enumerate([candidates[0],candidates[-1]]):
                image=cv2.imread(str(ROOT/row['image_path']))
                if image is None or sha256(ROOT/row['image_path'])!=row['image_sha256']:raise ValueError('IMAGE_CHANGED')
                h,w=image.shape[:2];scale=min(480/w,190/h);small=cv2.resize(image,(int(w*scale),int(h*scale)))
                canvas[n*220:n*220+small.shape[0],column*480:column*480+small.shape[1]]=small
                cv2.putText(canvas,row['source_video_id']+' '+str(row['source_frame_index']),(column*480+2,n*220+210),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
                members.append({'image_sha256':row['image_sha256'],'frame_lineage_receipt_sha256':row['frame_lineage_receipt_sha256']})
        sheet=output/f'pair_{index+1:02d}.jpg';cv2.imwrite(str(sheet),canvas)
        pair['evidence']=[{'kind':'VISUAL_SCENE_REVIEW','path':sheet.relative_to(ROOT).as_posix(),'sha256':sha256(sheet)}]
        pair['frame_evidence']=members;pair['scene_notes']='Actual first/last verified frames; camera/source independence UNKNOWN until reviewer establishes whole-source context.'
    report={'parent_manifest_sha256':sha256(parent),'sources':sources,'pairs':chosen,
            'ranking':'Class coverage, candidate count, source diversity, SHA tie-break; no model metrics',
            'all_opposing_pairs':len(ranked),'selected_for_human_review':len(chosen)}
    save(output/'source_priority_report.json',report);scope_hash=sha256(output/'source_priority_report.json')
    save(output/'scene_decisions.json',[])
    save(output/'baseline_protocol.json',json.loads((PARENT/'new_baseline_protocol.json').read_text(encoding='utf-8')))
    advance(output,[],None,output/'initial_gate_check')
    document=['<meta charset="utf-8"><title>Week5 source scene review</title><h1>Review 6 source pairs</h1><p>Không cần duyệt 1380 quan hệ. Không xác nhận độc lập chỉ từ khác hình ảnh: xem source/context, giữ UNKNOWN nếu thiếu chứng cứ.</p><label>Studio reviewer ID <input id="reviewer"></label><p>Nhập đúng ID tài khoản đăng nhập Studio; upload sẽ kiểm server-side.</p>']
    for i,pair in enumerate(chosen):
        details=[sources[s] for s in pair['member_sha256']]
        document.append(f'<section><h2>Pair {i+1} · {pair["image_impact"]} candidates</h2><pre>'+html.escape(json.dumps(details,ensure_ascii=False,indent=2))+f'</pre><img width="960" src="pair_{i+1:02d}.jpg"><p>Source SHA: '+html.escape(str(pair['member_sha256']))+f'</p><select id="d{i}"><option>UNKNOWN</option><option>CONFIRMED_INDEPENDENT</option><option>CONFIRMED_OVERLAP</option></select><p>Lý do và căn cứ độc lập/toàn footage:</p><textarea id="r{i}" rows="3" cols="100"></textarea></section>')
    document.append('<p>Decision không tự tạo GT hay mở training. Bằng chứng ảnh đại diện chỉ hỗ trợ review; cần xác định source/camera/context đủ để kết luận.</p><button onclick="download()">Xuất decision batch để upload Studio</button><script>const pairs='+json.dumps(chosen)+';const scope='+json.dumps(scope_hash)+''';function download(){let id=document.getElementById('reviewer').value.trim();if(!id){alert('Reviewer ID required');return;}let rows=pairs.map((p,i)=>({...p,classification:document.getElementById('d'+i).value,scene_reason:document.getElementById('r'+i).value,reviewer_id:id,reviewed_at_utc:new Date().toISOString(),scope_sha256:scope})).filter(p=>p.classification!=='UNKNOWN');if(rows.some(p=>!p.scene_reason.trim())){alert('Evidence reason required');return;}let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{type:'application/json'}));a.download='scene_decisions.json';a.click();}</script>''')
    with (output/'human_scene_review.html').open('x',encoding='utf-8') as stream:stream.write('\n'.join(document))
    print(json.dumps({'review_pairs':len(chosen),'candidate_sources':len(sources),'initial_train':0,'initial_val':0}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.output)
