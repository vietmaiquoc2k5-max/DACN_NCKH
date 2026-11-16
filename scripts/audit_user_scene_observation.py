"""Bind a supplied scoped observation to existing contact sheets and inspect metadata."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ground_truth_studio.config import Settings
from ground_truth_studio.integrity_service import sha256
from src.scoped_scene_observation import record_observation
from src.scene_subset import build_subset


def main() -> None:
    """Execute a new read-only mapping/metadata audit; do not create human decisions."""
    out=ROOT/'.tools/week5_user_observation_20261116/evidence';out.mkdir(exist_ok=False)
    parent=ROOT/'runs/benchmarks/week5_domain_adaptation/scene_20261115_72a6d1eb/human_source_shortlist.json'
    pairs=json.loads(parent.read_text(encoding='utf-8'))[:12]
    rows=json.loads((ROOT/'runs/benchmarks/week5_domain_adaptation/final_bbox_20261116_57d1b826/candidate_provenance_manifest.json').read_text(encoding='utf-8'))['rows']
    by_image={r['image_sha256']:r for r in rows};sources={};mapping=[]
    ffprobe=Settings.load().ffprobe
    if not ffprobe:raise ValueError('FFPROBE_REQUIRED')
    with (ROOT/'data/vn_pos_scene_groups.csv').open(encoding='utf-8-sig') as stream:scene=list(csv.DictReader(stream))
    negative=json.loads((ROOT/'data/negative_videos_manifest.json').read_text(encoding='utf-8'))['videos']
    for index,pair in enumerate(pairs,1):
        sheet=pair['contact_sheet'];assert sha256(ROOT/sheet['path'])==sheet['sha256']
        members=[]
        for member in sheet['members']:
            assert sha256(ROOT/member['image_path'])==member['image_sha256']
            row=by_image.get(member['image_sha256'],{})
            members.append(dict(member,decision=row.get('decision'),lineage_status=row.get('lineage_status'),
                                reservation=row.get('original_reservation'),split=row.get('split')))
        for source in pair['sources']:
            key=source['sha256']
            if key in sources:continue
            path=ROOT/source['paths'][0];assert sha256(path)==key
            command=[ffprobe,'-v','error','-show_format','-show_streams','-of','json',str(path)]
            result=subprocess.run(command,capture_output=True,timeout=60)
            metadata=json.loads(result.stdout) if result.returncode==0 else {}
            (out/(key+'.probe.json')).write_bytes(result.stdout)
            ids=set(source['video_ids'])
            sources[key]={'source_sha256':key,'paths':source['paths'],'video_ids':source['video_ids'],
                'probe_exit_code':result.returncode,'probe_stdout_sha256':sha256(out/(key+'.probe.json')),
                'format_tags':metadata.get('format',{}).get('tags',{}),
                'stream_tags':[s.get('tags',{}) for s in metadata.get('streams',[])],
                'scene_catalog_rows':[s for s in scene if s['video_id'] in ids],
                'negative_audit_rows':[{k:v for k,v in s.items() if k in ['video_id','sha256','decode_status','scenario_category']} for s in negative if s['video_id'] in ids],
                'authenticated_camera_or_origin_evidence_found':False,
                'reason':'Container/codec/encoder and source hash authenticate bytes, not camera/scene origin; catalog camera fields absent or source relation uncertain'}
        mapping.append({'sheet_number':index,'sheet_path':sheet['path'],'sheet_sha256':sheet['sha256'],
            'source_sha256':pair['source_pair'],'source_records':pair['sources'],'displayed_members':members,
            'visual_observation':'USER_REPORTED_VISUAL_NON_OVERLAP','whole_source_decision':'UNKNOWN',
            'missing_evidence':'Authenticated camera/scene/footage coverage beyond displayed representative frames'})
    statement='Tôi đã hoàn thành BBox QA 1.922 ảnh và trực tiếp xem toàn bộ 12 contact sheets tại `.tools/week5_scene_72a6d1eb/contacts/index.html`. Tôi xác nhận rằng theo quan sát trực quan, các cảnh trong 12 contact sheets không trùng nhau. Không yêu cầu tôi xem lại hoặc nhấn xác nhận hàng loạt thêm một lần nữa.'
    observation=record_observation(statement,[{'path':m['sheet_path'],'sha256':m['sheet_sha256']} for m in mapping])
    observation['interpretation_limit']='Preserve statement verbatim; no inference to unshown source combinations or whole edited videos. Sources may recur across sheets.'
    for name,value in [('user_observation.json',observation),('contact_sheet_mapping.json',mapping),('source_metadata_audit.json',sources)]:
        with (out/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    blocked={'3ed5b015393ce74d1da992dece66ce4a4b7abcd4b77ce5d3d23264794c4eacbe'}
    subset=build_subset(rows,set(),blocked)
    with (out/'subset_result.json').open('x',encoding='utf-8') as f:json.dump(subset,f,indent=2)
    print(json.dumps({'sheets_mapped':len(mapping),'unique_sources_inspected':len(sources),
        'metadata_probe_failures':sum(s['probe_exit_code']!=0 for s in sources.values()),
        'independence_proven_pairs':0,'train':subset['train_count'],'val':subset['val_count'],
        'new_reconfirmation_requests':0}))


if __name__=='__main__':main()
