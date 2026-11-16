"""Authenticated scene evidence import is fail closed, deterministic and conflict aware."""
from datetime import datetime,timezone
from pathlib import Path
import pytest
from src.scene_decision_import import verify_batch
from ground_truth_studio.integrity_service import sha256


@pytest.fixture
def case(tmp_path):
    a=tmp_path/'a.mp4';b=tmp_path/'b.mp4';e=tmp_path/'sheet.jpg'
    a.write_bytes(b'syntheticA');b.write_bytes(b'syntheticB');e.write_bytes(b'SYNTHETIC EVIDENCE')
    pair=[sha256(a),sha256(b)]
    scope={'scope_sha256':'context','pairs':[{'member_sha256':pair}],
           'sources':{pair[0]:{'path':a.name},pair[1]:{'path':b.name}}}
    actor={'id':'SYNTHETIC','role':'annotator_01'}
    decision={'member_sha256':pair,'reviewer_id':actor['id'],'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),
       'scope_sha256':'context','classification':'CONFIRMED_INDEPENDENT','scene_reason':'SYNTHETIC ONLY',
       'evidence':[{'path':e.name,'sha256':sha256(e),'kind':'VISUAL_SCENE_REVIEW'}]}
    return tmp_path,scope,actor,decision


def test_valid_evidence_import_and_conflict_rejection(case):
    root,scope,actor,d=case
    assert verify_batch(root,scope,[d],actor)==[d]
    with pytest.raises(ValueError,match='DUPLICATE_OR_CONFLICTING'):verify_batch(root,scope,[d,d],actor)
    with pytest.raises(ValueError,match='DUPLICATE_OR_CONFLICTING'):verify_batch(root,scope,[d],actor,[d])


@pytest.mark.parametrize('field,value',[('classification','UNKNOWN'),('scope_sha256','bad'),('reviewed_at_utc','2026-11-16'),('scene_reason','')])
def test_unknown_bad_context_time_or_reason_rejected(case,field,value):
    root,scope,actor,d=case;d[field]=value
    with pytest.raises(ValueError):verify_batch(root,scope,[d],actor)


def test_identity_source_and_evidence_hash_enforced(case):
    root,scope,actor,d=case
    with pytest.raises(PermissionError):verify_batch(root,scope,[d],dict(actor,id='OTHER'))
    (root/'a.mp4').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SOURCE_SHA'):verify_batch(root,scope,[d],actor)
    (root/'sheet.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError,match='EVIDENCE_HASH'):verify_batch(root,scope,[d],actor)


def test_transitive_overlap_cannot_be_declared_independent(case):
    root,scope,actor,d=case;c=root/'c.mp4';c.write_bytes(b'SYNTHETIC_C');cs=sha256(c)
    a,b=d['member_sha256'];scope['sources'][cs]={'path':c.name}
    scope['pairs']+=[{'member_sha256':[b,cs]},{'member_sha256':[a,cs]}]
    decisions=[dict(d,classification='CONFIRMED_OVERLAP'),
               dict(d,member_sha256=[b,cs],classification='CONFIRMED_OVERLAP'),
               dict(d,member_sha256=[a,cs])]
    with pytest.raises(ValueError,match='TRANSITIVE_SCENE'):verify_batch(root,scope,decisions,actor)


def test_authenticated_batch_advances_subset_but_never_training(case,monkeypatch):
    import json
    import scripts.prepare_week5_deadline as module
    root,scope,actor,d=case;parent=root/'parent';parent.mkdir();rows=[]
    for index,source in enumerate(d['member_sha256']):
        image=root/f'{index}.jpg';image.write_bytes(b'SYNTHETIC_IMAGE'+bytes([index]))
        label=root/f'{index}.txt';label.write_text('0 .5 .5 .2 .2')
        proof=root/f'{index}.proof.json';proof.write_text('SYNTHETIC_LINEAGE_RECEIPT')
        rows.append(dict(image_sha256=sha256(image),image_path=image.name,label_path=label.name,label_sha256=sha256(label),
            source_video_sha256=source,source_video_path=scope['sources'][source]['path'],
            split='train' if index==0 else 'val',human_approved=True,lineage_verified=True,
            frame_lineage_receipt=proof.name,frame_lineage_receipt_sha256=sha256(proof)))
    manifest=parent/'candidate_provenance_manifest.json';manifest.write_text(json.dumps({'rows':rows}))
    scope['parent_manifest_sha256']=sha256(manifest);bundle=root/'bundle';bundle.mkdir()
    scopepath=bundle/'source_priority_report.json';scopepath.write_text(json.dumps(scope));d['scope_sha256']=sha256(scopepath)
    monkeypatch.setattr(module,'ROOT',root);monkeypatch.setattr(module,'PARENT',parent)
    target=root/'result';result=module.advance(bundle,[d],actor,target)
    assert result['train_count']==result['val_count']==1
    ready=json.loads((target/'training_readiness.json').read_text())
    assert ready['delta_train']==ready['delta_val']==1 and not ready['WEEK5_TRAINING_READY']
