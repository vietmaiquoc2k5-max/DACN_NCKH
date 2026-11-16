"""Human-only quarantine resolution leaves every original forensic byte intact."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ground_truth_studio.bbox_snapshot import scan_reviews, committed_reviews, snapshot_studio
from ground_truth_studio.integrity_service import sha256


@pytest.fixture
def service():
    from test_bbox_qa import TestBBoxQA
    fixture=TestBBoxQA();fixture.setUp()
    try:yield fixture
    finally:fixture.doCleanups()


def corrupt(service):
    record=service.submit()
    folder=Path(record['label_version_path']).parent
    (folder/'review.sha256').write_bytes(bytes(64))
    return folder,{p.name:p.read_bytes() for p in folder.iterdir()}


def resolve(service,**changes):
    args=dict(token='SYNTHETIC_TOKEN',identity=service.row['image_sha256'],boxes=service.box,
              decision='APPROVE',evidence='SYNTHETIC_HUMAN_RECONFIRMATION',attest=True,negative=False,reconfirm=True)
    args.update(changes);return service.service.resolve(**args)


def test_resolution_requires_explicit_authenticated_action(service):
    corrupt(service)
    with pytest.raises(ValueError,match='EXPLICIT_RECONFIRMATION'):resolve(service,reconfirm=False)
    service.repository.actor.side_effect=PermissionError('SYNTHETIC_AUTH_REQUIRED')
    with pytest.raises(PermissionError):resolve(service)


def test_resolution_restores_new_decision_without_authenticating_old_checksum(service):
    folder,before=corrupt(service)
    assert service.service.history('TOKEN')==[]
    receipt=resolve(service)
    assert receipt['affected_versions']=={folder.name:{k:sha256(folder/k) for k in before}}
    assert receipt['original_checksum_authenticated'] is False
    records,issues=scan_reviews(service.root/'private')
    assert len(records)==1 and records[0]['version_id']==receipt['replacement_version_id']
    assert issues[0]['status']=='RESOLVED_BY_HUMAN_RECONFIRMATION'
    assert before=={p.name:p.read_bytes() for p in folder.iterdir()}
    assert len(committed_reviews(service.root/'private'))==1
    assert json.loads((service.service.export('TOKEN')/'bbox_export.json').read_text())['human_approved_count']==1


def test_all_images_must_be_resolved_before_global_export(service):
    folder,_=corrupt(service)
    # A second unrelated malformed version remains globally blocking.
    other=service.root/'private/bbox_qa/versions/unrelated';other.mkdir()
    (other/'review.json').write_text('{}');(other/'review.sha256').write_text('bad')
    resolve(service)
    with pytest.raises(ValueError):service.service.export('TOKEN')


@pytest.mark.parametrize('target',['receipt','original','replacement'])
def test_changed_evidence_relocks_export(service,target):
    folder,_=corrupt(service);receipt=resolve(service)
    if target=='receipt':
        path=service.root/'private/bbox_qa/resolutions'/receipt['resolution_id']/'resolution.json'
    elif target=='original':path=folder/'review.sha256'
    else:path=service.root/'private/bbox_qa/versions'/receipt['replacement_version_id']/'labels.txt'
    path.write_text('changed')
    with pytest.raises(ValueError):service.service.export('TOKEN')


def test_rejection_resolution_is_not_approval_and_backup_preserves_forensics(service):
    folder,before=corrupt(service);resolve(service,decision='REJECT')
    result=json.loads((service.service.export('TOKEN')/'bbox_export.json').read_text())
    assert result['human_approved_count']==0 and result['rejected_count']==1
    target=service.root/'snapshot';snapshot_studio(service.root/'private',target)
    assert all((target/'bbox_qa/versions'/folder.name/k).read_bytes()==v for k,v in before.items())
    assert list((target/'bbox_qa/resolutions').glob('*/resolution.sha256'))
    assert committed_reviews(target,original_private=service.root/'private')[0]['decision']=='REJECT'


def test_resolution_empty_label_requires_negative_attestation(service):
    corrupt(service)
    with pytest.raises(ValueError,match='NEGATIVE_APPROVAL'):resolve(service,boxes=[])
    resolve(service,boxes=[],negative=True)
    assert committed_reviews(service.root/'private')[0]['negative_frame_approved']


def test_ui_reconfirmation_requires_human_controls_and_saves_only_synthetic_receipt(service):
    import sys
    from streamlit.testing.v1 import AppTest
    folder,before=corrupt(service)
    service.row['image_name']='SYNTHETIC_QUARANTINE'
    main=sys.modules['__main__'];original=getattr(main,'__file__',None)
    script='''from ground_truth_studio.bbox_page import bbox_page
if __name__ == "__main__":
    bbox_page(None,"SYNTHETIC_TOKEN")
'''
    key=service.row['image_sha256']
    try:
        with patch('ground_truth_studio.bbox_page.BBoxReviewService',return_value=service.service):
            app=AppTest.from_string(script,default_timeout=30).run()
            app.checkbox(key='bbox_resolution_mode').check().run()
            assert not app.exception
            assert app.checkbox(key='reconfirm_'+key).value is False
            app.text_input(key='evidence_'+key).set_value('SYNTHETIC_HUMAN_RECONFIRMATION')
            app.checkbox(key='attest_'+key).check()
            app.checkbox(key='reconfirm_'+key).check()
            next(b for b in app.button if b.label=='Phê duyệt phiên bản bbox').click().run()
            assert not app.exception
            assert len(committed_reviews(service.root/'private'))==1
            assert list((service.root/'private/bbox_qa/resolutions').glob('*/resolution.sha256'))
            assert before=={p.name:p.read_bytes() for p in folder.iterdir()}
    finally:
        if original is not None:main.__file__=original
