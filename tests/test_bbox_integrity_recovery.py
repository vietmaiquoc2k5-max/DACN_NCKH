"""Corruption isolation preserves original bytes and never invents approvals."""
import json
from pathlib import Path

import pytest
import sys
from types import SimpleNamespace
from unittest.mock import Mock,patch

from ground_truth_studio.bbox_snapshot import scan_reviews, committed_reviews
from ground_truth_studio.integrity_service import sha256


@pytest.fixture
def storage():
    from test_week5_readiness import TestWeek5Readiness as _Fixture
    fixture=_Fixture();fixture.setUp()
    try:yield fixture
    finally:fixture.doCleanups()


@pytest.mark.parametrize('kind',['metadata','missing_checksum','partial','json','label','missing_metadata','path'])
def test_one_invalid_among_valid_is_visible_and_unchanged(storage,kind):
    good=storage.synthetic_review(1);bad=storage.synthetic_review(2)
    folder=storage.root/'private/bbox_qa/versions/2'
    if kind=='metadata':(folder/'review.sha256').write_bytes(bytes(64))
    elif kind=='missing_checksum':(folder/'review.sha256').unlink()
    elif kind=='partial':(folder/'review.json').write_text('{')
    elif kind=='json':
        (folder/'review.json').write_text('{');(folder/'review.sha256').write_text(sha256(folder/'review.json'))
    elif kind=='label':(folder/'labels.txt').write_text('changed')
    elif kind=='missing_metadata':(folder/'review.json').unlink()
    else:
        bad['label_version_path']='wrong';(folder/'review.json').write_text(json.dumps(bad));(folder/'review.sha256').write_text(sha256(folder/'review.json'))
    before={p.name:p.read_bytes() for p in folder.iterdir()}
    records,issues=scan_reviews(storage.root/'private')
    assert [r['version_id'] for r in records]==['1']
    assert len(issues)==1 and issues[0]['version_id']=='2'
    assert issues[0]['status']=='INTEGRITY_QUARANTINED'
    assert before=={p.name:p.read_bytes() for p in folder.iterdir()}
    if kind!='missing_checksum':
        with pytest.raises(ValueError):committed_reviews(storage.root/'private')


def test_corrupt_newer_version_never_resurrects_old_approval(storage):
    good=storage.synthetic_review(1);bad=storage.synthetic_review(2)
    folder=storage.root/'private/bbox_qa/versions/2'
    bad['image_sha256']=good['image_sha256']
    (folder/'review.json').write_text(json.dumps(bad))
    records,issues=scan_reviews(storage.root/'private')
    assert records==[] and issues


def test_valid_only_scan_remains_strict_and_authenticated(storage):
    good=storage.synthetic_review(1)
    assert committed_reviews(storage.root/'private')==[good]


def test_authenticated_bbox_ui_survives_one_quarantined_version(storage):
    from ground_truth_studio.bbox_service import BBoxReviewService
    from streamlit.testing.v1 import AppTest
    good=storage.synthetic_review(1);storage.synthetic_review(2)
    (storage.root/'private/bbox_qa/versions/2/review.sha256').write_bytes(bytes(64))
    queue=dict(good,image_name='SYNTHETIC',label_path=Path(good['label_version_path']).relative_to(storage.root).as_posix(),
               aliases=[],reserved_split='TRAIN_CANDIDATE')
    repo=Mock(settings=SimpleNamespace(repo=storage.root,private=storage.root/'private'))
    repo.actor.return_value={'id':'SYNTHETIC','role':'annotator_01'}
    with patch('ground_truth_studio.bbox_service.load_queue',return_value=[queue]):service=BBoxReviewService(repo)
    main=sys.modules['__main__'];before=getattr(main,'__file__',None)
    try:
        with patch('ground_truth_studio.bbox_page.BBoxReviewService',return_value=service):
            app=AppTest.from_string('''from ground_truth_studio.bbox_page import bbox_page
if __name__ == "__main__":
    bbox_page(None,"SYNTHETIC_TOKEN")
''',default_timeout=30).run()
            assert not app.exception
            assert any('INTEGRITY_QUARANTINED' in w.value for w in app.warning)
            assert any('Phê duyệt' in b.label for b in app.button)
            assert len(service.history('SYNTHETIC_TOKEN'))==1
    finally:
        if before is not None:main.__file__=before
