"""Final queue accounting must preserve rejection, pending and duplicate history."""
import pytest
from src.bbox_final_audit import summarize_decisions


def row(image,version,time,decision='APPROVE'):
    return dict(image_sha256=image,version_id=version,reviewed_at_utc=time,decision=decision)


def test_final_reviewed_does_not_imply_all_approved_or_ready():
    result=summarize_decisions({'a','b'},[row('a','1','01'),row('a','2','02','REJECT'),row('b','3','03')])
    assert result['reviewed_unique_images']==2 and result['pending']==0
    assert result['approved']==result['rejected']==1
    assert result['duplicate_history_versions']==1 and not result['training_eligible']


def test_missing_decision_is_pending_and_order_is_deterministic():
    rows=[row('a','2','02'),row('a','1','01','REJECT')]
    assert summarize_decisions({'a','b'},rows)==summarize_decisions({'a','b'},list(reversed(rows)))
    assert summarize_decisions({'a','b'},rows)['pending']==1


def test_unknown_identity_and_decision_fail_closed():
    with pytest.raises(ValueError,match='OUTSIDE_FROZEN_QUEUE'):summarize_decisions({'a'},[row('b','1','01')])
    with pytest.raises(ValueError,match='UNSUPPORTED'):summarize_decisions({'a'},[row('a','1','01','AUTO')])
