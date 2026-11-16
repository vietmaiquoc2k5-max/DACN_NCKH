"""A real scoped user statement must not be promoted into invented reviewer signoff."""
import pytest
from src.scoped_scene_observation import record_observation


def test_visual_statement_preserved_without_identity_or_time_invention():
    text='Tôi đã xem toàn bộ 12 contact sheets; theo quan sát trực quan các cảnh không trùng.'
    r=record_observation(text,[{'path':'synthetic.jpg','sha256':'synthetic'}])
    assert r['statement']==text and r['studio_reviewer_id'] is None and r['review_completed_at'] is None
    assert r['whole_source_independence']=='UNKNOWN' and not r['training_eligibility_granted']
    assert not r['reconfirmation_requested']


def test_empty_statement_or_scope_is_rejected():
    with pytest.raises(ValueError):record_observation('',[{}])
    with pytest.raises(ValueError):record_observation('User statement',[])
