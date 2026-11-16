"""Proposed welding guidance cannot silently become labels or incident GT."""
import pytest

from src.annotation_policy import proposed_guidance, require_policy_approval


@pytest.mark.parametrize('category',['WELDING_SPARKS','WELDING_FUME','AMBIGUOUS','ACTUAL_FIRE_INCIDENT'])
def test_observation_is_not_automatic_fire_smoke_bbox(category):
    result=proposed_guidance(category)
    assert result['suggested_frame_class_id'] is None
    assert result['review_status']=='HUMAN_REVIEW_REQUIRED'
    assert not result['auto_approved'] and not result['rewrite_existing_labels']


def test_controlled_flame_proposal_is_not_confirmed_fire_incident():
    result=proposed_guidance('CONTROLLED_FLAME')
    assert result['suggested_frame_class_id']==1
    assert result['confirmed_fire_incident'] is None
    assert result['policy_status']=='PROPOSED_PENDING_HUMAN_APPROVAL'


def test_unapproved_policy_does_not_authorize_label_migration():
    with pytest.raises(ValueError,match='HUMAN_POLICY_APPROVAL_REQUIRED'):require_policy_approval(None)
    with pytest.raises(ValueError):require_policy_approval({'decision':'APPROVE'})
