"""Scene evidence, fixed reservations and baseline membership contracts."""
import pytest
from src.scene_subset import build_subset, training_membership, validate_scene_decision


def rows():
    return [dict(image_sha256=str(i),split=s,human_approved=True,lineage_verified=True,
                 source_video_sha256=str(i),image_path=f'{i}.jpg',label_path=f'{i}.txt',
                 label_sha256='label',scene_group_id=f'scene{i}') for i,s in enumerate(['train','val','test'])]


def test_unknown_opposing_sources_never_form_ready_subset():
    result=build_subset(rows(),set(),set())
    assert result['train_count']==result['val_count']==0
    assert any('RESERVED_TEST_OR_UNASSIGNED' in r['reasons'] for r in result['excluded'])


def test_evidenced_pair_keeps_reservations_and_excludes_quarantined_source():
    result=build_subset(rows(),{frozenset(('0','1'))},set())
    assert result['train_count']==result['val_count']==1
    assert {r['split'] for r in result['rows']}=={'train','val'}
    assert not build_subset(rows(),{frozenset(('0','1'))},{'0'})['train_count']


def test_same_source_cannot_cross_splits_even_if_claimed_independent():
    data=rows();data[1]['source_video_sha256']='0'
    assert not build_subset(data,{frozenset(('0',))},set())['rows']


def test_small_verified_subset_does_not_wait_for_unrelated_unknown_sources():
    data=rows();data.append(dict(data[0],image_sha256='3',source_video_sha256='3'))
    result=build_subset(data,{frozenset(('0','1'))},set())
    assert {r['image_sha256'] for r in result['rows']}=={'0','1'}


def test_perceptual_only_and_unsigned_scene_decisions_fail():
    decision={'classification':'CONFIRMED_INDEPENDENT','member_sha256':['a','b'],
              'reviewer_id':'SYNTHETIC','reviewed_at_utc':'SYNTHETIC',
              'evidence':[{'kind':'PHASH','sha256':'x','path':'synthetic'}]}
    with pytest.raises(ValueError,match='EVIDENCE_REQUIRED'):validate_scene_decision(decision,{'a','b'})
    decision['evidence'][0]['kind']='VISUAL_SCENE_REVIEW';validate_scene_decision(decision,{'a','b'})
    with pytest.raises(ValueError,match='MEMBER_BINDING'):validate_scene_decision(decision,{'a','c'})


def test_baseline_membership_is_train_only_and_exact_hash_bound():
    data=rows();assert training_membership(data[:1])[0]['image_sha256']=='0'
    with pytest.raises(ValueError,match='TRAIN_ONLY'):training_membership(data)
    data[0]['label_sha256']=None
    with pytest.raises(ValueError,match='HASH_AND_PROVENANCE'):training_membership(data[:1])
