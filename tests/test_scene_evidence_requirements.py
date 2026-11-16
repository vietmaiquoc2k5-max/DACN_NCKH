"""Metadata is optional; actual split evidence and locked gates remain required."""
import json
from pathlib import Path
import pytest
from src.scene_subset import validate_scene_decision,build_subset


def test_visual_scene_evidence_schema_does_not_require_gps_url_or_camera_id():
    decision={'classification':'CONFIRMED_INDEPENDENT','member_sha256':['a','b'],
              'reviewer_id':'SYNTHETIC_REVIEWER','reviewed_at_utc':'SYNTHETIC',
              'evidence':[{'kind':'VISUAL_SCENE_REVIEW','path':'synthetic_contact.jpg','sha256':'synthetic'}]}
    validate_scene_decision(decision,{'a','b'})


def test_missing_metadata_is_not_the_selector_gate_but_unknown_relation_is():
    rows=[dict(image_sha256=str(i),source_video_sha256=str(i),split=s,human_approved=True,lineage_verified=True)
          for i,s in enumerate(['train','val'])]
    assert build_subset(rows,{frozenset(('0','1'))},set())['train_count']==1
    assert build_subset(rows,set(),set())['train_count']==0


def test_proposed_gate_revision_is_not_activated():
    path=Path(__file__).resolve().parents[1]/'configs/research/week5_scene_evidence_proposal_v3.json'
    proposal=json.loads(path.read_text())
    assert proposal['status']=='PROPOSED_NOT_ACTIVE' and not proposal['replaces_locked_protocol']
    assert not proposal['gps_required'] and not proposal['original_camera_id_required']
