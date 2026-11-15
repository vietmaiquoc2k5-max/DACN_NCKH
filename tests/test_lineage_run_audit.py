"""Run finality and resume selection must not promote incomplete evidence."""
import json
from pathlib import Path

import pytest

from ground_truth_studio.integrity_service import sha256
from src.lineage_run_audit import audit_run


def fixture(tmp_path):
    image=tmp_path/'image.jpg';image.write_bytes(b'synthetic image')
    source=tmp_path/'video.mp4';source.write_bytes(b'synthetic video')
    row={'image_sha256':sha256(image),'image_path':image.name,
         'source_path_candidate':source.name,'current_candidate_source_sha256':sha256(source),
         'source_frame_index_declared':'0'}
    folder=tmp_path/'output';folder.mkdir()
    proof={'image_sha256':row['image_sha256'],'source_frame_index':0,'source_video_path':source.name,
           'source_video_sha256':sha256(source),'status':'VERIFIED_DERIVED_IMAGE','source_pts_ticks':0,
           'time_base':'1/25','transformation':{'jpeg_quality':95},'match_evidence':'EXACT_ENCODED_IMAGE_SHA256'}
    return row,folder,proof


def test_missing_run_is_interrupted_and_resume_selection_is_idempotent(tmp_path):
    row,folder,proof=fixture(tmp_path)
    a=audit_run([row],folder,tmp_path,process_alive=False)
    b=audit_run([row],folder,tmp_path,process_alive=False)
    assert a==b and a['process_status']=='INTERRUPTED' and a['not_audited']==1
    assert a['missing_ids']==[row['image_sha256']]


def test_valid_files_without_observed_exit_are_not_completed(tmp_path):
    row,folder,proof=fixture(tmp_path)
    path=folder/(row['image_sha256']+'.json');path.write_text(json.dumps(proof))
    (folder/'summary.json').write_text(json.dumps({'scope_images':1,'receipts':{path.name:sha256(path)}}))
    assert audit_run([row],folder,tmp_path,process_alive=False)['process_status']=='NOT_VERIFIED'
    assert audit_run([row],folder,tmp_path,process_alive=False,exit_code=0)['process_status']=='COMPLETED'
    assert audit_run([row],folder,tmp_path,process_alive=True,exit_code=0)['process_status']=='RUNNING'


@pytest.mark.parametrize('field,value',[('source_pts_ticks',None),('time_base','0/1'),('source_frame_index',1),('source_video_sha256','bad')])
def test_bad_receipt_never_grants_verified_status(tmp_path,field,value):
    row,folder,proof=fixture(tmp_path);proof[field]=value
    (folder/(row['image_sha256']+'.json')).write_text(json.dumps(proof))
    result=audit_run([row],folder,tmp_path,process_alive=False,exit_code=0)
    assert result['verified']==0 and result['invalid_receipts']


def test_changed_source_and_duplicate_scope_are_rejected(tmp_path):
    row,folder,proof=fixture(tmp_path)
    (folder/(row['image_sha256']+'.json')).write_text(json.dumps(proof))
    (tmp_path/'video.mp4').write_bytes(b'changed')
    assert audit_run([row],folder,tmp_path,process_alive=False)['invalid_receipts']
    with pytest.raises(ValueError,match='DUPLICATE_SCOPE'):audit_run([row,row],folder,tmp_path,process_alive=False)
