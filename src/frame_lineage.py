"""Recover frame identity by actual sequential decode and exact image reproduction."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ground_truth_studio.integrity_service import sha256


def match_derived_image(frame: np.ndarray, image: Path) -> dict[str, Any] | None:
    """Require exact bytes after the documented legacy resize/JPEG95 pipeline.

    No visual similarity threshold or model prediction can establish identity.
    Unrecorded anonymization is not guessed or rerun.
    """
    h,w=frame.shape[:2]
    transform={'resize':None,'jpeg_quality':95,'encoder':'OpenCV '+cv2.__version__,
               'anonymization':'No transform needed to reproduce exact stored bytes'}
    if max(h,w)>1920:
        scale=1920/max(h,w)
        frame=cv2.resize(frame,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_AREA)
        transform['resize']={'width':int(w*scale),'height':int(h*scale),'interpolation':'INTER_AREA'}
    ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,95])
    if ok and hashlib.sha256(encoded.tobytes()).hexdigest()==sha256(image):
        return transform
    return None


def resolve_matches(matches: list[dict]) -> dict:
    """Never choose arbitrarily between different source/frame matches."""
    unique={(r['source_video_sha256'],r['source_frame_index']):r for r in matches}
    if len(unique)>1:return {'status':'AMBIGUOUS_SOURCE','matches':list(unique.values())}
    if not unique:return {'status':'UNVERIFIED_LEGACY','matches':[]}
    return next(iter(unique.values()))


def verify_video_frames(root: Path, source: Path, rows: list[dict], ffprobe: str) -> list[dict]:
    """Verify selected declared indices using sequential decoding from frame zero.

    Args:
        root: Repository owning image/source paths.
        source: Existing candidate source, authenticated against prior manifest.
        rows: Selected legacy image records, never used as GT automatically.
        ffprobe: Local FFprobe executable; retrieves integer PTS/time base.

    Returns:
        One lineage receipt per image. Missing evidence remains explicitly unverified.
    """
    base=[{'image_sha256':r['image_sha256'],'image_path':r['image_path'],
           'source_video_path':source.relative_to(root).as_posix(),
           'source_frame_index':int(r['source_frame_index_declared']),
           'source_pts_ticks':None,'time_base':None,'status':'UNVERIFIED_LEGACY',
           'search_scope':'Declared candidate video and frame index; no exhaustive source search',
           'matching_algorithm':'Sequential OpenCV decode; exact bytes of documented resize + JPEG95',
           'extractor_code_sha256':sha256(root/'scripts/process_new_vn_batches.py'),
           'verifier_code_sha256':sha256(Path(__file__))} for r in rows]
    if not source.is_file():
        return [dict(r,status='SOURCE_NOT_FOUND') for r in base]
    digest=sha256(source)
    if any(r['current_candidate_source_sha256']!=digest for r in rows):
        return [dict(r,status='SOURCE_HASH_MISMATCH') for r in base]
    for receipt in base:receipt['source_video_sha256']=digest
    command=[ffprobe,'-v','error','-select_streams','v:0','-show_frames','-show_streams',
             '-show_entries','frame=pts:stream=time_base','-of','json',str(source)]
    try:
        proc=subprocess.run(command,capture_output=True,timeout=240,check=True)
        metadata=json.loads(proc.stdout);frames=metadata.get('frames',[])
        time_base=metadata['streams'][0]['time_base']
        probe_sha=hashlib.sha256(proc.stdout).hexdigest()
    except (OSError,subprocess.SubprocessError,KeyError,ValueError,IndexError):
        return [dict(r,status='UNVERIFIED_LEGACY',reason='SOURCE_PTS_PROBE_FAILED') for r in base]
    indices={r['source_frame_index']:[] for r in base}
    for r in base:indices[r['source_frame_index']].append(r)
    capture=cv2.VideoCapture(str(source))
    try:
        for index in range(max(indices)+1):
            if not capture.grab():break
            if index not in indices:continue
            ok,frame=capture.retrieve()
            for receipt in indices[index]:
                if not ok:continue
                image=root/receipt['image_path']
                if not image.is_file():receipt.update(status='IMAGE_NOT_FOUND');continue
                if sha256(image)!=receipt['image_sha256']:receipt.update(status='IMAGE_HASH_MISMATCH');continue
                transform=match_derived_image(frame,image)
                pts=frames[index].get('pts') if index<len(frames) else None
                receipt.update(source_pts_ticks=pts,time_base=time_base,ffprobe_output_sha256=probe_sha)
                if transform is not None and isinstance(pts,int):
                    receipt.update(status='VERIFIED_DERIVED_IMAGE',transformation=transform,
                        match_evidence='EXACT_ENCODED_IMAGE_SHA256',confidence='DETERMINISTIC_EXACT_REPRODUCTION',
                        decoded_pixel_sha256=hashlib.sha256(frame.tobytes()).hexdigest())
                else:
                    receipt['reason']='IMAGE_NOT_EXACTLY_REPRODUCIBLE' if transform is None else 'INTEGER_SOURCE_PTS_MISSING'
    finally:
        capture.release()
    if sha256(source)!=digest:
        return [dict(r,status='SOURCE_HASH_MISMATCH',reason='SOURCE_CHANGED_DURING_DECODE') for r in base]
    return base
