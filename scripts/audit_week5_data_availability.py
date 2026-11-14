"""Discover D-Fire in the workspace and configured roots without mutating data."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from ground_truth_studio.integrity_service import sha256


def discover() -> dict[str,Any]:
    """Inspect configured paths and bound workspace discovery to two minutes.

    Returns:
        Actual locations/counts and discovery limits. Finding a directory is not
        a content/annotation/split readiness approval.
    """
    from ultralytics.utils import SETTINGS
    paths=sorted(set(ROOT.glob('*.yaml'))|set((ROOT/'configs').rglob('*.yaml'))|
                 set((ROOT/'runs/detect').rglob('args.yaml')))
    configs=[];roots=set()
    for path in paths:
        value=yaml.safe_load(path.read_text(encoding='utf-8'))
        if not isinstance(value,dict):continue
        configs.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path),
            'dataset_path':value.get('path'),'training_data':value.get('data'),
            'class_mapping':value.get('names')})
        if re.search(r'd[-_ ]?fire',path.name,re.I):
            declared=Path(value.get('path','.'))
            roots.update((base/declared).resolve() for base in [ROOT,path.parent])
    configured=Path(SETTINGS.get('datasets_dir',ROOT/'datasets'))
    for base in [ROOT/'datasets',ROOT,ROOT.parent,configured]:
        roots.update([base/'dfire',base/'D-Fire'])
    matching=[];bbox=[];errors=[];visited=0;deadline=time.monotonic()+120
    skipped={'.git','node_modules','__pycache__','.venv','venv','site-packages',
             'playback_cache','proxy_cache','.cache','cache','Cache','Code Cache','GPUCache'}
    for directory,subdirs,files in os.walk(ROOT.parent,onerror=lambda e:errors.append(type(e).__name__)):
        visited+=1
        subdirs[:]=[s for s in subdirs if s not in skipped and not s.startswith(
            ('chrome_profile','studio-chrome','recovery-chrome','chrome-qa'))]
        for name in subdirs+files:
            candidate=Path(directory)/name
            if re.search(r'd[-_ ]?fire',name,re.I):
                matching.append(str(candidate))
                if candidate.is_dir() and re.match(r'd[-_ ]?fire',name,re.I):roots.add(candidate)
            if re.search(r'(bbox|cvat|labelstudio|annotations?)',name,re.I) and Path(name).suffix.lower() in {'.csv','.json','.jsonl','.xml','.zip'}:
                bbox.append(str(candidate))
        if time.monotonic()>deadline:
            errors.append('DISCOVERY_DEADLINE_REACHED');break
    locations=[]
    for path in sorted(roots):
        counts={split:{kind:sum(1 for p in (path/split/kind).rglob('*') if p.is_file())
                       if (path/split/kind).is_dir() else 0 for kind in ['images','labels']}
                for split in ['train','val','test']}
        locations.append({'path':str(path),'exists':path.is_dir(),'counts':counts})
    found=any(any(c['images'] for c in entry['counts'].values()) for entry in locations)
    return {'configs':configs,'dfire_locations':locations,'matching_dfire_names':matching,
            'bbox_named_candidates':bbox,'workspace_directories_visited':visited,'errors':errors,
            'scope':'workspace plus configured dataset roots; cache/venv/browser profiles pruned',
            'dfire_status':'FOUND_REQUIRES_CONTENT_AUDIT' if found else 'BLOCKED_DATASET_MISSING',
            'original_datasets_modified':False}


def main() -> None:
    """Write a fresh receipt inside the repository; never replace a prior audit."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();path=args.output.resolve()
    if not path.is_relative_to(ROOT/'runs/benchmarks/week5_domain_adaptation'):
        raise ValueError('Unsafe output receipt path')
    if path.exists():raise FileExistsError('Choose a new discovery receipt')
    result=discover();path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(result['dfire_status'],'DIRECTORIES',result['workspace_directories_visited'],
          'ERRORS',len(result['errors']),flush=True)


if __name__=='__main__':main()
