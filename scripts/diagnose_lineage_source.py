"""Isolate decoder stderr and native exit per source, without altering source bytes."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ground_truth_studio.config import Settings, confined
from ground_truth_studio.integrity_service import sha256


def diagnose(source: Path, output: Path, ffmpeg: str) -> dict:
    """Decode one suspected source and preserve independent process exit evidence."""
    output.mkdir(parents=True,exist_ok=False)
    before=sha256(source)
    command=[ffmpeg,'-v','error','-threads','1','-i',str(source),'-map','0:v:0','-f','null','-']
    try:
        with (output/'stdout.log').open('wb') as stdout, (output/'stderr.log').open('wb') as stderr:
            process=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=600)
        code=process.returncode;timed_out=False
    except subprocess.TimeoutExpired:
        code=None;timed_out=True
    error=(output/'stderr.log').read_text(encoding='utf-8',errors='replace')
    tokens=('Invalid NAL unit size','Error splitting','missing picture','corrupt','Invalid data')
    result={'source':str(source),'source_sha256':before,'source_unchanged':sha256(source)==before,
            'command':command,'native_decoder_exit_code':code,'timed_out':timed_out,
            'stderr_sha256':sha256(output/'stderr.log'),
            'decode_integrity':'ERROR' if timed_out or code!=0 or any(t in error for t in tokens) else 'NO_REPORTED_ERROR',
            'error_excerpt':error[:4000],'python_exit_code':0,
            'note':'Python diagnostic succeeds when it records evidence; decoder integrity is a separate gate'}
    (output/'receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();ffmpeg=Settings.load().ffmpeg
    if not ffmpeg:raise ValueError('FFMPEG_REQUIRED')
    print(json.dumps(diagnose(confined(ROOT,args.source),args.output,ffmpeg)))
