"""A successful Python wrapper must not hide native decoder corruption."""
import subprocess
from pathlib import Path
from unittest.mock import patch

from scripts.diagnose_lineage_source import diagnose


def test_decoder_stderr_is_bound_to_source_even_with_zero_exit(tmp_path: Path):
    source=tmp_path/'synthetic.mp4';source.write_bytes(b'synthetic source')
    def run(command,stdout,stderr,timeout):
        stderr.write(b'Invalid NAL unit size (7937 > 7149).')
        return subprocess.CompletedProcess(command,0)
    with patch('scripts.diagnose_lineage_source.subprocess.run',side_effect=run):
        receipt=diagnose(source,tmp_path/'out','synthetic_ffmpeg')
    assert receipt['decode_integrity']=='ERROR'
    assert receipt['native_decoder_exit_code']==0
    assert receipt['python_exit_code']==0
    assert receipt['source_unchanged']
    assert receipt['source']==str(source)
