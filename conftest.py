"""Root pytest configuration ensuring platform-safe path resolution and tempdir handling."""
import os
import sys
import tempfile
from pathlib import Path

# Ensure temp directory stays within workspace to prevent WinError 5 on Windows sandbox
_workspace_tmp = Path(os.path.abspath(__file__)).parent / "temp_pytest"
_workspace_tmp.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(_workspace_tmp)
os.environ["TMP"] = str(_workspace_tmp)
os.environ["YOLO_CONFIG_DIR"] = str(_workspace_tmp)
tempfile.tempdir = str(_workspace_tmp)

# Apply safe path resolve before pytest does filesystem checks
import src.config
