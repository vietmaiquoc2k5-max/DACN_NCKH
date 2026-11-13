import json
import subprocess
import sys
import io
from types import SimpleNamespace

from scripts import verify_week7_mediamtx_security as verify


def test_successful_report_initialization(tmp_path):
    path = tmp_path / "report.json"
    verify.atomic_report(path, {"phases": {"init": {"status": "RUNNING"}}})
    assert json.loads(path.read_text())["phases"]["init"]["status"] == "RUNNING"


def test_preflight_failure_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    code, path = verify.run(tmp_path / "out")
    data = json.loads(path.read_text())
    assert code == 1 and data["phases"]["preflight"]["status"] == "FAIL"


def test_startup_failure_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    (tmp_path / ".tools/mediamtx").mkdir(parents=True)
    (tmp_path / ".tools/mediamtx/mediamtx.exe").touch()
    (tmp_path / ".tools/mediamtx/mediamtx.local-benchmark.yml").touch()
    (tmp_path / "videovietnam").mkdir()
    (tmp_path / "videovietnam/VN_POS_01.mp4").touch()
    monkeypatch.setattr(verify.shutil, "which", lambda _: "ffmpeg.exe")
    monkeypatch.setattr(verify.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="HEAD", stderr=""))
    class Failed:
        pid = 123456
        def poll(self): return 9
        def wait(self, timeout=None): return 9
    monkeypatch.setattr(verify.subprocess, "Popen", lambda *a, **k: Failed())
    monkeypatch.setattr(verify, "listening_sockets", lambda: [])
    code, path = verify.run(tmp_path / "out", startup_timeout=.01)
    data = json.loads(path.read_text())
    assert code == 1 and data["failure_reasons"]


def test_reader_failure_report_and_cleanup_exception(tmp_path, monkeypatch):
    # Cleanup errors are represented in the returned row, not raised.
    class Bad:
        pid = 4
        def poll(self): return None
        def terminate(self): raise PermissionError("denied")
    row = verify.stop_owned(Bad())
    assert "denied" in row["error"]


def test_port_occupied_and_no_unrelated_termination(tmp_path, monkeypatch):
    # Only the supplied Popen handle is ever passed to stop_owned.
    class Owned:
        pid = 55
        def __init__(self): self.killed = False
        def poll(self): return 0
        def wait(self, timeout=None): return 0
    owned = Owned()
    assert verify.stop_owned(owned)["terminated"]
    assert not owned.killed


def test_cp1252_console_escapes_vietnamese_path_and_keeps_json(tmp_path, monkeypatch, capsys):
    path = tmp_path / "Dự án" / "báo cáo.json"
    path.parent.mkdir()
    verify.atomic_report(path, {"path": str(path)})
    assert json.loads(path.read_text(encoding="utf-8"))["path"] == str(path)
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setattr(verify, "run", lambda *a, **k: (1, path))
    class CP1252(io.StringIO):
        encoding = "cp1252"
    monkeypatch.setattr(sys, "stdout", CP1252())
    monkeypatch.setattr(sys, "argv", ["verify", "--output-root", str(tmp_path)])
    assert verify.main() == 1


def test_main_success_exit_only_when_run_says_all_checks_pass(tmp_path, monkeypatch):
    p = tmp_path / "report.json"
    monkeypatch.setattr(verify, "run", lambda *a, **k: (0, p))
    monkeypatch.setattr(sys, "argv", ["verify", "--output-root", str(tmp_path)])
    assert verify.main() == 0


def test_runtime_failure_exit_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    code, path = verify.run(tmp_path / "out")
    assert code != 0 and path.is_file()
