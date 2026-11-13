"""Bounded local MediaMTX security and RTSP decode verification."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def atomic_report(path: Path, data: dict) -> None:
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


def stop_owned(proc: subprocess.Popen, timeout: float = 3) -> dict:
    row = {"pid": proc.pid, "terminated": False, "exit_code": None, "error": None}
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=timeout)
        row.update(terminated=True, exit_code=proc.poll())
    except Exception as exc:  # cleanup evidence must not hide the report
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def listening_sockets() -> list[dict]:
    result = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True, timeout=5)
    rows = []
    for line in result.stdout.splitlines():
        m = re.match(r"\s*TCP\s+(\S+)\s+(\S+)\s+LISTENING\s+(\d+)", line, re.I)
        if m:
            addr, _, pid = m.groups()
            host, _, port = addr.rpartition(":")
            if port.isdigit() and int(port) in {8554, 1935, 8888, 8889, 8189, 8890, 8892, 8893}:
                rows.append({"address": host.strip("[]"), "port": int(port), "pid": int(pid)})
    return rows


def run(output_root: Path, startup_timeout: float = 10, reader_seconds: float = 3) -> tuple[int, Path]:
    rid = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    folder = output_root / f"security_runtime_{rid}"
    folder.mkdir(parents=True, exist_ok=False)
    report_path = folder / "security_runtime_report.json"
    cfg = ROOT / ".tools/mediamtx/mediamtx.local-benchmark.yml"
    server = ROOT / ".tools/mediamtx/mediamtx.exe"
    video = ROOT / "videovietnam/VN_POS_01.mp4"
    ffmpeg = shutil.which("ffmpeg")
    report = {"run_id": rid, "git_head": "unknown", "effective_config": str(cfg), "phases": {}, "failure_reasons": [], "commands": []}
    procs: list[tuple[str, subprocess.Popen]] = []
    atomic_report(report_path, report)
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        report["git_head"] = head.stdout.strip()
        checks = {"mediamtx": server.is_file(), "ffmpeg": bool(ffmpeg), "config": cfg.is_file(), "video": video.is_file()}
        with socket.socket() as sock:
            checks["port_free"] = sock.connect_ex(("127.0.0.1", 8554)) != 0
        report["phases"]["preflight"] = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
        atomic_report(report_path, report)
        if not all(checks.values()):
            raise RuntimeError(f"Preflight failed: {checks}")
        ver = subprocess.run([str(server), "--version"], capture_output=True, text=True, timeout=5)
        report["mediamtx_version"] = (ver.stdout + ver.stderr).strip()
        logs = (folder / "mediamtx_startup.log").open("w", encoding="utf-8")
        cmd = [str(server), str(cfg)]
        report["commands"].append({"argv": cmd, "exit_code": None})
        mtx = subprocess.Popen(cmd, cwd=ROOT, stdout=logs, stderr=subprocess.STDOUT)
        procs.append(("mediamtx", mtx))
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline and mtx.poll() is None:
            if any(x["port"] == 8554 and x["pid"] == mtx.pid for x in listening_sockets()):
                break
            time.sleep(.2)
        sockets = listening_sockets()
        (folder / "listener_snapshot.json").write_text(json.dumps(sockets, indent=2), encoding="utf-8")
        report["phases"]["startup"] = {"status": "PASS" if mtx.poll() is None and any(x["port"] == 8554 and x["pid"] == mtx.pid and x["address"] == "127.0.0.1" for x in sockets) else "FAIL", "pid": mtx.pid, "sockets": sockets}
        atomic_report(report_path, report)
        if report["phases"]["startup"]["status"] != "PASS":
            raise RuntimeError("MediaMTX failed to start expected owned loopback RTSP socket")
        publog = (folder / "publisher.log").open("w", encoding="utf-8")
        pubcmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "warning", "-re", "-stream_loop", "-1", "-i", str(video), "-c", "copy", "-f", "rtsp", "-rtsp_transport", "tcp", "rtsp://127.0.0.1:8554/cam1"]
        pub = subprocess.Popen(pubcmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=publog)
        procs.append(("publisher", pub))
        # Wait until the publisher has remained alive and MediaMTX logs the
        # path as available before opening the decoding reader.
        deadline = time.monotonic() + min(15, startup_timeout + 5)
        startup_log = folder / "mediamtx_startup.log"
        while time.monotonic() < deadline and pub.poll() is None:
            if startup_log.exists() and "stream is available and online" in startup_log.read_text(encoding="utf-8", errors="replace"):
                break
            time.sleep(.2)
        report["phases"]["publisher_ready"] = {"status": "PASS" if pub.poll() is None and startup_log.exists() and "stream is available and online" in startup_log.read_text(encoding="utf-8", errors="replace") else "FAIL", "pid": pub.pid, "running": pub.poll() is None}
        atomic_report(report_path, report)
        if report["phases"]["publisher_ready"]["status"] != "PASS":
            raise RuntimeError("Publisher exited early or stream readiness timed out")
        reader_cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "info", "-rtsp_transport", "tcp", "-i", "rtsp://127.0.0.1:8554/cam1", "-map", "0:v:0", "-an", "-t", str(reader_seconds), "-vf", "showinfo", "-f", "null", "NUL"]
        rr = subprocess.run(reader_cmd, cwd=ROOT, capture_output=True, text=True, timeout=reader_seconds + 10)
        reader_text = rr.stderr or ""
        (folder / "rtsp_reader.log").write_text(reader_text, encoding="utf-8")
        # showinfo emits one line per decoded frame, independent of FFmpeg's
        # terminal progress rendering (which can report frame=0 on Windows).
        frames = len(re.findall(r"showinfo_?\d*.*?\bn:\s*\d+", reader_text, flags=re.IGNORECASE))
        codec = re.search(r"Video: ([^,\r\n]+)", reader_text)
        report["phases"]["rtsp"] = {"status": "PASS" if rr.returncode == 0 and frames > 0 else "FAIL", "publisher_pid": pub.pid, "publisher_running": pub.poll() is None, "reader_argv": reader_cmd, "reader_exit_code": rr.returncode, "decoded_frames": frames, "codec": codec.group(1) if codec else None, "warnings": [x for x in reader_text.splitlines() if "warn" in x.lower()]}
        atomic_report(report_path, report)
    except Exception as exc:
        report["failure_reasons"].append(f"{type(exc).__name__}: {exc}")
        report["phases"].setdefault("runtime", {"status": "FAIL"})
        atomic_report(report_path, report)
    finally:
        cleanup = []
        try:
            for name, proc in reversed(procs):
                row = stop_owned(proc)
                row["name"] = name
                cleanup.append(row)
            time.sleep(.3)
            cleanup.append({"port_8554_closed": not any(x["port"] == 8554 for x in listening_sockets())})
        except Exception as exc:
            cleanup.append({"error": f"{type(exc).__name__}: {exc}"})
        report["cleanup"] = cleanup
        report["phases"]["cleanup"] = {"status": "PASS" if cleanup and all(x.get("terminated", True) for x in cleanup) and cleanup[-1].get("port_8554_closed") else "FAIL"}
        atomic_report(report_path, report)
    success = all(report.get("phases", {}).get(k, {}).get("status") == "PASS" for k in ("preflight", "startup", "publisher_ready", "rtsp", "cleanup"))
    return (0 if success else 1), report_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/benchmarks/week7")
    parser.add_argument("--startup-timeout", type=float, default=10)
    parser.add_argument("--reader-seconds", type=float, default=3)
    args = parser.parse_args()
    code, report = run(args.output_root, args.startup_timeout, args.reader_seconds)
    # Keep the saved path Unicode; escape only the console representation when
    # the active Windows code page cannot encode the workspace path.
    message = f"exit_code={code} report={report.as_posix()}"
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        message.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        message = message.encode(encoding, errors="backslashreplace").decode(encoding, errors="strict")
    print(message)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

