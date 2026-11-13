"""Run a bounded four-stream MediaMTX/FFmpeg CameraManager measurement."""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from src.camera_manager import MultiCameraManager  # noqa: E402

LOG = logging.getLogger("week7_rtsp_metrics")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def stop_process(process: subprocess.Popen[Any] | None, timeout: float = 5.0) -> bool:
    if process is None:
        return True
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                return False
    return process.poll() is not None


def wait_port(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def wait_readable(url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            if cap.isOpened() and cap.read()[0]:
                return True
        finally:
            cap.release()
        time.sleep(0.25)
    return False


def run(duration: float, output_root: Path) -> tuple[int, Path]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    artifact_dir = output_root / f"rtsp_metrics_{run_id}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    reasons: list[str] = []
    publishers: list[subprocess.Popen[Any]] = []
    mediamtx: subprocess.Popen[Any] | None = None
    manager: MultiCameraManager | None = None
    temp = tempfile.TemporaryDirectory(prefix="firewatch_rtsp_")
    started = utc_now()
    measurement_start: float | None = None
    camera_rows: list[dict[str, Any]] = []
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
    ffmpeg = shutil.which("ffmpeg")
    server = ROOT / ".tools" / "mediamtx" / "mediamtx.exe"
    config = ROOT / ".tools" / "mediamtx" / "mediamtx.yml"
    video = ROOT / "videovietnam" / "VN_POS_01.mp4"
    try:
        for path, label in ((server, "MediaMTX executable"), (config, "MediaMTX configuration"), (video, "test video")):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")
        if not ffmpeg:
            raise FileNotFoundError("ffmpeg is not available on PATH")
        for proc in ("ffprobe",):
            if not shutil.which(proc):
                raise FileNotFoundError(f"{proc} is not available on PATH")
        logs = artifact_dir / "process_logs"
        logs.mkdir()
        mediamtx = subprocess.Popen([str(server), str(config)], cwd=ROOT, stdout=(logs / "mediamtx.stdout.log").open("wb"), stderr=(logs / "mediamtx.stderr.log").open("wb"), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if not wait_port("127.0.0.1", 8554, 10):
            raise RuntimeError("MediaMTX did not bind loopback port 8554")
        urls = [f"rtsp://127.0.0.1:8554/cam{i}" for i in range(1, 5)]
        for index, url in enumerate(urls, 1):
            log = (logs / f"publisher_{index}.log").open("wb")
            publishers.append(subprocess.Popen([
                ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "warning", "-re", "-stream_loop", "-1", "-i", str(video),
                "-c", "copy", "-f", "rtsp", "-rtsp_transport", "tcp", url,
            ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))
        for url in urls:
            if not wait_readable(url, 20):
                raise RuntimeError(f"RTSP endpoint was not readable: {url}")

        os.environ["FIREWATCH_RTSP_HOSTS"] = "127.0.0.1"
        db_path = Path(temp.name) / "metrics.db"
        manager = MultiCameraManager(db_path=str(db_path), model_path=str(ROOT / "best.pt"), auto_assign_videos=False)
        model_info = manager.detector.get_model_info()
        if not model_info.get("loaded"):
            raise RuntimeError(f"Detector checkpoint failed to load: {model_info.get('error')}")
        camera_ids = []
        for index, url in enumerate(urls, 1):
            camera_id = manager.add_camera(f"SIMULATED_RTSP_{index}", url, source_type="IP_CAMERA", source_reference=url, auto_start=False, inference_fps=8)
            camera_ids.append(camera_id)
            manager.start_camera(camera_id)

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            snapshots = {item["cam_id"]: item for item in manager.get_all_snapshots()}
            if all(snapshots[cid]["processed_frames"] > 0 for cid in camera_ids):
                break
            time.sleep(0.25)
        initial = {cid: manager.workers[cid].get_snapshot() for cid in camera_ids}
        measurement_start = time.monotonic()
        wall_start = utc_now()
        time.sleep(duration)
        measurement_end = time.monotonic()
        wall_end = utc_now()
        elapsed = measurement_end - measurement_start
        final = {cid: manager.workers[cid].get_snapshot() for cid in camera_ids}
        for cid in camera_ids:
            before, after = initial[cid], final[cid]
            decoded = max(0, int(after["captured_frames"]) - int(before["captured_frames"]))
            inferred = max(0, int(after["processed_frames"]) - int(before["processed_frames"]))
            camera_rows.append({
                "camera_id": cid, "source_id": f"SIMULATED_RTSP_{camera_ids.index(cid)+1}",
                "endpoint": urls[camera_ids.index(cid)], "measurement_start": wall_start,
                "measurement_end": wall_end, "duration_seconds": elapsed,
                "decoded_frames": decoded, "inference_frames": inferred,
                "capture_fps": decoded / elapsed if elapsed > 0 else None,
                "inference_fps": inferred / elapsed if elapsed > 0 else None,
                "reported_capture_fps": after.get("capture_fps"), "reported_inference_fps": after.get("inference_fps"),
                "inference_latency_ms": after.get("infer_ms"), "end_to_end_latency_ms": after.get("end_to_end_latency_ms"),
                "dropped_frames_counter": after.get("dropped_frames"), "reconnect_attempts": after.get("reconnect_attempts"),
                "final_state": after.get("status"),
            })
        success = all(row["decoded_frames"] > 0 and row["inference_frames"] > 0 for row in camera_rows)
        status = "PASS_METRICS" if success else "FAIL_METRICS"
        exit_code = 0 if success else 1
        (artifact_dir / "camera_metrics.json").write_text(json.dumps({
            "run_id": run_id, "commit": commit, "status": status, "failure_reasons": reasons,
            "started_at": started, "measurement_start": wall_start, "measurement_end": wall_end,
            "measurement_duration_seconds": elapsed,
            "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "opencv": cv2.__version__, "mediaMTX": "v1.21.3", "ffmpeg": subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, timeout=5).stdout.splitlines()[0]},
            "model": model_info, "measurement_method": "counter deltas / time.monotonic; camera workers actual captured and inferred frame counters",
            "cameras": camera_rows,
        }, indent=2), encoding="utf-8")
        return exit_code, artifact_dir
    except Exception as exc:
        reasons.append(f"{type(exc).__name__}: {exc}")
        (artifact_dir / "camera_metrics.json").write_text(json.dumps({"run_id": run_id, "commit": commit, "status": "BLOCKED", "failure_reasons": reasons, "cameras": camera_rows}, indent=2), encoding="utf-8")
        raise
    finally:
        if manager is not None:
            manager.close()
        for process in publishers:
            stop_process(process)
        stop_process(mediamtx)
        temp.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=15.0)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "benchmarks" / "week7")
    args = parser.parse_args()
    if args.duration_seconds <= 0 or args.duration_seconds > 3600:
        parser.error("duration must be greater than zero and at most one hour")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    code, path = run(args.duration_seconds, args.output_root)
    print(f"status={'PASS_METRICS' if code == 0 else 'FAIL_METRICS'} artifact={path / 'camera_metrics.json'} exit_code={code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
