"""Run a measured multi-camera stability session (minimum two hours).

Sources must be supplied as safe camera configuration references, normally
``env:CAMERA_RTSP_CAM01``. The corresponding environment variables hold the
credentials and ``FIREWATCH_RTSP_HOSTS`` allowlists each hostname.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.camera_manager import MultiCameraManager  # noqa: E402

LOG = logging.getLogger("week7_stability")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _resource_usage(cuda_available: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "gpu_utilization_percent": None,
        "gpu_utilization_source": "unavailable",
        "vram_allocated_bytes": None,
        "vram_reserved_bytes": None,
    }
    try:
        import psutil  # type: ignore[import-not-found]

        process = psutil.Process()
        result.update({
            "cpu_percent": psutil.cpu_percent(interval=None),
            "process_rss_bytes": process.memory_info().rss,
            "system_ram_percent": psutil.virtual_memory().percent,
        })
    except ImportError:
        result["resource_metrics"] = "psutil unavailable"
    if cuda_available:
        try:
            import torch

            result["vram_allocated_bytes"] = int(torch.cuda.memory_allocated())
            result["vram_reserved_bytes"] = int(torch.cuda.memory_reserved())
            result["gpu_utilization_source"] = "torch.cuda memory only; utilization unavailable"
        except (ImportError, RuntimeError):
            result["gpu_utilization_source"] = "CUDA runtime metrics unavailable"
    return result


def run(duration_seconds: int, sources: list[str], output_dir: Path) -> int:
    if duration_seconds < 7200:
        raise ValueError("duration_seconds must be at least 7200 (two hours)")
    if len(sources) < 4:
        raise ValueError("At least four camera sources are required")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = Path(os.getenv("FIREWATCH_MODEL_PATH", str(ROOT / "best.pt"))).expanduser()
    manager = MultiCameraManager(
        db_path=os.getenv("FIREWATCH_DB_PATH", str(output_dir / "alerts.db")),
        model_path=str(model),
        auto_assign_videos=False,
    )
    model_info = manager.detector.get_model_info()
    if not model.is_file() or not model_info.get("loaded"):
        manager.close()
        raise ValueError(f"A valid selected checkpoint is required: {model_info.get('error') or model}")
    cameras: list[str] = []
    try:
        for index, source_ref in enumerate(sources, start=1):
            cameras.append(manager.add_camera(
                name=f"Week 7 RTSP {index}", source=source_ref,
                source_type="IP_CAMERA", source_reference=source_ref,
                enabled=True, auto_start=False, inference_fps=60,
            ))
    except Exception:
        manager.close()
        raise

    start_wall = _utc_now()
    start = time.monotonic()
    checkpoint_hash = _sha256(model)
    environment = {
        "start_utc": start_wall,
        "requested_duration_seconds": duration_seconds,
        "camera_count": len(cameras),
        "source_types": ["RTSP reference"] * len(cameras),
        "source_references": sources,
        "model_path": str(model),
        "model_sha256": checkpoint_hash,
        "git_commit": _commit_sha(),
        "platform": platform.platform(),
        "python": sys.version,
        "hardware": platform.processor(),
        "device": model_info.get("device"),
        "cuda_available": str(model_info.get("device", "cpu")).startswith("cuda"),
    }
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    metrics_path = output_dir / "camera_metrics.jsonl"
    alerts_path = output_dir / "alert_latency.jsonl"
    exit_code = 0
    try:
        for camera_id in cameras:
            manager.start_camera(camera_id)
        with metrics_path.open("a", encoding="utf-8") as metrics, alerts_path.open("a", encoding="utf-8") as alerts:
            last_event_id: set[int] = set()
            last_delivery_state: dict[str, tuple[Any, ...]] = {}
            while time.monotonic() - start < duration_seconds:
                sample_time = _utc_now()
                for snap in manager.get_all_snapshots():
                    if snap["cam_id"] not in cameras:
                        continue
                    metrics.write(json.dumps({"timestamp_utc": sample_time, **{
                        key: snap.get(key) for key in (
                            "cam_id", "status", "capture_fps", "inference_fps", "display_fps",
                            "infer_ms", "end_to_end_latency_ms", "dropped_frames", "queue_depth",
                            "reconnect_attempts", "captured_frames", "processed_frames",
                            "inference_busy_ratio", "last_successful_inference_age_seconds",
                            "max_inference_gap_seconds", "frame_age_at_inference_ms",
                        )
                    }, **_resource_usage(bool(environment["cuda_available"]))}) + "\n")
                metrics.flush()
                for event in manager.event_db.query_events(limit=100, include_demo=False):
                    event_id = int(event["id"])
                    if event_id not in last_event_id:
                        last_event_id.add(event_id)
                        alerts.write(json.dumps({"timestamp_utc": sample_time, "event_id": event_id,
                                                 "event_confirmed_at": event.get("event_confirmed_at"),
                                                 "delivery_status": "NO_DELIVERY_ROW_YET"}) + "\n")
                with manager.telegram_alerts._connect() as conn:
                    deliveries = conn.execute("""SELECT delivery_id,event_id,status,event_confirmed_at,
                        queued_at,attempt_started_at,telegram_acknowledged_at,attempt_count,error_category
                        FROM alert_deliveries ORDER BY queued_at""").fetchall()
                for row in deliveries:
                    delivery_id = str(row[0])
                    fingerprint = tuple(row[1:])
                    if last_delivery_state.get(delivery_id) != fingerprint:
                        last_delivery_state[delivery_id] = fingerprint
                        alerts.write(json.dumps({"delivery_id": delivery_id, "event_id": row[1],
                            "status": row[2], "event_confirmed_at": row[3], "alert_queued_at": row[4],
                            "alert_attempt_started_at": row[5], "telegram_acknowledged_at": row[6],
                            "attempt_count": row[7], "error_category": row[8]}) + "\n")
                alerts.flush()
                time.sleep(5)
    except KeyboardInterrupt:
        LOG.warning("Stability run interrupted; recording actual duration")
        exit_code = 130
    finally:
        actual_duration = time.monotonic() - start
        end_wall = _utc_now()
        for camera_id in cameras:
            manager.stop_camera(camera_id)
        manager.close()
        samples = []
        with metrics_path.open(encoding="utf-8") as metrics:
            for line in metrics:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        latencies = sorted(float(row["infer_ms"]) for row in samples if row.get("infer_ms") is not None)
        def percentile(values: list[float], fraction: float) -> float | None:
            if not values:
                return None
            return round(values[min(len(values) - 1, int((len(values) - 1) * fraction))], 3)
        camera_summary = {}
        for camera_id in cameras:
            camera_samples = [row for row in samples if row.get("cam_id") == camera_id]
            camera_summary[camera_id] = {
                "samples": len(camera_samples),
                "capture_fps_mean": round(statistics.fmean([float(r["capture_fps"]) for r in camera_samples]), 3) if camera_samples else 0,
                "inference_fps_mean": round(statistics.fmean([float(r["inference_fps"]) for r in camera_samples]), 3) if camera_samples else 0,
                "inference_latency_p50_ms": percentile(sorted(float(r["infer_ms"]) for r in camera_samples if r.get("infer_ms") is not None), .50),
                "inference_latency_p95_ms": percentile(sorted(float(r["infer_ms"]) for r in camera_samples if r.get("infer_ms") is not None), .95),
                "end_to_end_latency_p50_ms": percentile(sorted(float(r["end_to_end_latency_ms"]) for r in camera_samples if r.get("end_to_end_latency_ms") is not None), .50),
                "end_to_end_latency_p95_ms": percentile(sorted(float(r["end_to_end_latency_ms"]) for r in camera_samples if r.get("end_to_end_latency_ms") is not None), .95),
                "inference_busy_ratio_last_sample": camera_samples[-1].get("inference_busy_ratio") if camera_samples else None,
                "last_successful_inference_age_seconds": camera_samples[-1].get("last_successful_inference_age_seconds") if camera_samples else None,
                "max_inference_gap_seconds": camera_samples[-1].get("max_inference_gap_seconds") if camera_samples else None,
                "frame_age_at_inference_ms_last_sample": camera_samples[-1].get("frame_age_at_inference_ms") if camera_samples else None,
                "dropped_frames_last_sample": camera_samples[-1].get("dropped_frames", 0) if camera_samples else None,
                "camera_failures": sum(1 for row in camera_samples if row.get("status") == "ERROR"),
            }
        with alerts_path.open(encoding="utf-8") as alerts:
            delivery_rows = [json.loads(line) for line in alerts if line.strip()]
        ack_latencies = []
        for row in delivery_rows:
            try:
                if row.get("event_confirmed_at") and row.get("telegram_acknowledged_at"):
                    delta = datetime.fromisoformat(row["telegram_acknowledged_at"]) - datetime.fromisoformat(row["event_confirmed_at"])
                    ack_latencies.append(delta.total_seconds())
            except (TypeError, ValueError):
                continue
        summary = {
            **environment,
            "end_utc": end_wall,
            "actual_duration_seconds": round(actual_duration, 3),
            "completed_required_duration": actual_duration >= 7200,
            "camera_ids": cameras,
            "camera_metrics_summary": camera_summary,
            "inference_latency_p50_ms": percentile(latencies, .50),
            "inference_latency_p95_ms": percentile(latencies, .95),
            "telegram_ack_latency_seconds": ack_latencies,
            "telegram_delivery_status_counts": {
                status: sum(1 for row in delivery_rows if row.get("status") == status)
                for status in sorted({row.get("status") for row in delivery_rows if row.get("status")})
            },
            "result": "COMPLETED" if actual_duration >= 7200 else "INTERRUPTED_OR_INCOMPLETE",
        }
        (output_dir / "stability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=int, default=7200)
    parser.add_argument("--source", action="append", required=True,
                        help="Camera source reference, e.g. env:CAMERA_RTSP_CAM01; repeat at least four times")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs" / "benchmarks" / "week7")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return run(args.duration_seconds, args.source, args.output_dir)
    except (ValueError, OSError) as exc:
        LOG.error("Stability benchmark could not start: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
