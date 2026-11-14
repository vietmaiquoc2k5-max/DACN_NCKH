"""Reproducible synthetic diagnostic for the bounded shared-detector lock.

This measures lock contention only. It does not model decoding, YOLO compute,
camera transport, or fire-event quality and must not be reported as RTSP FPS.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import defaultdict


def run(camera_count: int, duration: float, request_fps: float, service_ms: float) -> dict:
    lock = threading.Lock()
    stop = threading.Event()
    records = defaultdict(lambda: {"attempts": 0, "completed": 0, "busy": 0, "ticks": [], "ages_ms": []})

    def worker(camera_id: str) -> None:
        period = 1.0 / request_fps
        next_request = time.monotonic()
        while not stop.is_set():
            delay = next_request - time.monotonic()
            if delay > 0 and stop.wait(delay):
                break
            requested_at = time.monotonic()
            next_request += period
            row = records[camera_id]
            row["attempts"] += 1
            acquired = lock.acquire(timeout=0.004)
            if not acquired:
                row["busy"] += 1
                continue
            try:
                if stop.wait(service_ms / 1000.0):
                    break
                completed_at = time.monotonic()
                row["completed"] += 1
                row["ticks"].append(completed_at)
                row["ages_ms"].append((completed_at - requested_at) * 1000.0)
            finally:
                lock.release()

    threads = [threading.Thread(target=worker, args=(f"CAM-{index + 1:02d}",), daemon=True)
               for index in range(camera_count)]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    try:
        stop.wait(duration)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=2.0)
    actual_duration = time.monotonic() - started
    summary = {}
    for camera_id, row in sorted(records.items()):
        ticks = row["ticks"]
        gaps = [right - left for left, right in zip(ticks, ticks[1:])]
        attempts = row["attempts"]
        summary[camera_id] = {
            "attempts": attempts,
            "completed": row["completed"],
            "busy": row["busy"],
            "busy_ratio": round(row["busy"] / attempts, 4) if attempts else 0.0,
            "inference_fps": round(row["completed"] / actual_duration, 3),
            "max_success_gap_seconds": round(max(gaps), 4) if gaps else None,
            "max_request_age_ms": round(max(row["ages_ms"]), 3) if row["ages_ms"] else None,
        }
    return {
        "workload": {
            "kind": "synthetic shared-lock diagnostic; not a camera/model benchmark",
            "camera_count": camera_count,
            "requested_fps_per_camera": request_fps,
            "synthetic_service_ms": service_ms,
            "requested_duration_seconds": duration,
            "actual_duration_seconds": round(actual_duration, 4),
            "nonblocking_lock_timeout_ms": 4,
        },
        "per_camera": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", type=int, default=9)
    parser.add_argument("--duration-seconds", type=float, default=8.0)
    parser.add_argument("--request-fps", type=float, default=20.0)
    parser.add_argument("--service-ms", type=float, default=20.0)
    args = parser.parse_args()
    if args.cameras < 1 or args.duration_seconds <= 0 or args.request_fps <= 0 or args.service_ms < 0:
        parser.error("camera count/FPS/duration must be positive and service time cannot be negative")
    print(json.dumps(run(args.cameras, args.duration_seconds, args.request_fps, args.service_ms), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
