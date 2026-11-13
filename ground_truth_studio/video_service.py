"""Source-frame decoding, explicit timestamp provenance and bounded-memory HTTP media."""
from __future__ import annotations

import bisect
import json
import mimetypes
import math
import secrets
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2

from ground_truth_studio.config import Settings, confined
from ground_truth_studio.integrity_service import sha256


class VideoService:
    """Decode original frames sequentially; never derive VFR source PTS from FPS."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.RLock()
        self._reader: tuple[str, cv2.VideoCapture, int] | None = None
        self._metadata: dict[tuple[str, int, int], dict[str, Any]] = {}

    def inspect(self, video: dict[str, Any]) -> dict[str, Any]:
        """Verify bytes, decode readability, and index ffprobe PTS when available."""
        path = confined(self.settings.repo, video["relative_path"])
        if not path.is_file():
            raise ValueError("BLOCKED_INPUT: source video is missing")
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        with self._lock:
            if key in self._metadata:
                cached = self._metadata[key]
                if video.get("expected_sha256") and cached["source_sha256"] != video["expected_sha256"]:
                    raise ValueError("BLOCKED_INPUT: source SHA-256 mismatch")
                return cached
        digest = sha256(path)
        if video.get("expected_sha256") and digest != video["expected_sha256"]:
            raise ValueError("BLOCKED_INPUT: source SHA-256 mismatch")
        capture = cv2.VideoCapture(str(path))
        try:
            readable, image = capture.read()
            if not capture.isOpened() or not readable or image is None:
                raise ValueError("BLOCKED_INPUT: source codec cannot decode its first frame")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0 or count <= 0:
                raise ValueError("BLOCKED_INPUT: source duration/frame count unavailable")
            size = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        finally:
            capture.release()
        pts: list[float] = []
        duration = count / fps
        probe = self.settings.ffprobe
        if probe:
            command = [probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                       "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path)]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=180)
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if not line.strip():
                            continue
                        value = line.split(",")[0]
                        if value in {"N/A", ""}:
                            pts = []
                            break
                        pts.append(float(value))
                if not pts or not all(math.isfinite(value) for value in pts) or any(b <= a for a, b in zip(pts, pts[1:])):
                    pts = []
                if pts:
                    # Frame count comes from actual frame records, not container padding.
                    count = len(pts)
                    tail = pts[-1] - pts[-2] if len(pts) > 1 else 1 / fps
                    duration = pts[-1] - pts[0] + tail
            except (ValueError, subprocess.TimeoutExpired):
                pts = []
        metadata = {"source_sha256": digest, "frame_count": count, "fps": fps, "duration": duration,
                    "resolution": size, "source_pts": pts, "timestamp_provenance": "SOURCE_PTS" if pts else "ESTIMATED_FRAME_INDEX_OVER_FPS",
                    "duration_provenance": "LAST_SOURCE_PTS_PLUS_LAST_FRAME_DELTA" if pts else "ESTIMATED_FRAME_COUNT_OVER_FPS"}
        with self._lock:
            self._metadata[key] = metadata
        return metadata

    @staticmethod
    def index_at_time(metadata: dict[str, Any], seconds: float) -> int:
        """Select an actual decoded ordinal from source PTS, or label an estimate."""
        if not math.isfinite(seconds):
            raise ValueError("Seek timestamp must be finite")
        pts = metadata["source_pts"]
        index = bisect.bisect_right(pts, seconds + pts[0]) - 1 if pts else int(seconds * metadata["fps"])
        return min(max(index, 0), metadata["frame_count"] - 1)

    def frame(self, video: dict[str, Any], index: int) -> tuple[Any, dict[str, Any]]:
        """Decode the exact ordinal, including VFR, without CAP_PROP_POS_FRAMES seeking."""
        metadata = self.inspect(video)
        if type(index) is not int or not 0 <= index < metadata["frame_count"]:
            raise ValueError("Frame index outside video")
        path = confined(self.settings.repo, video["relative_path"])
        with self._lock:
            if self._reader is None or self._reader[0] != str(path) or self._reader[2] > index:
                self.close()
                self._reader = (str(path), cv2.VideoCapture(str(path)), 0)
            _, capture, next_index = self._reader
            for _ in range(next_index, index):
                if not capture.grab():
                    self.close()
                    raise ValueError("BLOCKED_INPUT: midstream source decode failed")
            ok, image = capture.read()
            if not ok or image is None:
                self.close()
                raise ValueError("BLOCKED_INPUT: selected frame cannot be decoded")
            self._reader = (str(path), capture, index + 1)
        pts = metadata["source_pts"]
        timestamp = pts[index] - pts[0] if pts else index / metadata["fps"]
        return image, {"frame_index": index, "timestamp_sec": timestamp,
                       "source_pts": pts[index] if pts else None,
                       "timestamp_provenance": metadata["timestamp_provenance"], "source_sha256": metadata["source_sha256"]}

    def close(self) -> None:
        """Release the current source reader."""
        with self._lock:
            if self._reader:
                self._reader[1].release()
                self._reader = None

    def playback_path(self, video: dict[str, Any], metadata: dict[str, Any]) -> Path:
        """Prepare a browser-compatible playback proxy; never replace the original."""
        source = confined(self.settings.repo, video["relative_path"])
        # Transcode only on explicit player use; raw MP4 may have browser-unsupported codecs too.
        if source.suffix.lower() == ".mp4":
            return source
        if not self.settings.ffmpeg:
            raise ValueError("FFmpeg required for AVI playback; exact source-frame viewer remains available")
        cache = self.settings.private / "playback_cache"
        cache.mkdir(exist_ok=True)
        target = cache / (metadata["source_sha256"] + ".mp4")
        if not target.exists():
            temporary = cache / (metadata["source_sha256"] + "." + secrets.token_hex(6) + ".mp4")
            result = subprocess.run([self.settings.ffmpeg, "-nostdin", "-v", "error", "-i", str(source),
                                     "-map", "0:v:0", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                                     "-fps_mode", "passthrough", "-movflags", "+faststart", "-an", str(temporary)],
                                    capture_output=True, timeout=600)
            if result.returncode:
                raise ValueError("BLOCKED_INPUT: FFmpeg cannot prepare playback; original remains intact")
            temporary.replace(target)
        return target


class MediaServer:
    """Serve only explicit media capabilities on loopback using HTTP Range and chunks."""

    def __init__(self, repository: Any):
        self.repository = repository
        self.grants: dict[str, tuple[Path, str]] = {}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                grant = owner.grants.get(self.path)
                if not grant:
                    self.send_error(404)
                    return
                path, token = grant
                try:
                    owner.repository.actor(token)
                except PermissionError:
                    self.send_error(403)
                    return
                size = path.stat().st_size
                start, end = 0, size - 1
                value = self.headers.get("Range")
                if value:
                    import re
                    match = re.fullmatch(r"bytes=(\d+)-(\d*)", value)
                    if not match:
                        self.send_error(416)
                        return
                    start = int(match[1])
                    end = min(int(match[2]) if match[2] else end, end)
                if start > end or start >= size:
                    self.send_error(416)
                    return
                self.send_response(206 if value else 200)
                self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(end - start + 1))
                if value:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                try:
                    with path.open("rb") as stream:
                        stream.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            block = stream.read(min(65536, remaining))
                            if not block:
                                break
                            self.wfile.write(block)
                            remaining -= len(block)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_: Any) -> None:
                pass  # Capability URLs must not appear in access logs.

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def grant(self, token: str, path: Path) -> str:
        self.repository.actor(token)
        # Only callers that resolved a catalog source/proxy may grant access.
        resolved = path.resolve()
        roots = [self.repository.settings.repo / "datasets", self.repository.settings.repo / "FireSmokeLabeler/videos",
                 self.repository.settings.private / "playback_cache"]
        if not any(resolved.is_relative_to(root.resolve()) for root in roots):
            raise ValueError("Media path is outside source/cache roots")
        key = "/media/" + secrets.token_urlsafe(32)
        self.grants[key] = (resolved, token)
        return f"http://127.0.0.1:{self.server.server_port}{key}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
