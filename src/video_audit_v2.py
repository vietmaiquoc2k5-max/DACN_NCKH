"""Independent full decode and integer source-timeline evidence, preserving v1."""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2

from src.video_audit import compute_streaming_sha256

POLICY = "independent_decode_eof_v2"
LOGGER = logging.getLogger(__name__)


def write_json(path: Path, value: Any) -> None:
    """Publish a derived run artifact atomically."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def run_tool(command: list[str], folder: Path, name: str, timeout: int) -> dict[str, Any]:
    """Run one bounded tool with streamed output, retaining raw stdout/stderr."""
    stdout, stderr = folder / (name + ".stdout"), folder / (name + ".stderr")
    with stdout.open("wb") as out, stderr.open("wb") as err:
        try:
            result = subprocess.run(command, stdout=out, stderr=err, timeout=timeout)
            code, timed_out = result.returncode, False
        except subprocess.TimeoutExpired:
            code, timed_out = None, True
    return {"command": command, "exit_code": code, "timed_out": timed_out,
            "stdout_path": stdout.name, "stderr_path": stderr.name,
            "stderr": stderr.read_text(encoding="utf-8", errors="replace").strip()}


def inspect_timeline(path: Path) -> dict[str, Any]:
    """Check exact integer PTS/durations, packet correspondence and monotonic DTS.

    B-frame packet PTS reordering is permitted. Display-frame PTS must increase;
    each explicit frame duration must end exactly at the next frame PTS. A VFR
    sequence is accepted using its durations rather than a guessed constant FPS.
    """
    frames: list[tuple[int, int]] = []
    packets: list[tuple[int, int]] = []
    errors: list[str] = []
    previous_dts = None
    dts_regressions = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if parts[0] not in {"frame", "packet"}:
            continue
        fields = dict(part.split("=", 1) for part in parts[1:] if "=" in part)
        try:
            pts, duration = int(fields["pts"]), int(fields["duration"])
            if duration <= 0:
                raise ValueError("nonpositive duration")
            if parts[0] == "frame":
                if int(fields["best_effort_timestamp"]) != pts:
                    errors.append("FRAME_PTS_BEST_EFFORT_CONTRADICTION")
                frames.append((pts, duration))
            else:
                dts = int(fields["dts"])
                if previous_dts is not None and dts < previous_dts:
                    dts_regressions += 1
                previous_dts = dts
                if "C" in fields.get("flags", "") or "D" in fields.get("flags", ""):
                    errors.append("CORRUPT_OR_DISCARD_PACKET")
                packets.append((pts, duration))
        except (KeyError, ValueError):
            errors.append("MISSING_OR_INVALID_SOURCE_TIMESTAMP_DURATION")
    discontinuities = sum(current[0] != previous[0] + previous[1] for previous, current in zip(frames, frames[1:]))
    if not frames or not packets:
        errors.append("EMPTY_TIMELINE")
    if discontinuities:
        errors.append("FRAME_TIMELINE_DISCONTINUITY")
    if dts_regressions:
        errors.append("PACKET_DTS_REGRESSION")
    if sorted(packets) != frames:
        errors.append("PACKET_FRAME_TIMELINE_MISMATCH")
    duration_ticks = frames[-1][0] + frames[-1][1] - frames[0][0] if frames else None
    return {"frame_records": len(frames), "packet_records": len(packets),
            "first_frame_pts_ticks": frames[0][0] if frames else None,
            "last_frame_pts_ticks": frames[-1][0] if frames else None,
            "last_frame_duration_ticks": frames[-1][1] if frames else None,
            "duration_ticks": duration_ticks, "duration_sum_ticks": sum(row[1] for row in frames),
            "frame_discontinuities": discontinuities, "dts_regressions": dts_regressions,
            "variable_frame_durations": len({row[1] for row in frames}) > 1,
            "packet_frame_pairs_match": sorted(packets) == frames,
            "errors": sorted(set(errors)), "continuous": not errors,
            "timeline_sha256": compute_streaming_sha256(path)}


def classify(evidence: dict[str, Any]) -> dict[str, Any]:
    """Apply v2 acceptance with no frame-count delta tolerance or source repair."""
    reasons: list[str] = []
    tools = evidence.get("tools", {})
    status = "NEEDS_REVIEW"
    if evidence.get("sha256_verified") is not True:
        reasons.append("SOURCE_SHA256_MISMATCH_OR_MISSING")
    for name in ("counts", "timeline", "ffmpeg"):
        tool = tools.get(name, {})
        if tool.get("timed_out"):
            reasons.append(name.upper() + "_TIMEOUT")
        elif tool.get("exit_code") != 0:
            reasons.append(name.upper() + "_NONZERO_OR_MISSING_EXIT")
        if tool.get("stderr"):
            reasons.append(name.upper() + "_DECODER_OR_PROBE_ERRORS")
    ffmpeg = tools.get("ffmpeg", {})
    decoder_error = any(token in ffmpeg.get("stderr", "").lower() for token in
                        ("invalid nal", "decoding error", "decode error", "error splitting", "error submitting packet", "invalid data found when processing input", "corrupt"))
    if decoder_error and not ffmpeg.get("timed_out"):
        status = "TRUNCATED_STREAM" if evidence.get("opencv_actual_decoded_frames", 0) else "DECODE_ERROR"
    if evidence.get("ffmpeg_progress_end") is not True:
        reasons.append("MISSING_FFMPEG_PROGRESS_END")
    counts = [evidence.get(name) for name in ("opencv_actual_decoded_frames", "ffprobe_nb_read_frames", "ffprobe_nb_read_packets", "ffmpeg_decoded_frames")]
    if any(type(count) is not int or count <= 0 for count in counts):
        reasons.append("INVALID_OR_MISSING_DECODE_COUNT")
    elif len(set(counts)) != 1:
        reasons.append("INDEPENDENT_DECODE_COUNTS_DISAGREE")
    timeline = evidence.get("timestamp_continuity", {})
    if timeline.get("continuous") is not True:
        reasons.extend(timeline.get("errors") or ["TIMESTAMP_CONTINUITY_UNVERIFIED"])
    if timeline.get("frame_records") != counts[1] or timeline.get("packet_records") != counts[2]:
        reasons.append("TIMELINE_COUNT_MISMATCH")
    try:
        time_base = Fraction(evidence["time_base"])
        ticks = timeline["duration_ticks"]
        if time_base <= 0 or type(ticks) is not int or ticks <= 0 or ticks != timeline["duration_sum_ticks"]:
            raise ValueError("Invalid timeline duration")
        if int(evidence["stream_duration_ts"]) != ticks:
            reasons.append("STREAM_DURATION_TIMELINE_MISMATCH")
        if int(evidence["stream_start_pts"]) != timeline["first_frame_pts_ticks"]:
            reasons.append("STREAM_START_TIMELINE_MISMATCH")
        duration = Fraction(ticks) * time_base
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        reasons.append("SOURCE_DURATION_UNVERIFIED")
        duration = None
    metadata_count = evidence.get("nb_frames_metadata")
    if metadata_count is not None and metadata_count != counts[1]:
        reasons.append("EXPLICIT_CONTAINER_SAMPLE_COUNT_CONTRADICTION")
    reasons = sorted(set(reasons))
    if not reasons:
        status = "NORMAL_EOF"
    recovered_metadata = evidence.get("opencv_reported_frame_count") != counts[1]
    return {"decode_status": status, "technical_eligible": status == "NORMAL_EOF",
            "reason_code": ("CONTAINER_FRAME_COUNT_MISMATCH_BUT_DECODE_VERIFIED" if recovered_metadata else "INDEPENDENT_DECODE_AND_TIMELINE_VERIFIED") if not reasons else reasons[0],
            "exclusion_reasons": reasons, "duration_seconds": float(duration) if duration and not reasons else None,
            "duration_exact_rational": str(duration) if duration and not reasons else None,
            "duration_provenance": "LAST_SOURCE_PTS_PLUS_EXPLICIT_FRAME_DURATION_MINUS_FIRST_PTS",
            "human_review_status": "PENDING_HUMAN_REVIEW", "semantic_verified_negative": False,
            "scientific_far_eligible": False, "scientific_far_per_hour": None,
            "operating_point": "PROVISIONAL", "scientific_acceptance": "PARTIAL"}


def audit_video(entry: dict[str, Any], root: Path, folder: Path, ffmpeg: str, ffprobe: str, timeout: int = 1800) -> dict[str, Any]:
    """Collect fresh full-pass evidence for one source without modifying its bytes."""
    from ground_truth_studio.config import confined
    path = confined(root, entry["relative_path"])
    folder.mkdir(parents=True, exist_ok=False)
    digest = compute_streaming_sha256(path)
    record: dict[str, Any] = {"video_id": entry["video_id"], "source_path": entry["relative_path"],
        "source_sha256": digest, "expected_sha256": entry["sha256"], "file_size_bytes": path.stat().st_size,
        "audit_policy_version": POLICY, "sha256_verified": digest == entry["sha256"], "tools": {}}
    if not record["sha256_verified"]:
        record.update(classify(record))
        write_json(folder / "record.json", record)
        return record
    command = [ffprobe, "-v", "error", "-threads", "2", "-select_streams", "v:0", "-count_frames", "-count_packets",
        "-show_entries", "stream=codec_name,nb_frames,nb_read_frames,nb_read_packets,duration,duration_ts,start_pts,time_base,avg_frame_rate,r_frame_rate:format=duration", "-of", "json", str(path)]
    record["tools"]["counts"] = run_tool(command, folder, "counts", timeout)
    raw = json.loads((folder / "counts.stdout").read_text(encoding="utf-8")) if record["tools"]["counts"]["exit_code"] == 0 else {}
    stream = raw.get("streams", [{}])[0]
    record["raw_probe"] = raw
    for field in ("time_base", "avg_frame_rate", "r_frame_rate", "codec_name"):
        record[field] = stream.get(field)
    for source, target in (("nb_frames", "nb_frames_metadata"), ("nb_read_frames", "ffprobe_nb_read_frames"),
                           ("nb_read_packets", "ffprobe_nb_read_packets"), ("duration_ts", "stream_duration_ts"), ("start_pts", "stream_start_pts")):
        value = stream.get(source)
        record[target] = int(value) if value not in (None, "N/A") else None
    record.update(container_duration=raw.get("format", {}).get("duration"), stream_duration=stream.get("duration"))
    command = [ffprobe, "-v", "error", "-threads", "2", "-select_streams", "v:0", "-show_frames", "-show_packets",
        "-show_entries", "frame=pts,best_effort_timestamp,duration:packet=pts,dts,duration,flags", "-of", "compact=p=1:nk=0", str(path)]
    record["tools"]["timeline"] = run_tool(command, folder, "timeline", timeout)
    record["timestamp_continuity"] = inspect_timeline(folder / "timeline.stdout")
    command = [ffmpeg, "-nostdin", "-nostats", "-v", "error", "-xerror", "-threads", "2", "-i", str(path),
        "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "null", "-", "-progress", "pipe:1"]
    record["tools"]["ffmpeg"] = run_tool(command, folder, "ffmpeg", timeout)
    progress = dict(line.split("=", 1) for line in (folder / "ffmpeg.stdout").read_text().splitlines() if "=" in line)
    record["ffmpeg_progress_end"] = progress.get("progress") == "end"
    record["ffmpeg_decoded_frames"] = int(progress["frame"].strip()) if progress.get("frame", "").strip().isdigit() else None
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(path))
    try:
        record["opencv_backend"] = capture.getBackendName() if capture.isOpened() else None
        record["opencv_reported_frame_count"] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else None
        record["opencv_fps"] = capture.get(cv2.CAP_PROP_FPS)
        count = 0
        while capture.grab():
            count += 1
        record["opencv_actual_decoded_frames"] = count
    finally:
        capture.release()
    record["source_sha256_after_decode"] = compute_streaming_sha256(path)
    record["sha256_verified"] = record["source_sha256_after_decode"] == digest == entry["sha256"]
    record.update(classify(record))
    record["decode_termination_evidence"] = "FFmpeg progress/end + independent full count/timeline pass + OpenCV sequential decode; not OpenCV ret=False alone"
    record["artifact_sha256"] = {p.name: compute_streaming_sha256(p) for p in folder.iterdir() if p.is_file()}
    write_json(folder / "record.json", record)
    (folder / "record.sha256").write_text(compute_streaming_sha256(folder / "record.json"), encoding="ascii")
    LOGGER.info("%s %s %s", entry["video_id"], record["decode_status"], record["reason_code"])
    return record


def resume_record(folder: Path, entry: dict[str, Any], root: Path) -> dict[str, Any]:
    """Reuse this run's completed evidence only after policy/source/artifact checks."""
    from ground_truth_studio.config import confined
    record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
    if (folder / "record.sha256").read_text(encoding="ascii") != compute_streaming_sha256(folder / "record.json"):
        raise ValueError("Resume record checksum changed")
    if (record.get("audit_policy_version") != POLICY or record.get("video_id") != entry["video_id"] or
            record.get("source_sha256") != entry["sha256"] or compute_streaming_sha256(confined(root, entry["relative_path"])) != entry["sha256"]):
        raise ValueError("Resume source/policy mismatch; use a fresh run")
    for name, digest in record.get("artifact_sha256", {}).items():
        if compute_streaming_sha256(confined(folder, name)) != digest:
            raise ValueError("Resume raw artifact changed")
    if classify(record) != {key: record.get(key) for key in classify(record)}:
        raise ValueError("Resume classification changed")
    return record
