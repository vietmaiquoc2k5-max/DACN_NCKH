"""CPU-only, isolated v2 audit; never changes sources or scientific acceptance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ground_truth_studio.config import Settings, confined
from src.video_audit import compute_streaming_sha256 as sha256
from src.video_audit_v2 import POLICY, audit_video, resume_record, write_json

V1 = ROOT / "runs/benchmarks/week4_dataset_recovery/run_20261113_v2"
DEFAULT_OUTPUT = ROOT / "runs/benchmarks/week4_decode_audit_v2/run_20261113"


def group(record: dict[str, Any]) -> str:
    """Assign predeclared execution priority from v1 evidence, not detector output."""
    ext = record.get("external_tool_check", {})
    pts = record.get("ffprobe_metadata", {}).get("packet_timestamp_summary", {})
    if ext.get("is_clean_decode") is not True:
        return "D"
    if ext.get("nb_read_frames") != record.get("full_decoded_frames"):
        return "B"
    if pts.get("status") != "AVAILABLE" or pts.get("dts_regressions_in_packet_order") != 0:
        return "C"
    return "A"


def protected_files() -> list[Path]:
    """Inventory source-of-truth and historical artifacts without private DB writes."""
    files = [ROOT / name for name in ("data/negative_videos_manifest.json", "data/video_catalog_local_v2.json",
        "data/ground_truth/events_v1.1_snapshot.jsonl", "data/ground_truth/checksums_v1.1.sha256", "src/video_audit.py")]
    files.extend(p for p in V1.rglob("*") if p.is_file())
    run = ROOT / "runs/benchmarks/week4_full_inference/run_20261111_233035_neg_full"
    files.extend(p for p in run.rglob("*") if p.is_file())
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    checkpoint = confined(ROOT, manifest["model_checkpoint"].replace("\\", "/"))
    if checkpoint.exists():
        files.append(checkpoint)
    return files


def preflight(output: Path) -> dict[str, Any]:
    """Create a fresh immutable source scope and preservation inventory."""
    if output.exists():
        raise ValueError("Output exists; use a fresh directory (or audit --resume for this run)")
    settings = Settings.load()
    if not settings.ffmpeg or not settings.ffprobe:
        raise ValueError("FFmpeg/ffprobe required")
    audit = json.loads((V1 / "technical_audit_63.json").read_text(encoding="utf-8"))
    catalog = json.loads((ROOT / "data/video_catalog_local_v2.json").read_text(encoding="utf-8"))
    lookup = {row["video_id"]: row for row in catalog["videos"]}
    sources = []
    for row in sorted(audit["videos"], key=lambda row: (group(row), -row.get("duration_seconds", 0), row["video_id"])):
        if row["decode_status"] != "NEEDS_REVIEW":
            continue
        entry = lookup[row["video_id"]]
        path = confined(ROOT, entry["relative_path"])
        if not path.is_file() or sha256(path) != entry["sha256"] or entry["sha256"] != row["sha256"]:
            raise ValueError("Source identity mismatch: " + entry["video_id"])
        sources.append({"video_id": entry["video_id"], "relative_path": entry["relative_path"], "sha256": entry["sha256"],
                        "priority_group": group(row), "v1_duration_seconds": row["duration_seconds"]})
    assert len(sources) == 58
    output.mkdir(parents=True)
    (output / "records").mkdir()
    payload = {"policy": POLICY, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": sources, "execution_rule": "ALL_58; group A/B/C/D, longest v1 duration first; no model outcomes used; no early stopping",
        "workers": 1, "ffmpeg_decoder_threads": 2, "ffmpeg": settings.ffmpeg, "ffprobe": settings.ffprobe,
        "implementation_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in (ROOT / "src/video_audit_v2.py", Path(__file__).resolve())},
        "protected_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in protected_files()},
        "tool_versions": {name: subprocess.check_output([binary, "-version"], text=True).splitlines()[0] for name, binary in (("ffmpeg", settings.ffmpeg), ("ffprobe", settings.ffprobe))},
        "human_review_status": "UNCHANGED_PENDING", "inference_run": False, "scientific_far_per_hour": None}
    write_json(output / "preflight.json", payload)
    return payload


def summarize(output: Path) -> dict[str, Any]:
    """Report new v2 results plus separately identified, authenticated v1 passes."""
    run = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    for name, digest in run["protected_sha256"].items():
        if sha256(confined(ROOT, name)) != digest:
            raise ValueError("Protected artifact changed: " + name)
    records = []
    for entry in run["scope"]:
        folder = output / "records" / entry["video_id"]
        if (folder / "record.json").exists():
            records.append(resume_record(folder, entry, ROOT))
    old = json.loads((V1 / "technical_audit_63.json").read_text(encoding="utf-8"))
    retained = []
    for row in old["videos"]:
        if row["decode_status"] == "NORMAL_EOF" and row.get("full_decode_verified") and row.get("sha256_verified"):
            if sha256(confined(ROOT, row["relative_path"])) != row["sha256"]:
                raise ValueError("Retained v1 source changed")
            retained.append({"video_id": row["video_id"], "source_sha256": row["sha256"],
                             "duration_seconds": row["duration_seconds"], "evidence_policy": "strict_full_stream_zero_tolerance_v1"})
    accepted = [{"video_id": row["video_id"], "source_sha256": row["source_sha256"], "duration_seconds": row["duration_seconds"],
                 "duration_exact_rational": row["duration_exact_rational"], "evidence_policy": POLICY} for row in records if row["technical_eligible"]]
    identity = sorted([*retained, *accepted], key=lambda row: row["video_id"])
    scope_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    hours = sum(row["duration_seconds"] for row in identity) / 3600
    result = {"policy": POLICY, "status": "COMPLETE" if len(records) == len(run["scope"]) else "RUNNING",
        "audited_v2": len(records), "expected_v2": len(run["scope"]), "v2_counts": dict(Counter(row["decode_status"] for row in records)),
        "v1_retained_normal_eof": retained, "v2_accepted": accepted,
        "v2_excluded": [{"video_id": row["video_id"], "decode_status": row["decode_status"], "reasons": row["exclusion_reasons"]} for row in records if not row["technical_eligible"]],
        "not_audited": [entry["video_id"] for entry in run["scope"] if entry["video_id"] not in {row["video_id"] for row in records}],
        "v1_hours": sum(row["duration_seconds"] for row in retained) / 3600,
        "recovered_hours_v2_only": sum(row["duration_seconds"] for row in accepted) / 3600,
        "technical_valid_negative_candidate_hours": hours, "technical_dataset_status": "PASS" if hours >= 5 else "PARTIAL",
        "eligible_scope_sha256": scope_hash, "eligible_scope_identity": identity,
        "v1_truncated_excluded_not_reaudited": [row["video_id"] for row in old["videos"] if row["decode_status"] == "TRUNCATED_STREAM"],
        "classification_diff": [{"video_id": row["video_id"], "v1": "NEEDS_REVIEW", "v2": row["decode_status"], "reason_code": row["reason_code"]} for row in records],
        "source_and_v1_preservation_verified": True, "human_review_status": "PENDING_HUMAN_REVIEW",
        "scientific_eligible_negative_hours": None, "far_per_hour": None, "poisson_ci_95": None,
        "operating_point": "PROVISIONAL", "scientific_week4": "PARTIAL", "official_acceptance": False}
    write_json(output / "summary.json", result)
    fields = ["video_id", "decode_status", "reason_code", "duration_seconds", "source_sha256", "opencv_reported_frame_count",
              "opencv_actual_decoded_frames", "ffprobe_nb_read_frames", "ffprobe_nb_read_packets", "ffmpeg_decoded_frames", "technical_eligible"]
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "audit", "summarize"], required=True)
    parser.add_argument("--scope", choices=["needs_review"], default="needs_review")
    parser.add_argument("--policy", choices=[POLICY], default=POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--one", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "runs/benchmarks/week4_decode_audit_v2") or args.timeout <= 0:
        parser.error("Use a new v2 benchmark directory and a positive timeout")
    if args.mode == "preflight":
        result = preflight(output)
        print(json.dumps({"scope": len(result["scope"]), "groups": dict(Counter(row["priority_group"] for row in result["scope"]))}))
        return
    run = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    if args.mode == "summarize":
        result = summarize(output)
        print(json.dumps({k: result[k] for k in ("status", "v2_counts", "technical_valid_negative_candidate_hours", "technical_dataset_status")}))
        return
    for name, digest in run["implementation_sha256"].items():
        if sha256(confined(ROOT, name)) != digest:
            raise ValueError("Audit implementation changed; use a fresh run")
    if args.one:
        entry = next(row for row in run["scope"] if row["video_id"] == args.one)
        audit_video(entry, ROOT, output / "records" / args.one, run["ffmpeg"], run["ffprobe"], args.timeout)
        return
    logging.basicConfig(filename=output / "execution.log", level=logging.INFO, format="%(asctime)s %(message)s")
    if not args.resume and any((output / "records").iterdir()):
        raise ValueError("Run already started; use --resume explicitly")
    for index, entry in enumerate(run["scope"], 1):
        folder = output / "records" / entry["video_id"]
        if (folder / "record.json").exists():
            row = resume_record(folder, entry, ROOT)
        else:
            if folder.exists():
                # Preserve this run's incomplete attempt before creating a new one.
                archive = output / "incomplete_attempts" / (entry["video_id"] + "_" + uuid.uuid4().hex)
                if not folder.resolve().is_relative_to(output) or not archive.resolve().is_relative_to(output):
                    raise ValueError("Unsafe incomplete-attempt path")
                archive.parent.mkdir(exist_ok=True)
                folder.rename(archive)
            command = [sys.executable, str(Path(__file__).resolve()), "--mode", "audit", "--output-dir", str(output), "--one", entry["video_id"], "--timeout", str(args.timeout)]
            log = output / (entry["video_id"] + ".worker.log")
            with log.open("xb") as stream:
                try:
                    process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout * 4 + 120)
                    if process.returncode:
                        raise RuntimeError("Worker failed; retained log: " + entry["video_id"])
                except subprocess.TimeoutExpired:
                    raise RuntimeError("Worker timeout, including OpenCV; incomplete evidence retained; not eligible")
            row = json.loads((folder / "record.json").read_text(encoding="utf-8"))
        logging.info("%s/%s %s %s %s", index, len(run["scope"]), entry["video_id"], row["decode_status"], row["reason_code"])
        print(f"AUDIT {index}/{len(run['scope'])} {entry['video_id']} {row['decode_status']} {row['reason_code']}", flush=True)
        # Incremental report; perform full protected/source revalidation at final summary.
    summarize(output)


if __name__ == "__main__":
    main()
