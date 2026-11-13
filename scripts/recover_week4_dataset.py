"""Reproducible local catalog and strict decode audit. No inference or human labels."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ground_truth_studio.config import Settings, confined
from ground_truth_studio.catalog_recovery import recover_catalog, save_json
from ground_truth_studio.integrity_service import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["catalog", "audit", "reconcile", "evaluate"], default="catalog")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publish-catalog", action="store_true")
    parser.add_argument("--audit-one")
    parser.add_argument("--review-export", type=Path)
    args = parser.parse_args()
    settings = Settings.load()
    output = args.output_dir.resolve()
    if not output.is_relative_to(settings.repo / "runs/benchmarks"):
        parser.error("Use a fresh benchmark directory inside this repository")
    if subprocess.check_output(["git", "ls-files", "--", str(output)], cwd=settings.repo, text=True).strip():
        parser.error("Refusing to overwrite tracked historical artifacts")
    output.mkdir(parents=True, exist_ok=True)
    if settings.ffmpeg:
        os.environ["PATH"] = str(Path(settings.ffmpeg).parent) + os.pathsep + os.environ.get("PATH", "")
    if args.phase == "catalog":
        recover_catalog(settings, output, publish=args.publish_catalog)
        return
    if args.phase == "reconcile":
        from ground_truth_studio.inference_reconciliation import reconcile
        reconcile(settings, output)
        return
    if args.phase == "evaluate":
        from ground_truth_studio.provisional_evaluation import evaluate_provisional
        reviews = []
        if args.review_export:
            folder = args.review_export.resolve()
            hashes = json.loads((folder / "file_sha256.json").read_text(encoding="utf-8"))
            for name, digest in hashes.items():
                if sha256(confined(folder, name)) != digest:
                    raise ValueError("Export checksum mismatch: " + name)
            reviews = json.loads((folder / "single_review_records.json").read_text(encoding="utf-8"))
            for row in reviews:
                if sha256(confined(settings.repo, row["relative_path"])) != row["source_sha256"]:
                    raise ValueError("Reviewed source changed")
        audit_file = output / "technical_audit_63.json"
        records = json.loads(audit_file.read_text(encoding="utf-8"))["videos"] if audit_file.exists() else []
        # This historical run has no authenticated full timeline. Do not turn its
        # aggregate episodes into trusted input merely because an export exists.
        result = evaluate_provisional(reviews, {row["video_id"]: row for row in records}, {})
        result["inference_input_status"] = "UNVERIFIED_MISSING_EVIDENCE"
        result["review_export"] = str(args.review_export) if args.review_export else None
        save_json(output / "provisional_evaluation.json", result)
        save_json(output / "ground_truth_candidate_status.json", {
            "status": result["human_review_status"], "submitted_records_in_input": len(reviews),
            "candidate_generated": bool(reviews), "fabricated_labels": False,
            "instruction": "Submit actual reviews then export in Studio; no candidate labels generated from drafts."})
        return
    local = json.loads((output / "video_catalog_local_v2.json").read_text(encoding="utf-8"))
    entries = [row for row in local["videos"] if row["source_group"] == "VN_NEG" and row["source_available"]]
    records_dir = output / "audit_records"
    records_dir.mkdir(exist_ok=True)
    if args.audit_one:
        from src.video_audit import audit_single_video
        entry = next(row for row in entries if row["video_id"] == args.audit_one)
        path = confined(settings.repo, entry["relative_path"])
        record = audit_single_video(path, root_dir=settings.repo, full_decode=True, cache_manager=None)
        record["source_sha256_after_decode"] = sha256(path)
        record["sha256_verified"] = record["sha256"] == record["source_sha256_after_decode"] == entry["sha256"]
        if not record["sha256_verified"]:
            record.update(decode_status="NEEDS_REVIEW", full_decode_verified=False, error_message="SOURCE_CHANGED_OR_HASH_MISMATCH")
        record["video_id"] = entry["video_id"]
        save_json(records_dir / (entry["video_id"] + ".json"), record)
        return
    from src.video_audit import AUDIT_POLICY_VERSION
    results = []
    started = time.time()
    for index, entry in enumerate(entries, 1):
        target = records_dir / (entry["video_id"] + ".json")
        if target.exists():
            prior = json.loads(target.read_text(encoding="utf-8"))
            if (prior.get("audit_policy_version") != AUDIT_POLICY_VERSION or
                    prior.get("sha256") != entry["sha256"] or
                    sha256(confined(settings.repo, entry["relative_path"])) != entry["sha256"]):
                raise ValueError("Resume evidence/source changed; choose a fresh output directory: " + entry["video_id"])
        if not target.exists():
            command = [sys.executable, str(Path(__file__).resolve()), "--phase", "audit", "--output-dir", str(output), "--audit-one", entry["video_id"]]
            try:
                result = subprocess.run(command, capture_output=True, timeout=3600)
                (records_dir / (entry["video_id"] + ".log")).write_bytes(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError(f"Audit child exit {result.returncode}")
            except (subprocess.TimeoutExpired, RuntimeError) as error:
                save_json(target, {"video_id": entry["video_id"], "decode_status": "NEEDS_REVIEW", "full_decode_verified": False,
                                   "sha256_verified": False, "error_message": str(error), "audit_policy_version": AUDIT_POLICY_VERSION})
        record = json.loads(target.read_text(encoding="utf-8"))
        results.append(record)
        counts = {state: sum(row["decode_status"] == state for row in results) for state in {row["decode_status"] for row in results}}
        save_json(output / "technical_audit_63.json", {"scope_id": local["scope_id"], "audit_policy_version": AUDIT_POLICY_VERSION,
            "status": "COMPLETE" if len(results) == len(entries) else "RUNNING", "completed_videos": len(results), "expected_videos": len(entries),
            "elapsed_seconds": time.time() - started, "decode_status_counts": counts, "videos": results,
            "decode_only_hours": sum(row.get("duration_seconds", 0) for row in results if row["decode_status"] == "NORMAL_EOF" and row.get("full_decode_verified") and row.get("sha256_verified")) / 3600,
            "far_eligible_hours": None, "semantic_gt_verified": False})
        print(f"AUDIT {index}/{len(entries)} {entry['video_id']} {record['decode_status']} elapsed={time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
