"""Collect real completed Week 5 runs on hash-verified validation labels.

This produces measured comparisons, not a selected or accepted model. The locked
video acceptance and baseline membership receipt remain separate requirements.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ground_truth_studio.config import confined
from scripts.finalize_week5_deliverable import digest, write_json
from scripts.run_week5_domain_adaptation import validate_inputs
from src.domain_adaptation import read_yolo_boxes, verify_file_hash
from src.week5_comparison import collect_comparison


def collect(args: argparse.Namespace) -> None:
    """Validate source artifacts before allowing any measured comparison output."""
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol["file_sha256"] = digest(args.protocol)
    vn, dfire = validate_inputs(dataset, "replay_20_percent")
    if not all(dataset.get("readiness_gates", {}).get(k) is True for k in ("VN_BBOX_READY", "DFIRE_READY", "SPLIT_READY")):
        raise ValueError("INDEPENDENT_READINESS_GATES_REQUIRED")
    baseline = json.loads(args.baseline_receipt.read_text(encoding="utf-8"))
    if (baseline.get("status") != "TRAINED" or baseline.get("training_executed") is not True
            or baseline.get("membership_status") != "VERIFIED"
            or baseline.get("checkpoint_sha256") != protocol["checkpoint_sha256"]):
        raise ValueError("BASELINE_TRAINING_MEMBERSHIP_RECEIPT_REQUIRED")
    planned = {r["image_sha256"] for r in dfire if r["split"] == "train"}
    if not planned or set(baseline.get("presented_image_sha256", [])) != planned:
        raise ValueError("BASELINE_DFIRE_TRAIN_MEMBERSHIP_MISMATCH")
    verify_file_hash(confined(ROOT, protocol["checkpoint"]), protocol["checkpoint_sha256"])
    support = {}
    for corpus, rows in (("vn", vn), ("dfire", dfire)):
        counts = {"SMOKE": 0, "FIRE": 0}
        for row in rows:
            if row["split"] == "val":
                for box in read_yolo_boxes(confined(ROOT, row["label_path"])):
                    counts[{0: "SMOKE", 1: "FIRE"}[int(box[0])]] += 1
        support[corpus] = counts
    receipts = []
    input_hashes = {str(path): digest(path) for path in (args.dataset_manifest, args.protocol, args.baseline_receipt)}
    for path in args.receipts:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        verify_file_hash(path.parent / "fit/weights/best.pt", receipt["best_checkpoint_sha256"])
        locked_path = path.parent / "locked_inputs.json"
        locked = json.loads(locked_path.read_text(encoding="utf-8"))
        if locked["dataset_manifest_sha256"] != digest(args.dataset_manifest):
            raise ValueError("LOCKED_DATASET_MISMATCH")
        for relative, checksum in locked["staged_input_sha256"].items():
            verify_file_hash(confined(path.parent, relative), checksum)
        receipts.append(receipt)
        input_hashes[str(path)] = digest(path)
        input_hashes[str(locked_path)] = digest(locked_path)
    rows = collect_comparison(receipts, protocol, digest(args.dataset_manifest), True, support)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "comparison.json", {"rows": rows, "scope": "validation", "support_units": "bbox_instances",
               "input_sha256": input_hashes, "selection_status": "NOT_SELECTED_REQUIRES_LOCKED_VIDEO_ACCEPTANCE"})
    with (args.output_dir / "comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Collect only real, complete, common-initialization experiment matrices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
