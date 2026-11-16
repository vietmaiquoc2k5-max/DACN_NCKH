"""Train a fresh D-Fire-only baseline after locked split and measured pilot gates."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ground_truth_studio.config import confined
from scripts.finalize_week5_deliverable import digest
from scripts.run_week5_domain_adaptation import data_config, stage_rows, validate_budget
from src.domain_adaptation import (
    extract_map, read_yolo_boxes, validate_class_mapping, validate_dataset_rows, verify_file_hash,
)
from src.week5_comparison import training_device


def baseline_rows(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """Require actual D-Fire provenance and split gates before any model load."""
    validate_class_mapping(dataset.get("class_mapping"))
    if (dataset.get("status") != "READY_FOR_TRAINING" or dataset.get("split_locked") is not True
            or dataset.get("leakage_review_complete") is not True
            or not all(dataset.get("readiness_gates", {}).get(k) is True for k in ("DFIRE_READY", "SPLIT_READY"))):
        raise ValueError("DFIRE_LOCKED_INDEPENDENT_DATA_REQUIRED")
    if (dataset.get("dfire_annotation_origin") != "ORIGINAL_DFIRE_ANNOTATIONS"
            or not dataset.get("dfire_split_provenance_sha256")):
        raise ValueError("DFIRE_ORIGINAL_ANNOTATIONS_AND_SPLIT_PROVENANCE_REQUIRED")
    rows = [r for r in dataset["rows"] if r.get("corpus") == "DFIRE"]
    validate_dataset_rows(rows, require_human=False)
    if not {"train", "val"} <= {r["split"] for r in rows}:
        raise ValueError("DFIRE_TRAIN_AND_VALIDATION_REQUIRED")
    for row in rows:
        if row.get("reserved_test") is True and row["split"] != "test":
            raise ValueError("RESERVED_TEST_PROTECTION")
    return rows


def execute(args: argparse.Namespace) -> int:
    """Measure actual presentations, epochs and checkpoint hashes; no auto-resume."""
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "runs/benchmarks/week5_domain_adaptation"):
        raise ValueError("UNSAFE_OUTPUT_DIRECTORY")
    output.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {"status": "PREFLIGHT", "training_executed": False,
                              "epochs_actually_completed": 0, "membership_status": "NOT_VERIFIED",
                              "presented_image_sha256": [], "epoch_history": []}
    def save() -> None:
        # This is run-local progress, not production annotation storage.
        pending = output / "baseline_receipt.pending"
        pending.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
        pending.replace(output / "baseline_receipt.json")
    save()
    try:
        dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        rows = baseline_rows(dataset)
        if args.seed not in protocol["training_limits"]["seeds"]:
            raise ValueError("UNDECLARED_SEED")
        limits = protocol["training_limits"]
        if (protocol.get("split_locked") is not True or protocol.get("dataset_sha256") != digest(args.dataset_manifest)):
            raise ValueError("BASELINE_PROTOCOL_NOT_BOUND_TO_LOCKED_DATA")
        initialization = confined(ROOT, protocol["initialization_path"])
        verify_file_hash(initialization, protocol["initialization_sha256"])
        budget = json.loads(args.budget_receipt.read_text(encoding="utf-8"))
        validate_budget(budget, digest(args.dataset_manifest), protocol["initialization_sha256"], limits["epochs"])
        import cv2
        import torch
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import DetectionTrainer
        device = training_device(torch.cuda.is_available(), limits["cpu_fallback"])
        if budget.get("device") != device:
            raise ValueError("PILOT_DEVICE_MISMATCH")
        active = [r for r in rows if r["split"] in {"train", "val"}]
        for row in active:
            for hash_field, path_field in (("image_sha256", "image_path"), ("label_sha256", "label_path")):
                verify_file_hash(confined(ROOT, row[path_field]), row[hash_field])
            if cv2.imread(str(confined(ROOT, row["image_path"]))) is None:
                raise ValueError("UNREADABLE_IMAGE")
            read_yolo_boxes(confined(ROOT, row["label_path"]))
        for split in ("train", "val"):
            stage_rows([r for r in active if r["split"] == split], output / "dfire", split)
        data_config(output / "dfire.yaml", output / "dfire")
        expected = Counter(p.stem for p in (output / "dfire/images/train").glob("*"))
        epoch_files: list[str] = []
        before = time.monotonic()
        receipt.update(dataset_sha256=digest(args.dataset_manifest), protocol_sha256=digest(args.protocol),
                       initialization_sha256=protocol["initialization_sha256"], seed=args.seed,
                       budget_receipt_sha256=digest(args.budget_receipt), class_mapping={"0": "smoke", "1": "fire"},
                       locked_train_image_sha256=sorted(r["image_sha256"] for r in active if r["split"] == "train"))
        (output / "dataset_membership.json").write_text(json.dumps(active, indent=2), encoding="utf-8")
        class MembershipTrainer(DetectionTrainer):
            """Record images actually loaded for optimization, not planned inputs."""
            def preprocess_batch(self, batch: dict) -> dict:
                epoch_files.extend(Path(path).stem for path in batch["im_file"])
                return super().preprocess_batch(batch)
        def started(trainer: Any) -> None:
            receipt.update(status="RUNNING", training_executed=True, device=str(trainer.device))
            save()
        def batch_start(trainer: Any) -> None:
            if time.monotonic() - before > budget["max_training_hours"] * 3600:
                raise TimeoutError("TRAINING_RESOURCE_BUDGET_EXCEEDED")
        def epoch_end(trainer: Any) -> None:
            if int(trainer.epoch) + 1 <= receipt["epochs_actually_completed"] or receipt["epochs_actually_completed"] >= limits["epochs"]:
                return
            if Counter(epoch_files) != expected:
                raise ValueError("ACTUAL_TRAIN_MEMBERSHIP_MISMATCH")
            receipt["presented_image_sha256"] = receipt["locked_train_image_sha256"]
            receipt["epochs_actually_completed"] = int(trainer.epoch) + 1
            receipt["epoch_history"].append({"epoch": int(trainer.epoch) + 1,
                                             "presentations": len(epoch_files),
                                             "metrics": {str(k): float(v) for k, v in trainer.metrics.items()}})
            save()
        model = YOLO(str(initialization))
        model.add_callback("on_train_start", started)
        model.add_callback("on_train_epoch_start", lambda trainer: epoch_files.clear())
        model.add_callback("on_train_batch_start", batch_start)
        model.add_callback("on_fit_epoch_end", epoch_end)
        keys = ("epochs", "imgsz", "batch", "workers", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
                "warmup_epochs", "deterministic", "amp", "mosaic", "mixup", "copy_paste", "cutmix", "close_mosaic", "fraction", "compile")
        model.train(trainer=MembershipTrainer, data=str(output / "dfire.yaml"), project=str(output), name="fit",
                    exist_ok=False, seed=args.seed, device=device, time=None, patience=limits["epochs"] + 1,
                    **{key: limits[key] for key in keys})
        if receipt["epochs_actually_completed"] != limits["epochs"]:
            raise ValueError("PARTIAL_BASELINE_TRAINING")
        checkpoint = output / "fit/weights/best.pt"
        trained = YOLO(str(checkpoint))
        validate_class_mapping(trained.names)
        receipt["validation"] = extract_map(trained.val(data=str(output / "dfire.yaml"), split="val",
            imgsz=limits["imgsz"], conf=.001, iou=.7, device=device, workers=0, plots=False))
        receipt.update(status="TRAINED", membership_status="VERIFIED", checkpoint_sha256=digest(checkpoint),
                       checkpoint_path=str(checkpoint), runtime_seconds=time.monotonic() - before)
        save()
        return 0
    except (Exception, KeyboardInterrupt) as error:
        receipt.update(status="INTERRUPTED" if receipt["training_executed"] else "BLOCKED",
                       error=str(error), error_type=type(error).__name__)
        save()
        return 1 if receipt["training_executed"] else 2


def main() -> None:
    """Run only after the actual dataset, initialization and pilot are hash bound."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--budget-receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    raise SystemExit(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
