"""Validate comparable Week 5 receipts without turning missing results into scores."""

from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Any

from src.domain_adaptation import map_delta, select_adapted_model

STRATEGIES = ("full_finetune", "freeze_backbone", "replay_20_percent")


def training_device(cuda_available: bool, cpu_fallback: bool = True) -> str:
    """Choose one GPU or CPU explicitly; reject an unauthorized CPU fallback."""
    if cuda_available:
        return "0"
    if not cpu_fallback:
        raise ValueError("CUDA_REQUIRED_CPU_FALLBACK_DISABLED")
    return "cpu"


def select_final_model(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Add proposal MT4/class-support requirements to locked v2 video selection.

    Deltas are fractions, so 0.05 means five AP percentage points. This stricter
    deliverable acceptance never changes the training or evaluation protocol.
    """
    accepted = []
    for candidate in candidates:
        if candidate.get("scope") != "validation":
            raise ValueError("TEST_RESULTS_CANNOT_SELECT_MODEL")
        delta = candidate.get("delta_vn_map50")
        retention = candidate.get("delta_dfire_map50")
        supports = candidate.get("class_support", {})
        if any(type(supports.get(c, {}).get(label)) is not int or supports[c][label] <= 0
               for c in ("vn", "dfire") for label in ("SMOKE", "FIRE")):
            continue
        if (type(delta) not in (int, float) or type(retention) not in (int, float)
                or not math.isfinite(delta) or not math.isfinite(retention)
                or delta < .05 or retention < -.02):
            continue
        accepted.append(candidate)
    result = select_adapted_model(accepted)
    return {"status": "VALIDATION_SELECTED_TEST_SEALED" if result["selected"] else "NO_ACCEPTED_WINNER",
            "selected": result["selected"]}


def measured_metrics(value: dict[str, Any], support: dict[str, int]) -> dict[str, Any]:
    """Validate AP and retain absent classes as unavailable, including aggregates."""
    result: dict[str, Any] = {"support": support, "per_class": {}}
    for name in ("SMOKE", "FIRE"):
        if type(support.get(name)) is not int or support[name] < 0:
            raise ValueError("INVALID_CLASS_SUPPORT")
        row = value.get("per_class", {}).get(name, {})
        result["per_class"][name] = {}
        for key in ("map50", "map50_95"):
            metric = row.get(key) if support[name] else None
            if support[name] and (type(metric) not in (float, int)
                                  or not math.isfinite(metric) or not 0 <= metric <= 1):
                raise ValueError("MISSING_OR_INVALID_SUPPORTED_CLASS_AP")
            result["per_class"][name][key] = metric
    for key in ("map50", "map50_95"):
        metric = value.get(key) if all(support.values()) else None
        if all(support.values()) and (type(metric) not in (float, int)
                                      or not math.isfinite(metric) or not 0 <= metric <= 1):
            raise ValueError("INVALID_MAP")
        result[key] = metric
    return result


def comparison_row(name: str, metrics: dict[str, Any] | None = None,
                   baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute signed deltas and nonnegative forgetting in fractional AP units."""
    row: dict[str, Any] = {"model": name, "status": "NOT_MEASURED" if metrics is None else "MEASURED"}
    for corpus in ("vn", "dfire"):
        for metric in ("map50", "map50_95"):
            value = (metrics or {}).get(corpus, {}).get(metric)
            original = (baseline or {}).get(corpus, {}).get(metric)
            row[f"{corpus}_{metric}"] = value
            row[f"delta_{corpus}_{metric}"] = map_delta(value, original)
        for label in ("SMOKE", "FIRE"):
            for metric in ("map50", "map50_95"):
                row[f"{corpus}_{label.lower()}_{metric}"] = (
                    (metrics or {}).get(corpus, {}).get("per_class", {}).get(label, {}).get(metric))
            row[f"{corpus}_{label.lower()}_support"] = (metrics or {}).get(corpus, {}).get("support", {}).get(label)
    for metric in ("map50", "map50_95"):
        delta = row[f"delta_dfire_{metric}"]
        row[f"forgetting_{metric}"] = None if delta is None else max(0.0, -delta)
    return row


def validate_comparability(receipts: list[dict[str, Any]], protocol: dict[str, Any],
                           dataset_sha256: str, baseline_membership_verified: bool) -> None:
    """Reject partial runs, different initializations/budgets and missing seeds."""
    if protocol.get("evaluation", {}).get("split_for_selection") != "val":
        raise ValueError("TEST_RESULTS_CANNOT_SELECT_MODEL")
    if not baseline_membership_verified:
        raise ValueError("BASELINE_MEMBERSHIP_NOT_VERIFIED")
    expected = {(strategy, seed) for strategy in STRATEGIES for seed in protocol["seeds"]}
    keys = [(r.get("strategy"), r.get("seed")) for r in receipts]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("INCOMPLETE_OR_DUPLICATED_RUN_MATRIX")
    for receipt in receipts:
        if (receipt.get("status") != "TRAINED" or receipt.get("training_executed") is not True
                or receipt.get("epochs_actually_completed") != protocol["epochs"]):
            raise ValueError("REAL_COMPLETED_TRAINING_REQUIRED")
        if receipt.get("baseline_sha256") != protocol["checkpoint_sha256"]:
            raise ValueError("INITIALIZATION_MISMATCH")
        if receipt.get("dataset_sha256") != dataset_sha256:
            raise ValueError("EVALUATION_MEMBERSHIP_MISMATCH")
        if receipt.get("config_sha256") != protocol["file_sha256"]:
            raise ValueError("OPTIMIZATION_OR_EVALUATOR_MISMATCH")
        if not receipt.get("best_checkpoint_sha256"):
            raise ValueError("CHECKPOINT_EVIDENCE_REQUIRED")


def selection_blockers(readiness: dict[str, Any]) -> list[str]:
    """Use explicit boolean evidence; counts and review completion cannot open gates."""
    required = ("SPLIT_READY", "VN_BBOX_READY", "DFIRE_READY", "RESOURCE_PILOT_READY", "WEEK5_TRAINING_READY")
    return [key for key in required if readiness.get(key) is not True]


def collect_comparison(receipts: list[dict[str, Any]], protocol: dict[str, Any],
                       dataset_sha256: str, baseline_membership_verified: bool,
                       support: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    """Collect per-seed runner measurements only on one locked validation scope.

    Support must be counted from the hash-verified evaluation labels, never
    prediction counts. Missing classes produce null aggregate AP. Selection is
    separate and still requires the locked paired-video/runtime evidence.
    """
    validate_comparability(receipts, protocol, dataset_sha256, baseline_membership_verified)
    baselines: dict[int, dict[str, Any]] = {}
    output = []
    for receipt in sorted(receipts, key=lambda r: (r["seed"], r["strategy"])):
        base = {c: measured_metrics(receipt["baseline_validation"][c], support[c]) for c in ("vn", "dfire")}
        adapted = {c: measured_metrics(receipt["adapted_validation"][c], support[c]) for c in ("vn", "dfire")}
        seed = receipt["seed"]
        if seed in baselines and base != baselines[seed]:
            raise ValueError("BASELINE_EVALUATION_NOT_COMPARABLE")
        if seed not in baselines:
            output.append({**comparison_row("Baseline", base, base), "seed": seed})
            baselines[seed] = base
        output.append({**comparison_row(receipt["strategy"], adapted, base), "seed": seed})
    return output


def export_completed_checkpoint(source: Path, target: Path, receipt: dict[str, Any], *, required_epochs: int) -> str:
    """Copy a completed run checkpoint exclusively, never create a winner alias.

    This exports individual strategy artifacts, not acceptance. A best_adapted
    export is intentionally unavailable without the separate video acceptance.
    """
    allowed = dict(zip(STRATEGIES, ("strategy_a_best.pt", "strategy_b_best.pt", "strategy_c_best.pt")))
    if target.name != allowed.get(receipt.get("strategy")):
        raise ValueError("CHECKPOINT_EXPORT_NOT_ACCEPTED_WINNER")
    if (receipt.get("status") != "TRAINED" or receipt.get("training_executed") is not True
            or required_epochs <= 0 or receipt.get("epochs_actually_completed") != required_epochs):
        raise ValueError("COMPLETED_RUN_REQUIRED")
    payload = source.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != receipt.get("best_checkpoint_sha256"):
        raise ValueError("CHECKPOINT_SHA256_MISMATCH")
    with target.open("xb") as stream:
        stream.write(payload)
    return actual
