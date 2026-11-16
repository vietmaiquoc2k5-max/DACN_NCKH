"""Synthetic measurements exercise reporting contracts, not scientific results."""
from __future__ import annotations

import copy
import hashlib

import pytest

from src.week5_comparison import (
    STRATEGIES, collect_comparison, comparison_row, export_completed_checkpoint,
    measured_metrics, selection_blockers, validate_comparability,
)


def metric(value=.6):
    return {"map50": value, "map50_95": value / 2,
            "per_class": {c: {"map50": value, "map50_95": value / 2} for c in ("SMOKE", "FIRE")}}


def matrix():
    protocol = {"seeds": [42, 123, 456], "epochs": 30, "checkpoint_sha256": "a" * 64,
                "file_sha256": "b" * 64, "evaluation": {"split_for_selection": "val"}}
    receipts = [{"strategy": strategy, "seed": seed, "status": "TRAINED", "training_executed": True,
                 "epochs_actually_completed": 30, "baseline_sha256": "a" * 64,
                 "config_sha256": "b" * 64, "dataset_sha256": "c" * 64,
                 "best_checkpoint_sha256": "d" * 64,
                 "baseline_validation": {c: metric() for c in ("vn", "dfire")},
                 "adapted_validation": {"vn": metric(.7), "dfire": metric(.58)}}
                for strategy in STRATEGIES for seed in protocol["seeds"]]
    return receipts, protocol


def test_three_seed_common_scope_delta_and_forgetting():
    receipts, protocol = matrix()
    rows = collect_comparison(receipts, protocol, "c" * 64, True,
                              {c: {"SMOKE": 10, "FIRE": 8} for c in ("vn", "dfire")})
    assert len(rows) == 12
    adapted = [r for r in rows if r["model"] != "Baseline"]
    assert all(r["delta_vn_map50"] == pytest.approx(.1) for r in adapted)
    assert all(r["forgetting_map50"] == pytest.approx(.02) for r in adapted)


@pytest.mark.parametrize("field,value,error", [
    ("baseline_sha256", "different", "INITIALIZATION"),
    ("dataset_sha256", "different", "MEMBERSHIP"),
    ("config_sha256", "different", "EVALUATOR"),
    ("epochs_actually_completed", 1, "COMPLETED"),
    ("training_executed", False, "COMPLETED"),
    ("status", "BLOCKED", "COMPLETED"),
    ("best_checkpoint_sha256", None, "CHECKPOINT"),
])
def test_changed_initialization_scope_budget_or_partial_run_rejected(field, value, error):
    receipts, protocol = matrix()
    receipts[0][field] = value
    with pytest.raises(ValueError, match=error):
        validate_comparability(receipts, protocol, "c" * 64, True)


def test_membership_unknown_and_missing_seed_rejected():
    receipts, protocol = matrix()
    with pytest.raises(ValueError, match="MEMBERSHIP"):
        validate_comparability(receipts, protocol, "c" * 64, False)
    for invalid in (receipts[:-1], receipts + [receipts[0]]):
        with pytest.raises(ValueError, match="MATRIX"):
            validate_comparability(invalid, protocol, "c" * 64, True)


def test_missing_class_not_zero_or_interpretable_aggregate():
    result = measured_metrics(metric(), {"SMOKE": 20, "FIRE": 0})
    assert result["per_class"]["FIRE"]["map50"] is None
    assert result["map50"] is None
    assert result["per_class"]["SMOKE"]["map50"] == .6


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 1.1, None])
def test_invalid_ap_rejected(value):
    data = metric()
    data["per_class"]["FIRE"]["map50"] = value
    with pytest.raises(ValueError, match="AP"):
        measured_metrics(data, {"SMOKE": 2, "FIRE": 1})


def test_baseline_evaluation_drift_rejected():
    receipts, protocol = matrix()
    receipts[0]["baseline_validation"]["vn"] = metric(.8)
    with pytest.raises(ValueError, match="COMPARABLE"):
        collect_comparison(receipts, protocol, "c" * 64, True,
                           {c: {"SMOKE": 2, "FIRE": 1} for c in ("vn", "dfire")})


def test_no_metric_is_not_zero_and_no_negative_forgetting():
    assert comparison_row("A")["forgetting_map50"] is None
    assert comparison_row("A", {"dfire": {"map50": .7}}, {"dfire": {"map50": .6}})["forgetting_map50"] == 0


def test_export_hash_and_exclusive_write_without_winner_alias(tmp_path):
    source = tmp_path / "fixture.bin"
    source.write_bytes(b"synthetic fixture; not a trained checkpoint")
    receipts, _ = matrix()
    receipt = copy.deepcopy(receipts[0])
    receipt["best_checkpoint_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / "strategy_a_best.pt"
    export_completed_checkpoint(source, target, receipt, required_epochs=30)
    assert target.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        export_completed_checkpoint(source, target, receipt, required_epochs=30)
    with pytest.raises(ValueError, match="WINNER"):
        export_completed_checkpoint(source, tmp_path / "best_adapted.pt", receipt, required_epochs=30)
    source.write_bytes(b"modified")
    target2 = tmp_path / "another" / "strategy_a_best.pt"
    with pytest.raises(ValueError, match="SHA256"):
        export_completed_checkpoint(source, target2, receipt, required_epochs=30)
    assert not target2.exists()


def test_approved_counts_never_open_training_gate():
    assert len(selection_blockers({"approved": 1922, "BBOX_REVIEW_COMPLETE": True})) == 5


def baseline_dataset():
    rows = [{"corpus": "DFIRE", "split": split, "image_sha256": str(i) * 64,
             "source_group_id": f"source{i}", "scene_group_id": f"scene{i}"}
            for i, split in enumerate(("train", "val", "test"), 1)]
    return {"class_mapping": ["smoke", "fire"], "status": "READY_FOR_TRAINING", "split_locked": True,
            "leakage_review_complete": True, "readiness_gates": {"DFIRE_READY": True, "SPLIT_READY": True},
            "dfire_annotation_origin": "ORIGINAL_DFIRE_ANNOTATIONS", "dfire_split_provenance_sha256": "a" * 64,
            "rows": rows}


def test_baseline_train_only_and_reserved_test_guard():
    from scripts.run_week5_fresh_baseline import baseline_rows
    data = baseline_dataset()
    assert len(baseline_rows(data)) == 3
    data["rows"][0]["reserved_test"] = True
    with pytest.raises(ValueError, match="RESERVED_TEST"):
        baseline_rows(data)


@pytest.mark.parametrize("field,value,error", [
    ("split_locked", False, "INDEPENDENT"),
    ("leakage_review_complete", False, "INDEPENDENT"),
    ("status", "BLOCKED", "INDEPENDENT"),
    ("dfire_annotation_origin", "LEGACY", "ORIGINAL"),
    ("class_mapping", ["fire", "smoke"], "MAPPING"),
])
def test_baseline_gate_rejects_legacy_alias_or_unlocked_data(field, value, error):
    from scripts.run_week5_fresh_baseline import baseline_rows
    data = baseline_dataset()
    data[field] = value
    with pytest.raises(ValueError, match=error):
        baseline_rows(data)


def test_baseline_same_scene_across_splits_rejected():
    from scripts.run_week5_fresh_baseline import baseline_rows
    data = baseline_dataset()
    data["rows"][1]["scene_group_id"] = data["rows"][0]["scene_group_id"]
    with pytest.raises(ValueError, match="LEAKAGE"):
        baseline_rows(data)


def test_device_cpu_fallback_and_single_gpu():
    from src.week5_comparison import training_device
    assert training_device(False, True) == "cpu"
    assert training_device(True, False) == "0"
    with pytest.raises(ValueError, match="CUDA_REQUIRED"):
        training_device(False, False)


def test_comparison_test_scope_rejected_even_for_complete_receipts():
    receipts, protocol = matrix()
    protocol["evaluation"]["split_for_selection"] = "test"
    with pytest.raises(ValueError, match="TEST_RESULTS"):
        validate_comparability(receipts, protocol, "c"*64, True)


def test_final_acceptance_mt4_classes_and_video_gates():
    from src.week5_comparison import select_final_model
    candidate = dict(scope="validation", status="MEASURED", vn_map50_95=.6, dfire_map50_95=.6,
                     video_recall=.9, event_recall=.9, far_per_hour=.1, delay_sec=2,
                     baseline_video_recall=.9, baseline_event_recall=.9, baseline_far_per_hour=.2,
                     checkpoint_sha256="a"*64, runtime_feasible=True,
                     delta_vn_map50=.06, delta_dfire_map50=-.01,
                     class_support={c: {"SMOKE": 3, "FIRE": 2} for c in ("vn", "dfire")})
    assert select_final_model([candidate])["selected"] == candidate
    for changes in ({"delta_vn_map50": .049}, {"delta_dfire_map50": -.021},
                    {"class_support": {}}, {"event_recall": None}):
        assert select_final_model([{**candidate, **changes}])["status"] == "NO_ACCEPTED_WINNER"
    with pytest.raises(ValueError, match="TEST_RESULTS"):
        select_final_model([{**candidate, "scope": "test"}])
