"""Prepare the final deliverable from sealed evidence, without starting training.

This command intentionally cannot select models from ad-hoc test measurements.
Completed training receipts must first pass the comparison API and the locked
paired-video acceptance procedure before an accepted checkpoint can be exported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.week5_comparison import STRATEGIES, comparison_row, selection_blockers


def digest(path: Path) -> str:
    """Hash immutable input or output bytes."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Refuse to overwrite an existing preparation artifact."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def prepare(evidence: Path, output: Path) -> int:
    """Emit an honest blocked handoff; fail rather than discard completed runs."""
    readiness = json.loads((evidence / "readiness.json").read_text(encoding="utf-8"))
    protocol_path = ROOT / "configs/research/week5_execution_protocol_v2.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    blockers = selection_blockers(readiness)
    if not blockers:
        raise ValueError("DATA_READY_USE_REAL_TRAINING_AND_VALIDATED_RECEIPTS_NOT_EMPTY_PREPARATION")
    for name in ("clean_vn_manifest.json", "clean_dfire_manifest.json"):
        if json.loads((evidence / name).read_text(encoding="utf-8")).get("rows"):
            raise ValueError("NONEMPTY_SUBSET_REQUIRES_REAL_MEMBERSHIP_COLLECTION")
    if digest(ROOT / protocol["checkpoint"]) != protocol["checkpoint_sha256"]:
        raise ValueError("HISTORICAL_CHECKPOINT_HASH_CHANGED")
    baseline_preflight = json.loads((evidence / "new_baseline_protocol.json").read_text(encoding="utf-8"))
    if digest(ROOT / baseline_preflight["initialization_path"]) != baseline_preflight["initialization_sha256"]:
        raise ValueError("INITIALIZATION_HASH_CHANGED")
    output.mkdir(parents=True, exist_ok=False)
    (output / "execution_logs").mkdir()
    config_dir = output / "experiment_configs"
    config_dir.mkdir()
    tracked = ["readiness.json", "review_summary.json", "clean_vn_manifest.json",
               "clean_dfire_manifest.json", "candidate_provenance_manifest.json",
               "baseline_membership.json", "new_baseline_protocol.json"]
    inputs = {name: {"path": str((evidence / name).resolve()), "sha256": digest(evidence / name)} for name in tracked}
    write_json(output / "dataset_membership.json", {
        "status": "BLOCKED_NOT_A_TRAINING_MANIFEST", "parent_artifacts": inputs,
        "class_mapping": {"0": "smoke", "1": "fire"}, "rows": [],
        "eligible_rows": [], "vn_train": 0, "vn_val": 0, "vn_test": 0,
        "dfire_train": 0, "dfire_val": 0, "dfire_test": 0,
        "test_policy": "SEALED; no test evaluation or model selection performed",
    })
    write_json(config_dir / "locked_protocol_v2.json", protocol)
    baseline = json.loads((evidence / "new_baseline_protocol.json").read_text(encoding="utf-8"))
    write_json(config_dir / "fresh_baseline_preparation.json", baseline)
    for strategy in STRATEGIES:
        for seed in protocol["seeds"]:
            write_json(config_dir / f"{strategy}_seed{seed}.json", {
                "status": "BLOCKED_NOT_EXECUTABLE_UNTIL_NEW_BASELINE_AND_READY_DATA",
                "strategy": strategy, "seed": seed, "protocol_sha256": digest(protocol_path),
                "common_baseline_sha256": None, "dataset_sha256": None,
                "epochs": protocol["epochs"], "historical_checkpoint_not_authorized_for_comparison": protocol["checkpoint_sha256"],
            })
    rows = [comparison_row(name) for name in ("Baseline", "A", "B", "C")]
    write_json(output / "comparison.json", {"status": "BLOCKED", "metric_units": "fraction [0,1]; delta is signed; forgetting=max(0,-delta_DFire)",
               "missing_metrics": "null means not measured, never zero", "rows": rows})
    with (output / "comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "model_selection.json", {
        "status": "NO_ACCEPTED_WINNER", "selected": None, "blockers": blockers,
        "acceptance": {"protocol_sha256": digest(protocol_path), "validation_only": True,
                       "required_classes": ["SMOKE", "FIRE"],
                       "proposal_MT4_target_delta_vn_map50": 0.05,
                       "proposal_MT4_max_dfire_map50_drop": 0.02,
                       "locked_v2_paired_video_and_runtime_gates_preserved": True},
        "protocol_v3_activated": False, "test_exposed": False,
    })
    write_json(output / "checkpoint_sha256.json", {
        "exported": {}, "reason": "No new baseline or adaptation completed; no accepted winner",
        "historical_checkpoint": {"path": protocol["checkpoint"], "expected_sha256": protocol["checkpoint_sha256"],
                                  "actual_sha256": digest(ROOT / protocol["checkpoint"]),
                                  "membership_status": "NOT_VERIFIED", "accepted": False},
        "proposed_initialization": {"path": baseline["initialization_path"],
                                    "actual_sha256": digest(ROOT / baseline["initialization_path"]),
                                    "expected_sha256": baseline["initialization_sha256"]},
    })
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for filename, title in (("delta_map_chart.png", "VN / D-Fire delta mAP"), ("forgetting_chart.png", "D-Fire forgetting")):
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.set_title(title)
        axis.axis("off")
        axis.text(.5, .55, "NOT MEASURED — DATA READINESS BLOCKED", ha="center", va="center")
        axis.text(.5, .3, "Baseline / A / B / C: no scores, no ranking", ha="center", va="center")
        figure.savefig(output / filename, dpi=150, bbox_inches="tight")
        plt.close(figure)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (output / "WEEK5_FINAL_REPORT.md").write_text(
        f"# Tuần 5 — PARTIAL / BLOCKED\n\nHEAD: `{head}`.\n\n"
        "Không có baseline mới hoặc A/B/C hoàn tất; 0 run training mới, 0 epoch mới. "
        "Bảng so sánh để trống metric; hình biểu diễn trạng thái chưa đo, không phải kết quả mAP. "
        "NO_ACCEPTED_WINNER; không xuất checkpoint giả.\n\n"
        f"Gate chưa đạt: {', '.join(blockers)}. VN và D-Fire eligible train/val/test: 0/0/0.\n\n"
        "Chuẩn bị baseline mới từ D-Fire TRAIN sạch, khóa validation và membership SHA; "
        "sau pilot đo thật, tạo protocol kế thừa trỏ checkpoint baseline mới rồi chạy A/B/C "
        "cùng checkpoint, seed và budget. Protocol v3 vẫn PROPOSED_NOT_ACTIVE. "
        "Không dùng checkpoint cũ có membership chưa xác minh cho so sánh chính thức.\n\n"
        "Tiêu chí MT4: VN +5 điểm mAP50, D-Fire giảm tối đa 2 điểm; "
        "đồng thời giữ các gate video/runtime của protocol v2. "
        "Mọi lựa chọn trên validation; test tiếp tục niêm phong.\n\n"
        "Xem execution_logs/tests.json và acceptance_tests.md để biết tests, exit codes và SHA log thực tế.\n",
        encoding="utf-8")
    write_json(output / "artifact_sha256.json", {p.relative_to(output).as_posix(): digest(p)
               for p in sorted(output.rglob("*")) if p.is_file()})
    return 2


def main() -> None:
    """Command-line entry point; exit 2 denotes evidenced data blockage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(prepare(args.evidence, args.output_dir))


if __name__ == "__main__":
    main()
