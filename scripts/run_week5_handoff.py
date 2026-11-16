"""Single entry point for Week 5 GPU Domain Adaptation training and evaluation.

Designed for seamless reproducibility on recipient GPU machines.
Validates all inputs, verifies CUDA, executes smoke tests, trains strategies
and runs comparisons cleanly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("Week5Handoff")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_cuda_and_gpu(requested_device: str) -> str:
    """Detect and validate CUDA device, VRAM and print actionable guidance."""
    import torch
    if requested_device.lower() == "cpu":
        LOG.warning("Running on CPU as requested by user. GPU training recommended for speed.")
        return "cpu"

    if not torch.cuda.is_available():
        LOG.error("CUDA is not available in current PyTorch installation!")
        LOG.error("Please install CUDA-enabled PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        LOG.error("Or run with --device cpu for debugging.")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    device_id = int(requested_device) if requested_device.isdigit() else 0
    if device_id >= device_count:
        LOG.error(f"Requested device {requested_device} exceeds available GPUs ({device_count})")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(device_id)
    vram_bytes = torch.cuda.get_device_properties(device_id).total_memory
    vram_gb = vram_bytes / (1024 ** 3)
    LOG.info(f"Target GPU [{device_id}]: {gpu_name} ({vram_gb:.2f} GB VRAM)")

    if vram_gb < 4.0:
        LOG.warning("VRAM < 4GB. Consider reducing batch size (--batch 4) to prevent CUDA Out Of Memory.")
    return str(device_id)


def run_gpu_smoke_test(device: str) -> None:
    """Run quick forward/backward pass on GPU to ensure CUDA acceleration works."""
    LOG.info("Running GPU smoke test...")
    import torch
    from ultralytics import YOLO

    init_path = ROOT / "yolov8n.pt"
    if not init_path.exists():
        LOG.error(f"Initialization model {init_path} missing!")
        sys.exit(1)

    dev = "cpu" if device == "cpu" else f"cuda:{device}"
    model = YOLO(str(init_path)).to(dev)
    dummy_x = torch.randn(2, 3, 640, 640, device=dev)
    t0 = time.monotonic()
    out = model(dummy_x, verbose=False)
    t1 = time.monotonic()
    LOG.info(f"GPU smoke test PASSED! Forward pass duration: {(t1 - t0)*1000:.1f} ms on {dev}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["all", "baseline", "full_finetune", "freeze_backbone", "replay_20_percent"],
                        help="Strategy to train. Use 'all' to train A, B, and C sequentially.")
    parser.add_argument("--device", default="0", help="Training device: '0', '1', or 'cpu'. Default: 0")
    parser.add_argument("--batch", type=int, default=8, help="Batch size. Default: 8")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs. Default: 30")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument("--manifest", type=Path, default=ROOT / "handoff/week5/dataset_manifest.json",
                        help="Path to portable dataset manifest.")
    parser.add_argument("--protocol", type=Path, default=ROOT / "handoff/week5/protocol.json",
                        help="Path to training protocol.")
    parser.add_argument("--budget", type=Path, default=None,
                        help="Path to budget receipt. Defaults to budget_receipt_gpu.json (or cpu if --device cpu).")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/benchmarks/week5_domain_adaptation/gpu_training",
                        help="Base output directory for runs.")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick GPU smoke test and exit.")
    parser.add_argument("--compare", action="store_true", help="Run comparison collector across completed runs.")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted run from last checkpoint.")
    args = parser.parse_args()

    actual_device = check_cuda_and_gpu(args.device)

    if args.smoke_test:
        run_gpu_smoke_test(actual_device)
        LOG.info("Smoke test finished successfully. Ready for full training.")
        return

    # Verify input manifest & protocol
    if not args.manifest.exists():
        LOG.error(f"Dataset manifest not found: {args.manifest}")
        sys.exit(1)
    if not args.protocol.exists():
        LOG.error(f"Protocol not found: {args.protocol}")
        sys.exit(1)

    budget_path = args.budget
    if budget_path is None:
        budget_path = ROOT / ("handoff/week5/budget_receipt_cpu.json" if actual_device == "cpu" else "handoff/week5/budget_receipt_gpu.json")
    if not budget_path.exists():
        LOG.error(f"Budget receipt not found: {budget_path}")
        sys.exit(1)

    # 1. Run Baseline training
    if args.strategy == "baseline":
        LOG.info("Starting fresh D-Fire baseline training...")
        baseline_out = args.output_dir / "baseline"
        cmd = [
            sys.executable, str(ROOT / "scripts/run_week5_fresh_baseline.py"),
            "--dataset-manifest", str(args.manifest),
            "--protocol", str(ROOT / "runs/benchmarks/week5_domain_adaptation/final_bbox_20261116_57d1b826/new_baseline_protocol.json"),
            "--budget-receipt", str(budget_path),
            "--seed", str(args.seed),
            "--output-dir", str(baseline_out)
        ]
        LOG.info(f"Command: {' '.join(cmd)}")
        res = subprocess.run(cmd, cwd=ROOT)
        sys.exit(res.returncode)

    # 2. Run Adaptation Strategies
    strategies_to_run = ["full_finetune", "freeze_backbone", "replay_20_percent"] if args.strategy == "all" else [args.strategy] if args.strategy else []

    if strategies_to_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for strat in strategies_to_run:
            run_out = args.output_dir / f"run_{strat}_seed{args.seed}"
            LOG.info(f"\n========================================================")
            LOG.info(f" EXECUTING STRATEGY: {strat.upper()} (Seed {args.seed})")
            LOG.info(f" Output: {run_out}")
            LOG.info(f"========================================================")
            cmd = [
                sys.executable, str(ROOT / "scripts/run_week5_domain_adaptation.py"),
                "--dataset-manifest", str(args.manifest),
                "--protocol", str(args.protocol),
                "--budget-receipt", str(budget_path),
                "--strategy", strat,
                "--seed", str(args.seed),
                "--output-dir", str(run_out)
            ]
            if args.resume:
                cmd.append("--resume")
            LOG.info(f"Running command: {' '.join(cmd)}")
            res = subprocess.run(cmd, cwd=ROOT)
            if res.returncode != 0:
                LOG.error(f"Strategy {strat} failed with exit code {res.returncode}")
                sys.exit(res.returncode)
            LOG.info(f"Strategy {strat} completed successfully!")

    # 3. Run Comparison
    if args.compare or args.strategy == "all":
        LOG.info("\n========================================================")
        LOG.info(" RUNNING COMPARISON COLLECTOR")
        LOG.info("========================================================")
        receipts = []
        for strat in ["full_finetune", "freeze_backbone", "replay_20_percent"]:
            rpath = args.output_dir / f"run_{strat}_seed{args.seed}" / "training_receipt.json"
            if rpath.exists():
                receipts.append(str(rpath))
            else:
                LOG.warning(f"Receipt missing for {strat}: {rpath}")

        baseline_receipt = ROOT / "runs/benchmarks/week5_domain_adaptation/execution_20261116/baseline/baseline_receipt.json"
        if not baseline_receipt.exists():
            LOG.warning(f"Default baseline receipt not found at {baseline_receipt}")

        if len(receipts) == 3 and baseline_receipt.exists():
            cmp_out = args.output_dir / "comparison"
            cmd = [
                sys.executable, str(ROOT / "scripts/compare_week5_runs.py"),
                "--dataset-manifest", str(args.manifest),
                "--protocol", str(args.protocol),
                "--baseline-receipt", str(baseline_receipt),
                "--receipts"
            ] + receipts + ["--output-dir", str(cmp_out)]
            LOG.info(f"Running compare command: {' '.join(cmd)}")
            res = subprocess.run(cmd, cwd=ROOT)
            if res.returncode == 0:
                LOG.info(f"Comparison completed successfully! Results in: {cmp_out}")
            else:
                LOG.error(f"Comparison failed with exit code {res.returncode}")
        else:
            LOG.info(f"Found {len(receipts)}/3 strategy receipts. Comparison requires all 3 completed runs.")


if __name__ == "__main__":
    main()

