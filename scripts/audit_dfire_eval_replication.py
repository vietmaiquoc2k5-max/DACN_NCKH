"""Independent evaluation replication of Week 5 checkpoints on D-Fire Validation.
Auditor script to reproduce exact mAP metrics on GPU without caching.
"""
import json
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parent.parent
data_yaml = ROOT / 'runs/benchmarks/week5_domain_adaptation/gpu_training/run_replay_20_percent_seed42/dfire_eval.yaml'

models = {
    'Baseline': ROOT / 'runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt',
    'strategy_a': ROOT / 'runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/checkpoints/strategy_a_best.pt',
    'strategy_b': ROOT / 'runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/checkpoints/strategy_b_best.pt',
    'strategy_c': ROOT / 'runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/checkpoints/strategy_c_best.pt'
}

results = {}
for name, p in models.items():
    print(f'Loading {name} from {p}...')
    m = YOLO(str(p))
    metrics = m.val(
        data=str(data_yaml),
        split='val',
        imgsz=640,
        conf=0.001,
        iou=0.7,
        device='cuda:0',
        workers=0,
        plots=False,
        verbose=False
    )
    box = metrics.box
    ap_idx = list(box.ap_class_index)
    per_class = {}
    for pos, idx in enumerate(ap_idx):
        c_name = {0: 'smoke', 1: 'fire'}[int(idx)]
        per_class[c_name] = {'map50': float(box.ap50[pos]), 'map50_95': float(box.ap[pos])}
    
    results[name] = {
        'all_map50': float(box.map50),
        'all_map50_95': float(box.map),
        'per_class': per_class
    }
    print(f'=== {name} D-Fire Val ===')
    print('  mAP50:', float(box.map50))
    print('  mAP50-95:', float(box.map))
    for c_name, vals in per_class.items():
        print(f"  {c_name}: mAP50={vals['map50']}, mAP50-95={vals['map50_95']}")

out_file = ROOT / 'runs/benchmarks/week5_domain_adaptation/independent_dfire_eval_results.json'
out_file.write_text(json.dumps(results, indent=2), encoding='utf-8')
print(f'Saved independent replication results to {out_file}')
