import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.benchmark_week4_bytetrack import load_dataset_catalog

catalog = load_dataset_catalog(Path("data/video_dataset_list.csv"))
print(f"Available positive: {len(catalog['available_positive'])}")
print(f"Missing positive:   {len(catalog['missing_positive'])}")
print(f"Available negative: {len(catalog['available_negative'])}")
print(f"Missing negative:   {len(catalog['missing_negative'])}")

total_neg_sec = sum(v["duration_sec"] for v in catalog["available_negative"])
print(f"Total available negative duration: {total_neg_sec:.2f}s = {total_neg_sec/3600:.4f}h")
for v in catalog["available_negative"]:
    print(f"  {v['video_id']} ({v['file_name']}) -> Exists: {v['exists_on_disk']}, Dur: {v['duration_sec']}s")
