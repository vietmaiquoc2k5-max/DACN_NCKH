import json
import csv
from pathlib import Path
import cv2

events_path = Path("data/ground_truth/events.jsonl")
with open(events_path, "r", encoding="utf-8") as f:
    events = [json.loads(line) for line in f if line.strip()]

csv_path = Path("data/video_dataset_list.csv")
video_meta = {}
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        video_meta[r["video_id"]] = r

print(f"Total events in JSONL: {len(events)}")
verified_events = [e for e in events if e.get("verification_status") == "verified"]
pending_events = [e for e in events if e.get("verification_status") != "verified"]
print(f"Verified: {len(verified_events)}, Pending: {len(pending_events)}")

# Check video files for pending events
for e in pending_events:
    v_id = e["video_id"]
    meta = video_meta.get(v_id, {})
    rel_path = meta.get("relative_path", "")
    file_name = meta.get("file_name", "")
    
    # Check potential locations
    p1 = Path(rel_path)
    p2 = Path("FireSmokeLabeler/videos") / file_name
    p3 = Path("FireSmokeLabeler/videos") / f"{v_id}.mp4"
    
    actual_path = None
    if p1.exists():
        actual_path = p1
    elif p2.exists():
        actual_path = p2
    elif p3.exists():
        actual_path = p3
        
    e["actual_video_path"] = str(actual_path) if actual_path else None
    print(f"Event: {v_id:8s} | {e['label']:6s} | {e['start_sec']:5.1f}s - {e['end_sec']:5.1f}s | File: {actual_path}")
