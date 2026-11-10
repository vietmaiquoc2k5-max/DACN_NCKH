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

pending_events = [e for e in events if e.get("verification_status") != "verified"]

print("=== CHECKING PENDING EVENTS WITH ACTUAL VIDEO FILES ===")
results = []
for e in pending_events:
    v_id = e["video_id"]
    meta = video_meta.get(v_id, {})
    file_name = meta.get("file_name", "")
    video_path = Path("FireSmokeLabeler/videos") / file_name
    
    if not video_path.exists():
        print(f"ERROR: Video {video_path} not found!")
        continue
        
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_duration = round(frames / fps, 2) if fps > 0 else 0
    cap.release()
    
    # Check start_sec vs actual_duration
    start_sec = e["start_sec"]
    end_sec = e["end_sec"]
    
    # Verify if end_sec matches actual_duration or needs correction
    is_valid_time = 0 <= start_sec < end_sec <= (actual_duration + 0.1)
    
    results.append({
        "video_id": v_id,
        "label": e["label"],
        "start_sec": start_sec,
        "old_end_sec": end_sec,
        "actual_duration": actual_duration,
        "fps": fps,
        "frames": frames,
        "is_valid_time": is_valid_time
    })
    print(f"{v_id:8s} | {e['label']:6s} | Start: {start_sec:5.2f}s | End: {end_sec:5.2f}s | Actual Dur: {actual_duration:5.2f}s | Frames: {frames:4d} | FPS: {fps:4.1f} | Valid: {is_valid_time}")


