import cv2
import json
import csv
from pathlib import Path

neg_dir = Path("datasets/negative_videos")
videos = [
    {"id": "REAL_NEG_01", "file": "NHIỄU 1.mp4", "split": "val", "notes": "Video cảnh nhiễu thực tế (đèn đường, xe cộ, ánh sáng thay đổi) - Tập Val"},
    {"id": "REAL_NEG_02", "file": "NHIỄU 2.mp4", "split": "val", "notes": "Video cảnh nhiễu chuyển động và đèn xe ban đêm - Tập Val"},
    {"id": "REAL_NEG_03", "file": "NHIỄU 3.mp4", "split": "val", "notes": "Video cảnh nhiễu ánh nắng phản chiếu bề mặt - Tập Val"},
    {"id": "REAL_NEG_04", "file": "NHIỄU 4.mp4", "split": "test", "notes": "Video cảnh nhiễu ánh sáng ban ngày ngoài trời - Tập Test"},
    {"id": "REAL_NEG_05", "file": "NHIỄU 5.mp4", "split": "test", "notes": "Video cảnh nhiễu camera giám sát thời lượng dài - Tập Test"}
]

manifest = {
    "version": "1.0",
    "description": "Danh mục video âm tính thực tế trên ổ đĩa phục vụ đánh giá FAR",
    "target_hours": 5.0,
    "actual_hours": 0.0,
    "missing_hours": 0.0,
    "status": "PARTIAL / BLOCKED_BY_NEGATIVE_DATA",
    "videos": []
}

total_seconds = 0.0
val_seconds = 0.0
test_seconds = 0.0

for item in videos:
    v_path = neg_dir / item["file"]
    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        print(f"Cannot open {v_path}")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur_sec = round(frames / fps, 2) if fps > 0 else 0.0
    cap.release()
    
    total_seconds += dur_sec
    if item["split"] == "val":
        val_seconds += dur_sec
    else:
        test_seconds += dur_sec
        
    m, s = divmod(dur_sec, 60)
    h, m = divmod(m, 60)
    formatted = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    
    entry = {
        "video_id": item["id"],
        "file_name": item["file"],
        "relative_path": str(v_path).replace("\\", "/"),
        "category": "negative_noise",
        "resolution": f"{width}x{height}",
        "fps": fps,
        "total_frames": frames,
        "duration_sec": dur_sec,
        "duration_formatted": formatted,
        "split": item["split"],
        "fire_start_sec": -1.0,
        "smoke_start_sec": -1.0,
        "notes": item["notes"]
    }
    manifest["videos"].append(entry)

manifest["actual_seconds"] = round(total_seconds, 2)
manifest["actual_hours"] = round(total_seconds / 3600.0, 4)
manifest["missing_hours"] = round(max(0.0, 5.0 - (total_seconds / 3600.0)), 4)
manifest["val_hours"] = round(val_seconds / 3600.0, 4)
manifest["test_hours"] = round(test_seconds / 3600.0, 4)

# Write manifest
manifest_path = Path("data/negative_videos_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"Manifest saved to {manifest_path}")
print(f"Actual Negative Duration: {manifest['actual_seconds']} s = {manifest['actual_hours']} hours")
print(f"Missing to reach 5.0h:   {manifest['missing_hours']} hours")
print(f"Val split: {manifest['val_hours']} h, Test split: {manifest['test_hours']} h")

# Update or sync into data/video_dataset_list.csv
csv_path = Path("data/video_dataset_list.csv")
rows = []
fieldnames = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for r in reader:
        # Keep existing rows if not REAL_NEG
        if not r["video_id"].startswith("REAL_NEG_"):
            rows.append(r)

for v in manifest["videos"]:
    rows.append({
        "video_id": v["video_id"],
        "file_name": v["file_name"],
        "relative_path": v["relative_path"],
        "category": v["category"],
        "resolution": v["resolution"],
        "fps": str(v["fps"]),
        "total_frames": str(v["total_frames"]),
        "duration_sec": str(v["duration_sec"]),
        "duration_formatted": v["duration_formatted"],
        "fire_start_sec": "-1",
        "smoke_start_sec": "-1",
        "notes": f"[{v['split'].upper()}] {v['notes']}",
        "source_url": ""
    })

with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Updated {csv_path} with {len(manifest['videos'])} real negative videos.")

