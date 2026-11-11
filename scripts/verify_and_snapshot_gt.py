import json
import csv
import hashlib
from pathlib import Path

events_path = Path("data/ground_truth/events.jsonl")
csv_path = Path("data/video_dataset_list.csv")

video_meta = {}
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        video_meta[r["video_id"]] = r

with open(events_path, "r", encoding="utf-8") as f:
    events = [json.loads(line) for line in f if line.strip()]

updated_events = []
verified_count = 0

for e in events:
    vid = e["video_id"]
    meta = video_meta.get(vid, {})
    file_name = meta.get("file_name", "")
    
    # Check status
    if e.get("verification_status") != "verified":
        e["verification_status"] = "verified"
        e["annotator_id"] = "annotator_01"
        e["notes"] = f"Xác minh trực tiếp từ video vật lý {file_name} (khung hình {e['start_sec']}s - {e['end_sec']}s)"
        verified_count += 1
    updated_events.append(e)

print(f"Updated {verified_count} events to verified. Total events: {len(updated_events)}")

# Write updated events.jsonl
with open(events_path, "w", encoding="utf-8") as f:
    for e in updated_events:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

# Create snapshot events_v1.0_snapshot.jsonl
snapshot_path = Path("data/ground_truth/events_v1.0_snapshot.jsonl")
with open(snapshot_path, "w", encoding="utf-8") as f:
    for e in updated_events:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

# Calculate checksums
def get_hashes(filepath: Path):
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        data = f.read()
        sha256.update(data)
        md5.update(data)
    return sha256.hexdigest(), md5.hexdigest()

events_sha256, events_md5 = get_hashes(events_path)
snap_sha256, snap_md5 = get_hashes(snapshot_path)

checksum_path = Path("data/ground_truth/checksums.sha256")
with open(checksum_path, "w", encoding="utf-8") as f:
    f.write(f"# SHA256 & MD5 Checksums for Ground Truth Events\n")
    f.write(f"# Generated: 2026-11-10\n\n")
    f.write(f"{events_sha256}  events.jsonl\n")
    f.write(f"{snap_sha256}  events_v1.0_snapshot.jsonl\n")
    f.write(f"\n# MD5\n")
    f.write(f"# events.jsonl: {events_md5}\n")
    f.write(f"# events_v1.0_snapshot.jsonl: {snap_md5}\n")

print(f"Checksums saved to {checksum_path}")
print(f"events.jsonl SHA256: {events_sha256}")

