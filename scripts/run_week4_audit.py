"""Kiểm toán toàn diện kho dữ liệu Video và Hình ảnh thực tế phục vụ Tuần 4.
Được gọi trực tiếp bởi agent để sinh các file dữ liệu nghiệm thu:
- runs/benchmarks/week4_final/data_audit.json
- runs/benchmarks/week4_final/data_audit.csv
- docs/WEEK4_DATA_AUDIT.md
- docs/WEEK4_INITIAL_AUDIT.md
"""
import os
import sys
import csv
import json
import hashlib
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "video_dataset_list.csv"
OUT_DIR = ROOT / "runs" / "benchmarks" / "week4_final"
DOCS_DIR = ROOT / "docs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

def audit_videos():
    audit_records = []
    stats = {
        "total_in_catalog": 0,
        "valid_count": 0,
        "missing_count": 0,
        "corrupted_count": 0,
        "pos_valid_count": 0,
        "pos_valid_duration_sec": 0.0,
        "neg_valid_count": 0,
        "neg_valid_duration_sec": 0.0,
        "missing_pos_count": 0,
        "missing_neg_count": 0,
    }

    print("Auditing 192 videos in catalog...")
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total_in_catalog"] += 1
            vid = row["video_id"].strip()
            rel_path = row.get("relative_path", "").strip()
            cat = row.get("category", "").strip().lower()
            is_pos = ("neg" not in cat) and (not vid.upper().startswith("NEG")) and ("_NEG_" not in vid.upper())
            dur_cat = float(row.get("duration_sec") or 0.0)

            full_path = ROOT / rel_path
            rec = {
                "video_id": vid,
                "file_name": row.get("file_name", ""),
                "relative_path": rel_path,
                "category": "positive" if is_pos else "negative",
                "catalog_duration_sec": dur_cat,
                "file_exists": full_path.exists(),
                "file_size_bytes": 0,
                "status": "UNKNOWN",
                "width": 0,
                "height": 0,
                "fps": 0.0,
                "total_frames": 0,
                "decoded_frames_sampled": 0,
                "real_duration_sec": 0.0,
                "sha256_prefix": "",
                "error_msg": "",
                "disturbance_scene": row.get("notes", ""),
                "source_url": row.get("source_url", ""),
            }

            if not full_path.exists():
                rec["status"] = "MISSING"
                stats["missing_count"] += 1
                if is_pos:
                    stats["missing_pos_count"] += 1
                else:
                    stats["missing_neg_count"] += 1
            else:
                rec["file_size_bytes"] = full_path.stat().st_size
                try:
                    h = hashlib.sha256()
                    with open(full_path, "rb") as vf:
                        chunk = vf.read(1024 * 1024)
                        h.update(chunk)
                    rec["sha256_prefix"] = h.hexdigest()[:16]
                except Exception as e:
                    rec["sha256_prefix"] = "error"

                # OpenCV verification
                cap = cv2.VideoCapture(str(full_path))
                if not cap.isOpened():
                    rec["status"] = "CORRUPTED"
                    rec["error_msg"] = "OpenCV could not open video stream"
                    stats["corrupted_count"] += 1
                else:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 2)
                    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    rec["width"] = w
                    rec["height"] = h
                    rec["fps"] = fps
                    rec["total_frames"] = tot_frames

                    if fps > 0 and tot_frames > 0:
                        rec["real_duration_sec"] = round(tot_frames / fps, 2)
                    else:
                        rec["real_duration_sec"] = dur_cat

                    frames_read = 0
                    for _ in range(5):
                        ret, fr = cap.read()
                        if ret and fr is not None:
                            frames_read += 1
                        else:
                            break
                    rec["decoded_frames_sampled"] = frames_read
                    cap.release()

                    if frames_read > 0:
                        rec["status"] = "VALID"
                        stats["valid_count"] += 1
                        if is_pos:
                            stats["pos_valid_count"] += 1
                            stats["pos_valid_duration_sec"] += rec["real_duration_sec"]
                        else:
                            stats["neg_valid_count"] += 1
                            stats["neg_valid_duration_sec"] += rec["real_duration_sec"]
                    else:
                        rec["status"] = "CORRUPTED"
                        rec["error_msg"] = "Could not decode initial video frames"
                        stats["corrupted_count"] += 1

            audit_records.append(rec)

    # Save data_audit.json
    with open(OUT_DIR / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump({"summary": stats, "records": audit_records}, f, indent=2, ensure_ascii=False)

    # Save data_audit.csv
    fieldnames = [
        "video_id", "file_name", "relative_path", "category", "status",
        "file_exists", "file_size_bytes", "width", "height", "fps",
        "total_frames", "real_duration_sec", "sha256_prefix",
        "disturbance_scene", "source_url", "error_msg"
    ]
    with open(OUT_DIR / "data_audit.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(audit_records)

    return stats, audit_records

def audit_images():
    vn_base = ROOT / "datasets" / "vn_dataset"
    splits = ["train", "val", "test"]
    vn_stats = {"total_images": 0, "total_labels": 0, "by_split": {}}
    video_prefixes = {"train": set(), "val": set(), "test": set()}

    for sp in splits:
        img_p = vn_base / "images" / sp
        lbl_p = vn_base / "labels" / sp
        imgs = list(img_p.glob("*.*"))
        lbls = list(lbl_p.glob("*.txt"))
        vn_stats["total_images"] += len(imgs)
        vn_stats["total_labels"] += len(lbls)
        vn_stats["by_split"][sp] = {
            "images": len(imgs),
            "labels": len(lbls),
        }
        for img_file in imgs:
            name = img_file.stem
            # extract prefix, e.g. VN_POS_01, VN_NEG_06
            parts = name.split("_")
            if len(parts) >= 3:
                prefix = f"{parts[0]}_{parts[1]}_{parts[2]}"
                video_prefixes[sp].add(prefix)

    leakage_train_val = video_prefixes["train"].intersection(video_prefixes["val"])
    leakage_train_test = video_prefixes["train"].intersection(video_prefixes["test"])
    leakage_val_test = video_prefixes["val"].intersection(video_prefixes["test"])

    vn_stats["video_leakage"] = {
        "train_val_overlap": list(leakage_train_val),
        "train_test_overlap": list(leakage_train_test),
        "val_test_overlap": list(leakage_val_test),
        "is_leakage_free": (len(leakage_train_val) == 0 and len(leakage_train_test) == 0 and len(leakage_val_test) == 0),
    }

    # D-Fire stats
    dfire_base = ROOT / "datasets" / "dfire"
    dfire_stats = {"total_images": 0, "total_labels": 0, "by_split": {}}
    for sp in ["train", "val", "test"]:
        df_img_p = dfire_base / sp / "images"
        df_lbl_p = dfire_base / sp / "labels"
        df_imgs = list(df_img_p.glob("*.*")) if df_img_p.exists() else []
        df_lbls = list(df_lbl_p.glob("*.txt")) if df_lbl_p.exists() else []
        dfire_stats["total_images"] += len(df_imgs)
        dfire_stats["total_labels"] += len(df_lbls)
        dfire_stats["by_split"][sp] = {
            "images": len(df_imgs),
            "labels": len(df_lbls),
        }

    return vn_stats, dfire_stats

if __name__ == "__main__":
    v_stats, records = audit_videos()
    vn_stats, df_stats = audit_images()

    print("\n--- AUDIT SUMMARY ---")
    print(f"Total Videos in Catalog: {v_stats['total_in_catalog']}")
    print(f"Valid Videos on Disk: {v_stats['valid_count']} (POS: {v_stats['pos_valid_count']}, NEG: {v_stats['neg_valid_count']})")
    print(f"Valid POS Duration: {v_stats['pos_valid_duration_sec']:.1f}s ({v_stats['pos_valid_duration_sec']/3600.0:.2f}h)")
    print(f"Valid NEG Duration: {v_stats['neg_valid_duration_sec']:.1f}s ({v_stats['neg_valid_duration_sec']/3600.0:.2f}h)")
    print(f"Missing Videos: {v_stats['missing_count']} (POS: {v_stats['missing_pos_count']}, NEG: {v_stats['missing_neg_count']})")
    print(f"Corrupted Videos: {v_stats['corrupted_count']}")
    print(f"VN Dataset: {vn_stats['total_images']} images, {vn_stats['total_labels']} labels")
    print(f"VN Splits: Train {vn_stats['by_split']['train']['images']}, Val {vn_stats['by_split']['val']['images']}, Test {vn_stats['by_split']['test']['images']}")
    print(f"VN Video Leakage Free: {vn_stats['video_leakage']['is_leakage_free']}")
    print(f"D-Fire Dataset: {df_stats['total_images']} images, {df_stats['total_labels']} labels")
