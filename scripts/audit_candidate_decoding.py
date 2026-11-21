#!/usr/bin/env python3
"""Audit script for sequential full-stream decoding of 12 candidate negative videos and VN_NEG_48."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent

CANDIDATES = [
    "VN_NEG_03", "VN_NEG_42", "VN_NEG_43", "VN_NEG_44", "VN_NEG_45",
    "VN_NEG_46", "VN_NEG_47", "VN_NEG_48", "VN_NEG_49", "VN_NEG_50",
    "VN_NEG_51", "VN_NEG_52", "VN_NEG_53"
]

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()

def audit_video(vid: str) -> dict:
    path = ROOT / "datasets" / "negative_videos" / f"{vid}.mp4"
    if not path.exists():
        return {
            "video_id": vid, "exists": False, "verdict": "FAIL_FILE_NOT_FOUND"
        }
    
    file_size = path.stat().st_size
    t_start = time.time()
    sha256 = compute_sha256(path)
    
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "video_id": vid, "exists": True, "file_size_bytes": file_size,
            "sha256": sha256, "verdict": "FAIL_CANNOT_OPEN"
        }
    
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    declared_fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps_clean = round(fps, 2) if fps > 0.0 else 25.0
    declared_dur = round(declared_fc / fps_clean, 2) if (declared_fc > 0 and fps_clean > 0) else 0.0
    
    decoded_fc = 0
    errors = []
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame.size == 0:
            errors.append(f"Empty decoded frame at index {decoded_fc}")
            break
        decoded_fc += 1
        
    actual_dur = round(decoded_fc / fps_clean, 2) if fps_clean > 0 else 0.0
    diff = declared_fc - decoded_fc
    diff_pct = (diff / declared_fc) * 100.0 if declared_fc > 0 else 0.0
    
    # Check tail reachability if stopped early
    can_read_tail = False
    if decoded_fc < declared_fc:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, declared_fc - 1))
        ret_tail, frame_tail = cap.read()
        if ret_tail and frame_tail is not None and frame_tail.size > 0:
            can_read_tail = True
            errors.append(f"Mid-stream corruption: stopped at {decoded_fc}/{declared_fc}, tail {declared_fc-1} reachable")
            
    cap.release()
    t_elapsed = round(time.time() - t_start, 2)
    
    if can_read_tail:
        eof_status = "DECODE_ERROR"
        tech_verdict = f"FAIL_MID_STREAM_CORRUPTION_AT_FRAME_{decoded_fc}"
    elif decoded_fc == 0:
        eof_status = "DECODE_ERROR"
        tech_verdict = "FAIL_ZERO_FRAMES_DECODED"
    elif diff < 0:
        eof_status = "NORMAL_EOF"
        tech_verdict = "PASS_NORMAL_EOF_EXCEEDS_HEADER"
    elif diff == 0:
        eof_status = "NORMAL_EOF"
        tech_verdict = "PASS_NORMAL_EOF"
    else:
        is_genuine_padding = (
            len(errors) == 0 and diff <= 2 and abs(declared_dur - actual_dur) <= (2.0 / fps_clean)
        )
        if is_genuine_padding:
            eof_status = "NORMAL_EOF"
            tech_verdict = "PASS_NORMAL_EOF_CONTAINER_PADDING"
        elif diff <= 10 or diff_pct <= 0.5:
            eof_status = "UNKNOWN_EOF"
            tech_verdict = f"FAIL_UNKNOWN_EOF_MISSING_{diff}_FRAMES"
        else:
            eof_status = "TRUNCATED_STREAM"
            tech_verdict = f"FAIL_TRUNCATED_STREAM_MISSING_{diff}_FRAMES"
            
    res = {
        "video_id": vid,
        "filename": f"{vid}.mp4",
        "file_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "file_size_bytes": file_size,
        "sha256": sha256,
        "fps": fps_clean,
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
        "declared_frame_count": declared_fc,
        "decoded_frame_count": decoded_fc,
        "frame_diff": diff,
        "frame_diff_pct": round(diff_pct, 4),
        "declared_duration_sec": declared_dur,
        "decoded_duration_sec": actual_dur,
        "duration_diff_sec": round(abs(declared_dur - actual_dur), 4),
        "eof_status": eof_status,
        "technical_verdict": tech_verdict,
        "errors": errors,
        "elapsed_seconds": t_elapsed
    }
    print(f"[{vid}] {tech_verdict} | Decl: {declared_fc} | Decoded: {decoded_fc} | Elapsed: {t_elapsed}s", flush=True)
    return res

def main():
    print(f"Starting sequential full-stream decode audit for {len(CANDIDATES)} videos...", flush=True)
    results = []
    for vid in CANDIDATES:
        res = audit_video(vid)
        results.append(res)
        
    out_path = ROOT / "runs" / "audits" / "week6" / "full_stream_decode_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nAudit complete. Results saved to {out_path}", flush=True)

if __name__ == "__main__":
    main()
