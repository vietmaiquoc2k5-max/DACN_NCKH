import cv2
from pathlib import Path

dl = Path("C:/Users/thien/Downloads")
videos = sorted(dl.glob("*.mp4"))

print(f"Found {len(videos)} videos in Downloads:")
total_noise_sec = 0.0
total_pos_sec = 0.0

for v in videos:
    cap = cv2.VideoCapture(str(v))
    if not cap.isOpened():
        print(f"Cannot open {v.name}")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur_sec = frames / fps if fps > 0 else 0
    cap.release()
    
    is_noise = "NHIỄU" in v.name.upper()
    tag = "NEGATIVE/NOISE" if is_noise else "POSITIVE"
    if is_noise:
        total_noise_sec += dur_sec
    else:
        total_pos_sec += dur_sec
        
    print(f"{v.name:25s} | {width:4d}x{height:4d} | {fps:5.2f} fps | {frames:6d} frames | {dur_sec:8.2f}s ({dur_sec/60:6.2f}m) | {tag}")

print("="*75)
print(f"Total Negative/Noise Duration: {total_noise_sec:.2f} s = {total_noise_sec/3600:.4f} hours")
print(f"Total Positive Duration:       {total_pos_sec:.2f} s = {total_pos_sec/3600:.4f} hours")
print(f"Grand Total:                   {total_noise_sec + total_pos_sec:.2f} s = {(total_noise_sec + total_pos_sec)/3600:.4f} hours")

