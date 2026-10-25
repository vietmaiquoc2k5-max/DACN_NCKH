import os
import sys
import cv2
import pandas as pd
from pathlib import Path

# Đảm bảo in UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

CSV_PATH = Path("data/video_dataset_list.csv")
NEG_DIR = Path("datasets/test_videos/neg")

def scan_negative_dataset():
    NEG_DIR.mkdir(parents=True, exist_ok=True)
    
    video_exts = [".avi", ".mp4", ".mov", ".mkv", ".webm"]
    files = [f for f in NEG_DIR.iterdir() if f.is_file() and f.suffix.lower() in video_exts]

    print("\n" + "="*70)
    print("🎥 BÁO CÁO TIẾN ĐỘ TẬP VIDEO ÂM TÍNH (NEGATIVE DATASET - MT1)")
    print("="*70)
    print(f"• Thư mục lưu trữ: {NEG_DIR.resolve()}")
    print(f"• Số lượng file video hiện có: {len(files)} video\n")

    total_sec = 0.0
    records = []

    for vf in sorted(files, key=lambda x: x.name):
        cap = cv2.VideoCapture(str(vf))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fc = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        dur = fc / fps if fps > 0 else 0
        total_sec += dur

        mins, secs = divmod(int(dur), 60)
        hrs, mins = divmod(mins, 60)
        dur_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"

        records.append({
            "name": vf.name,
            "resolution": f"{w}x{h}",
            "fps": round(fps, 1),
            "duration": dur_str,
            "duration_sec": round(dur, 1)
        })

    # In ra 10 file đầu tiên làm mẫu
    print(f"{'Tên file':<30} | {'Độ phân giải':<14} | {'FPS':<6} | {'Thời lượng'}")
    print("-" * 65)
    for r in records[:10]:
        print(f"{r['name']:<30} | {r['resolution']:<14} | {r['fps']:<6} | {r['duration']}")
    if len(records) > 10:
        print(f"... và {len(records) - 10} video khác.")

    total_hours = total_sec / 3600.0
    target_hours = 5.0
    percent = min(100.0, (total_hours / target_hours) * 100)

    print("\n" + "="*70)
    print("🎯 TIẾN ĐỘ ĐỐI CHIẾU CHỈ TIÊU ĐỀ CƯƠNG (MT1: ≥ 5 GIỜ CẢNH NHIỄU):")
    print(f"• Thời lượng hiện có:  {total_hours:.2f} giờ ({total_sec/60:.1f} phút)")
    print(f"• Mục tiêu đề cương:   {target_hours:.1f} giờ")
    print(f"• Tỷ lệ hoàn thành:    {percent:.1f}%")
    
    if total_hours >= target_hours:
        print("  -> ĐÁNH GIÁ: ✅ ĐẠT VƯỢT CHỈ TIÊU MT1")
    else:
        print(f"  -> CẦN BỔ SUNG THÊM: {target_hours - total_hours:.2f} giờ video CCTV bình thường.")
        print("  💡 HƯỚNG DẪN:")
        print("     1. Chép thêm các video camera an ninh (độ dài 1-2 tiếng mỗi file) vào thư mục:")
        print(f"        {NEG_DIR.resolve()}")
        print("     2. Chạy lại script này để cập nhật tự động vào hệ thống.")
    print("="*70 + "\n")

if __name__ == "__main__":
    scan_negative_dataset()
