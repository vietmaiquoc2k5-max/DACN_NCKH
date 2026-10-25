import os
import sys
import shutil
import cv2
import pandas as pd
from pathlib import Path

# Đảm bảo in tiếng Việt trên console Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DOWNLOADS_DIR = Path(r"C:\Users\vietm\Downloads")
FIRE_VIDEOS_DIR = DOWNLOADS_DIR / "fire_videos.1406"
SMOKE_VIDEOS_DIR = DOWNLOADS_DIR / "smoke_videos.1407"

TARGET_DIR = Path("datasets/test_videos")
CSV_PATH = Path("data/video_dataset_list.csv")

def copy_and_index_videos():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    sources = [
        {"dir": FIRE_VIDEOS_DIR / "pos", "category": "fire_pos", "subfolder": "pos"},
        {"dir": SMOKE_VIDEOS_DIR / "pos", "category": "smoke_pos", "subfolder": "pos"},
        {"dir": FIRE_VIDEOS_DIR / "neg", "category": "fire_neg", "subfolder": "neg"},
        {"dir": SMOKE_VIDEOS_DIR / "neg", "category": "smoke_neg", "subfolder": "neg"},
    ]
    
    records = []
    video_idx = 1
    
    print("\n" + "="*70)
    print("🎬 TẬP HỢP VÀ ĐÁNH CHỈ MỤC BỘ VIDEO KIỂM THỬ (TEST VIDEOS)")
    print("="*70)

    for src in sources:
        src_path = src["dir"]
        if not src_path.exists():
            print(f"⚠️ Không tìm thấy thư mục: {src_path}")
            continue
            
        dest_subdir = TARGET_DIR / src["subfolder"]
        dest_subdir.mkdir(parents=True, exist_ok=True)
        
        files = [f for f in src_path.iterdir() if f.is_file() and f.suffix.lower() in [".avi", ".mp4", ".mov", ".mkv"]]
        print(f"• Xử lý {len(files)} video từ {src['category']} ({src_path.name})...")
        
        for vf in files:
            dest_file = dest_subdir / vf.name
            if not dest_file.exists():
                shutil.copy2(vf, dest_file)
                
            # Đọc metadata video bằng OpenCV
            cap = cv2.VideoCapture(str(dest_file))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            
            duration_sec = round(total_frames / fps, 2) if fps > 0 else 0
            mins, secs = divmod(int(duration_sec), 60)
            hrs, mins = divmod(mins, 60)
            dur_fmt = f"{hrs:02d}:{mins:02d}:{secs:02d}"
            
            is_pos = "pos" in src["category"]
            v_id = f"POS_{video_idx:02d}" if is_pos else f"NEG_{video_idx:02d}"
            video_idx += 1
            
            records.append({
                "video_id": v_id,
                "file_name": vf.name,
                "relative_path": str(dest_file.as_posix()),
                "category": src["category"],
                "resolution": f"{width}x{height}",
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration_sec": duration_sec,
                "duration_formatted": dur_fmt,
                "fire_start_sec": 0 if is_pos and "fire" in src["category"] else -1,
                "smoke_start_sec": 0 if is_pos and "smoke" in src["category"] else -1,
                "notes": f"Dataset chuẩn {src['category']}"
            })
            
    # Ghi ra CSV
    df = pd.DataFrame(records)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    
    pos_count = len(df[df["category"].str.contains("pos")])
    neg_count = len(df[df["category"].str.contains("neg")])
    pos_dur = df[df["category"].str.contains("pos")]["duration_sec"].sum()
    neg_dur = df[df["category"].str.contains("neg")]["duration_sec"].sum()

    print("\n" + "="*70)
    print("📊 BÁO CÁO TỔNG HỢP VIDEO ĐÃ ĐƯA VÀO DỰ ÁN:")
    print("="*70)
    print(f"• Tổng số video dương tính (Có khói/lửa - POS): {pos_count} video ({pos_dur/60:.1f} phút)")
    print(f"  -> Chỉ tiêu Đề cương (≥ 20 video): {'✅ ĐẠT VƯỢT CHỈ TIÊU' if pos_count >= 20 else '❌ CHƯA ĐỦ'}")
    print(f"• Tổng số video âm tính (Cảnh nhiễu - NEG):     {neg_count} video ({neg_dur/60:.1f} phút)")
    print(f"• Thư mục lưu trữ: {TARGET_DIR.resolve()}")
    print(f"• File danh mục CSV: {CSV_PATH.resolve()}")
    print("="*70 + "\n")

if __name__ == "__main__":
    copy_and_index_videos()
