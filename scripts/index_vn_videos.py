"""Đưa video Việt Nam vào data/video_dataset_list.csv.

- Quét thư mục videovietnam/ (pos và neg để chung, phân biệt bằng tên VN_POS_xx / VN_NEG_xx).
- Đọc độ phân giải, FPS, thời lượng bằng OpenCV.
- Chỉ thêm video chưa có trong CSV; KHÔNG ghi đè mốc thời gian đã gán tay.
- Mốc fire_start_sec / smoke_start_sec để -1 và ghi chú "CHUA_GAN_MOC":
  phải gán tay bằng scripts/tag_video_timestamps.py --prefix VN_.
- source_url điền tay (hoặc qua bảng SOURCE_IDS bên dưới).
"""
import sys
from pathlib import Path

import cv2
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

CSV_PATH = Path("data/video_dataset_list.csv")
VN_DIR = Path("videovietnam")

# ID YouTube suy ra từ dung lượng file gốc (tên YTSave.to_...) -> CẦN NGƯỜI XÁC NHẬN.
# Giá trị kết thúc bằng "?" là ID chưa đầy đủ.
SOURCE_IDS = {
    "VN_POS_01": "https://www.youtube.com/shorts/Q3LTwdupN54",
    "VN_POS_02": "https://www.youtube.com/shorts/tt9AsOLf-jc",
    "VN_POS_03": "https://www.youtube.com/watch?v=BeZ8p3hclk8",
    "VN_POS_04": "https://www.youtube.com/watch?v=m1VkQVUCB-?",
    "VN_POS_05": "https://www.youtube.com/watch?v=ZXv5sSacrAQ",
    "VN_NEG_01": "https://www.youtube.com/watch?v=VsdzfX3X2oo",
}


def fmt_dur(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def main():
    df = pd.read_csv(CSV_PATH)
    if "source_url" not in df.columns:
        df["source_url"] = ""
    existing = set(df["video_id"].astype(str))

    rows = []
    for p in sorted(VN_DIR.glob("VN_*.mp4")):
        vid = p.stem
        if vid in existing:
            continue
        if "_POS_" in vid:
            category = "vn_pos"
        elif "_NEG_" in vid:
            category = "vn_neg"
        else:
            print(f"⚠️ Bỏ qua {p.name}: tên không có _POS_ / _NEG_")
            continue

        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            print(f"⚠️ Không mở được {p.name}")
            continue
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = round(cap.get(cv2.CAP_PROP_FPS) or 0, 2)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        dur = round(n / fps, 2) if fps > 0 else 0

        notes = ["Video Viet Nam"]
        if category == "vn_pos":
            notes.append("CHUA_GAN_MOC")
        if h > w:
            notes.append("video doc - kiem tra co phai camera co dinh")
        url = SOURCE_IDS.get(vid, "")
        if url:
            notes.append("URL suy ra tu ten goc - can xac nhan")

        rows.append({
            "video_id": vid, "file_name": p.name, "relative_path": p.as_posix(),
            "category": category, "resolution": f"{w}x{h}", "fps": fps,
            "total_frames": n, "duration_sec": dur, "duration_formatted": fmt_dur(dur),
            "fire_start_sec": -1.0, "smoke_start_sec": -1.0,
            "notes": "; ".join(notes), "source_url": url,
        })

    if not rows:
        print("Không có video mới.")
        return
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ Đã thêm {len(rows)} video vào {CSV_PATH}")
    for r in rows:
        print(f"  {r['video_id']:<10} {r['category']:<7} {r['resolution']:>10} {r['duration_formatted']}  {r['source_url']}")


if __name__ == "__main__":
    main()
