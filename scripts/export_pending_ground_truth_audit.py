"""Xuất danh sách sự kiện Ground Truth chờ thẩm định chéo (Pending Events Audit Tool).

Phục vụ việc đối soát độc lập của Người gán nhãn thứ hai (Annotator 02).
- Liệt kê toàn bộ 24 sự kiện pending_verification.
- Cung cấp timestamp của annotator_01, khung dung sai tolerance 0.5s.
- Hướng dẫn chi tiết tiêu chuẩn thẩm định cho từng video.
Xuất tệp:
- data/ground_truth/pending_events_audit_list.csv
- data/ground_truth/pending_events_audit_list.json
"""
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("pending_gt_audit")


def export_pending_audit() -> None:
    LOGGER.info("=== BẮT ĐẦU XUẤT DANH SÁCH KIỂM ĐỊNH GROUND TRUTH PENDING ===")

    events_path = ROOT_DIR / "data" / "ground_truth" / "events.jsonl"
    catalog_path = ROOT_DIR / "data" / "video_dataset_list.csv"

    # Đọc thời lượng video từ catalog
    video_durations = {}
    if catalog_path.exists():
        with catalog_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                video_durations[row.get("video_id", "")] = float(row.get("duration_sec", 0.0))

    pending_list = []
    with events_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            status = item.get("verification_status", "pending_verification")
            if status != "cross_verified":
                vid = item.get("video_id", "")
                dur = video_durations.get(vid, 0.0)
                s1 = float(item.get("start_sec_annotator_01") or item.get("start_sec", 0.0))
                e1 = float(item.get("end_sec_annotator_01") or item.get("end_sec", dur))

                entry = {
                    "line_no": idx,
                    "video_id": vid,
                    "camera_id": item.get("camera_id", f"CAM_{vid}"),
                    "label": item.get("label", ""),
                    "video_duration_sec": dur,
                    "annotator_01_id": item.get("annotator_01") or item.get("annotator_id", "annotator_01"),
                    "start_sec_annotator_01": s1,
                    "end_sec_annotator_01": e1,
                    "acceptable_onset_min_sec": round(max(0.0, s1 - 0.5), 2),
                    "acceptable_onset_max_sec": round(s1 + 0.5, 2),
                    "start_sec_annotator_02": "",  # Chờ annotator_02 điền
                    "end_sec_annotator_02": "",    # Chờ annotator_02 điền
                    "current_status": status,
                    "action_required": "Annotator 02 cần xem video và ghi nhận mốc onset/offset độc lập",
                    "notes": item.get("notes", ""),
                }
                pending_list.append(entry)

    LOGGER.info("Tìm thấy %d sự kiện đang ở trạng thái pending_verification", len(pending_list))

    # Xuất file CSV
    out_csv = ROOT_DIR / "data" / "ground_truth" / "pending_events_audit_list.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "line_no", "video_id", "camera_id", "label", "video_duration_sec",
            "annotator_01_id", "start_sec_annotator_01", "end_sec_annotator_01",
            "acceptable_onset_min_sec", "acceptable_onset_max_sec",
            "start_sec_annotator_02", "end_sec_annotator_02",
            "current_status", "action_required", "notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in pending_list:
            writer.writerow(r)
    LOGGER.info("Đã lưu tệp CSV chờ kiểm định tại: %s", out_csv)

    # Xuất file JSON
    out_json = ROOT_DIR / "data" / "ground_truth" / "pending_events_audit_list.json"
    payload = {
        "metadata": {
            "title": "Ground Truth Pending Events Audit Manifest",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_pending_events": len(pending_list),
            "cross_validation_rules": {
                "tolerance_sec": 0.5,
                "rule_1": "Annotator 02 phải độc lập xem video gốc và xác định mốc bắt đầu (onset) và kết thúc (offset).",
                "rule_2": "Nếu |start_01 - start_02| <= 0.5s: tự động hợp lệ và được nâng lên cross_verified.",
                "rule_3": "Nếu |start_01 - start_02| > 0.5s: ghi nhận disputed và cần hai annotator họp thống nhất mốc thời gian.",
                "rule_4": "Tuyệt đối không tự động sao chép giá trị của annotator_01 sang annotator_02.",
            },
        },
        "pending_events": pending_list,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Đã lưu tệp JSON chờ kiểm định tại: %s", out_json)


if __name__ == "__main__":
    export_pending_audit()

