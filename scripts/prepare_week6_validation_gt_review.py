#!/usr/bin/env python3
"""Preparation tool for Week 6 Validation Ground Truth Cross-Verification.

Generates an independent, double-blind review worksheet for Annotator 2 (Lê Minh Thiên)
covering exactly the 5 Validation positive events:
- POS_07 / FIRE
- POS_08 / FIRE
- POS_08 / SMOKE
- POS_17 / SMOKE
- POS_18 / SMOKE

Scientific Invariant:
Annotator 1 timestamps (onset and offset) are strictly excluded from the review worksheet
to ensure genuine independent observation without bias or anchoring effect.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOG = logging.getLogger("validation_gt_prep")

TARGET_VALIDATION_VIDEOS = {"POS_07", "POS_08", "POS_17", "POS_18"}
TARGET_EVENT_SPECS = [
    ("POS_07", "FIRE"),
    ("POS_08", "FIRE"),
    ("POS_08", "SMOKE"),
    ("POS_17", "SMOKE"),
    ("POS_18", "SMOKE"),
]


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 digest of a file.

    Args:
        path: Path to the target file.

    Returns:
        Hexadecimal SHA-256 string.
    """
    if not path.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Safely write CSV file via temporary file to prevent corruption.

    Args:
        path: Destination path.
        rows: List of dictionary records.
        fieldnames: Column header list.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(dir=str(path.parent), prefix="tmp_gt_", suffix=".csv")
    try:
        with os.fdopen(temp_fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def resolve_positive_video_path(video_id: str) -> Path | None:
    """Resolve physical video path on disk for a positive video.

    Args:
        video_id: Video identifier (e.g. POS_07 or VN_POS_07).

    Returns:
        Path if found, None otherwise.
    """
    clean_id = video_id.replace("VN_", "")
    candidates = [
        ROOT / f"datasets/negative_videos/VN_{clean_id}.mp4",
        ROOT / f"datasets/negative_videos/{clean_id}.mp4",
        ROOT / f"datasets/test_videos/{clean_id}.mp4",
        ROOT / f"datasets/test_videos/pos/{clean_id}.mp4",
        ROOT / f"data/videos/{clean_id}.mp4",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def extract_validation_events(events_path: Path) -> list[dict[str, Any]]:
    """Extract exactly the 5 target validation positive events from events.jsonl.

    Args:
        events_path: Path to data/ground_truth/events.jsonl.

    Returns:
        List of event dictionaries matching the target specification.

    Raises:
        FileNotFoundError: If events_path does not exist.
        ValueError: If fewer than 5 or duplicate events are encountered.
    """
    if not events_path.exists():
        raise FileNotFoundError(f"Ground truth events file not found: {events_path}")

    extracted: list[dict[str, Any]] = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            vid = item.get("video_id", "")
            lbl = (item.get("label") or "").upper()
            clean_vid = vid.replace("VN_", "")

            if clean_vid in TARGET_VALIDATION_VIDEOS:
                # Check if this matches one of the 5 specs
                if (clean_vid, lbl) in TARGET_EVENT_SPECS:
                    item_copy = dict(item)
                    item_copy["source_line_no"] = line_no
                    item_copy["clean_video_id"] = clean_vid
                    extracted.append(item_copy)

    LOG.info("Extracted %d validation events from %s", len(extracted), events_path)
    if len(extracted) != 5:
        raise ValueError(f"Expected exactly 5 validation events, but found {len(extracted)}")

    return extracted


def prepare_validation_gt_worksheet(
    events_path: Path,
    output_dir: Path
) -> tuple[Path, Path]:
    """Build blind review worksheet and instruction guide for Annotator 2.

    Args:
        events_path: Path to events.jsonl.
        output_dir: Directory where review artifacts will be written.

    Returns:
        Tuple of (worksheet_csv_path, instruction_md_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    events = extract_validation_events(events_path)

    worksheet_rows = []
    for ev in events:
        vid = ev["clean_video_id"]
        lbl = ev["label"].upper()
        line_no = ev["source_line_no"]
        vpath = resolve_positive_video_path(vid)
        vsha = sha256_file(vpath) if vpath else "FILE_NOT_FOUND"
        rel_vpath = str(vpath.relative_to(ROOT)).replace("\\", "/") if vpath else "NOT_FOUND"

        event_id = f"VAL_{vid}_{lbl}"
        # If there are duplicate labels on same video (e.g. POS_08 has FIRE and SMOKE, distinct labels)
        worksheet_rows.append({
            "event_id": event_id,
            "source_line_no": line_no,
            "video_id": vid,
            "camera_id": ev.get("camera_id", f"BENCH_{vid}"),
            "target_label": lbl,
            "video_relative_path": rel_vpath,
            "video_sha256": vsha,
            "annotator_01_id": ev.get("annotator_01") or ev.get("annotator_id", "annotator_01"),
            # BLIND PROTOCOL: annotator_01 timestamps are strictly omitted!
            "annotator_02_id": "",
            "annotator_02_start_sec": "",
            "annotator_02_end_sec": "",
            "annotator_02_decision": "PENDING_REVIEW",
            "attest_full_view": "FALSE",
            "attest_independent": "FALSE",
            "review_timestamp": "",
            "review_notes": "",
        })

    fieldnames = [
        "event_id",
        "source_line_no",
        "video_id",
        "camera_id",
        "target_label",
        "video_relative_path",
        "video_sha256",
        "annotator_01_id",
        "annotator_02_id",
        "annotator_02_start_sec",
        "annotator_02_end_sec",
        "annotator_02_decision",
        "attest_full_view",
        "attest_independent",
        "review_timestamp",
        "review_notes",
    ]

    worksheet_csv = output_dir / "validation_gt_review_worksheet.csv"
    atomic_write_csv(worksheet_csv, worksheet_rows, fieldnames)
    LOG.info("Wrote validation GT review worksheet to %s", worksheet_csv)

    # Generate instructions Markdown
    instructions_md = output_dir / "validation_gt_review_instructions.md"
    _write_gt_instructions(instructions_md, worksheet_rows)
    LOG.info("Wrote review instructions to %s", instructions_md)

    return worksheet_csv, instructions_md


def _write_gt_instructions(path: Path, rows: list[dict[str, Any]]) -> None:
    """Generate Markdown instruction manual for Annotator 2."""
    lines = [
        "# QUY TRÌNH KIỂM ĐỊNH CHÉO GROUND TRUTH ĐỘC LẬP – TẬP VALIDATION (TUẦN 6)",
        "## FireWatch – Independent Double-Blind Ground Truth Protocol",
        "",
        "> **Tiêu chuẩn khoa học:** Quy trình SOP Bước 5 đề cương NCKH.  ",
        "> **Người thực hiện độc lập (Annotator 2):** Lê Minh Thiên (hoặc kiểm định viên độc lập được chỉ định).  ",
        "> **Nguyên tắc mù đôi (Blind Protocol):** Toàn bộ mốc thời gian của Người gán nhãn 1 được ẩn để tránh hiệu ứng mỏ neo (anchoring bias).  ",
        "",
        "---",
        "",
        "## 1. DANH SÁCH 5 SỰ KIỆN CẦN KIỂM ĐỊNH CHÉO",
        "",
        "| Event ID | Video ID | Nhãn sự kiện | Tệp video nguồn | Trạng thái hiện tại |",
        "| :--- | :---: | :---: | :--- | :---: |",
    ]

    for r in rows:
        lines.append(f"| `{r['event_id']}` | **{r['video_id']}** | `{r['target_label']}` | `{r['video_relative_path']}` | `{r['annotator_02_decision']}` |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. HƯỚNG DẪN CHI TIẾT CÁC BƯỚC THẨM ĐỊNH",
        "",
        "1. **Mở video gốc bằng trình phát chuyên dụng (VLC / MPC-HC):**",
        "   - Xem ở độ phân giải gốc và tốc độ 1.0x.",
        "   - Đảm bảo video hiển thị đầy đủ chi tiết khói mỏng hoặc đốm lửa ban đầu.",
        "",
        "2. **Ghi nhận mốc thời gian độc lập:**",
        "   - `annotator_02_start_sec`: Khung hình đầu tiên quan sát thấy dấu hiệu bắt đầu của ngọn lửa (`FIRE`) hoặc khói (`SMOKE`). Đơn vị: giây (chính xác đến 0.1s).",
        "   - `annotator_02_end_sec`: Khung hình ngọn lửa tắt hoàn toàn hoặc khói tan hết / ra khỏi khung hình.",
        "",
        "3. **Điền biểu mẫu thẩm định:**",
        "   - Mở tệp `runs/audits/week6/validation_gt_review_worksheet.csv`.",
        "   - Điền tên reviewer vào cột `annotator_02_id` (ví dụ: `Lê Minh Thiên`). Không được trùng với `annotator_01_id`.",
        "   - Điền mốc giây vào `annotator_02_start_sec` và `annotator_02_end_sec`.",
        "   - Chọn `annotator_02_decision`: `CONFIRM_EVENT` (xác nhận có sự kiện), `REJECT_EVENT` (không có sự kiện), hoặc `UNCERTAIN`.",
        "   - Đặt `attest_full_view`: `TRUE` (xác nhận đã xem toàn bộ video).",
        "   - Đặt `attest_independent`: `TRUE` (xác nhận đánh giá hoàn toàn độc lập).",
        "   - Ghi thời điểm kiểm định dạng ISO 8601 vào `review_timestamp` (ví dụ: `2026-11-17T18:00:00`).",
        "",
        "4. **Kiểm tra dung sai và đối soát:**",
        "   - Sau khi Annotator 2 nộp kết quả, chạy công cụ đối soát:",
        "     ```powershell",
        "     .\\.venv\\Scripts\\python.exe scripts/validate_week6_validation_reviews.py",
        "     ```",
        "   - Nếu $|start_1 - start_2| \\le 0.5\\text{s}$, hệ thống công nhận `AGREED_CONSENSUS` và cho phép mở khóa Validation Gate.",
        "   - Nếu sai số $> 0.5\\text{s}$, hai bên tiến hành họp đối soát theo SOP để thống nhất.",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Week 6 Validation Ground Truth Review Worksheet")
    parser.add_argument("--events", type=Path, default=ROOT / "data/ground_truth/events.jsonl",
                        help="Path to events.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/audits/week6",
                        help="Output directory for worksheet and instructions")
    args = parser.parse_args()

    try:
        ws, ins = prepare_validation_gt_worksheet(args.events, args.output_dir)
        LOG.info("Preparation complete. Worksheet: %s", ws)
        return 0
    except Exception as err:
        LOG.error("Failed to prepare validation GT worksheet: %s", err, exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
