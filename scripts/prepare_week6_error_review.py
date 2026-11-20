#!/usr/bin/env python3
"""Error Review Preparation Tool for Week 6.
Extracts snapshot keyframes from negative video sources for the 60 false alarm cases
and generates the formal human review manifest and report without altering original files.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
LOG = logging.getLogger('error_review_prep')

STANDARD_TAXONOMY = [
    'VEHICLE_LIGHT',
    'REFLECTION',
    'LED_DISPLAY',
    'SUNSET_OR_SUNRISE',
    'DUST_OR_HAZE',
    'STEAM',
    'CLOUD_OR_FOG',
    'BRIGHT_OBJECT',
    'CAMERA_ARTIFACT',
    'UNKNOWN',
    'OTHER'
]


def _safe_rel(p: Path) -> str:
    """Return path relative to ROOT if possible, otherwise absolute path."""
    try:
        return str(p.relative_to(ROOT)).replace('\\', '/')
    except ValueError:
        return str(p).replace('\\', '/')


def prepare_review_package(
    catalog_csv: Path,
    video_dir: Path,
    output_dir: Path,
    extract_frames: bool = True
) -> dict[str, Any]:
    """Extract review keyframes and build review manifest."""
    if not catalog_csv.exists():
        raise FileNotFoundError(f'Error cases catalog not found: {catalog_csv}')

    evidence_dir = output_dir / 'error_review_evidence'
    evidence_dir.mkdir(parents=True, exist_ok=True)

    with open(catalog_csv, newline='', encoding='utf-8') as f:
        cases = list(csv.DictReader(f))

    LOG.info('Preparing review package for %d error cases...', len(cases))
    manifest_rows = []
    extracted_frames_count = 0
    missing_video_count = 0

    for c in cases:
        case_id = c.get('error_id', '')
        vid = c.get('video_id', '')
        ts_str = c.get('timestamp', '0.0s')
        frame_idx = int(c.get('frame_index', 0))
        ckpt_sha = c.get('model_sha256', '')

        # Resolve video file
        candidate_paths = [
            video_dir / f'{vid}.mp4',
            ROOT / f'datasets/negative_videos/{vid}.mp4',
            ROOT / f'negative_videos/{vid}.mp4'
        ]
        video_path = next((p for p in candidate_paths if p and p.exists()), None)

        evidence_filename = f'{case_id}_{vid}_f{frame_idx:06d}.jpg'
        evidence_path = evidence_dir / evidence_filename

        if video_path and extract_frames and not evidence_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Save snapshot image without overlaying text to preserve raw evidence
                    cv2.imwrite(str(evidence_path), frame)
                    extracted_frames_count += 1
                cap.release()
        elif evidence_path.exists():
            extracted_frames_count += 1
        elif not video_path:
            missing_video_count += 1
            LOG.warning('Video file missing for case %s: %s', case_id, vid)

        manifest_rows.append({
            'case_id': case_id,
            'event_id': f'EV_{case_id}',
            'video_id': vid,
            'timestamp': ts_str,
            'frame_index': frame_idx,
            'checkpoint_sha256': ckpt_sha,
            'evidence_path': _safe_rel(evidence_path) if evidence_path.exists() else 'NOT_EXTRACTED',
            'suspected_cause': 'UNASSIGNED',
            'reviewer_name': '',
            'review_timestamp': '',
            'reviewer_status': 'PENDING_REVIEW',
            'review_notes': ''
        })

    # Save review manifest CSV
    manifest_csv = output_dir / 'error_review_manifest.csv'
    with open(manifest_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'case_id', 'event_id', 'video_id', 'timestamp', 'frame_index',
            'checkpoint_sha256', 'evidence_path', 'suspected_cause',
            'reviewer_name', 'review_timestamp', 'reviewer_status', 'review_notes'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Generate Markdown guide and report
    report_md = output_dir / 'error_review_report.md'
    _write_review_report(report_md, manifest_rows, extracted_frames_count, missing_video_count)

    summary = {
        'total_cases': len(manifest_rows),
        'extracted_evidence_frames': extracted_frames_count,
        'missing_video_count': missing_video_count,
        'manifest_csv_path': _safe_rel(manifest_csv),
        'evidence_dir': _safe_rel(evidence_dir),
        'status': 'READY_FOR_HUMAN_REVIEW',
        'taxonomy': STANDARD_TAXONOMY
    }
    LOG.info('Error review package created at %s (%d frames extracted)', output_dir, extracted_frames_count)
    return summary


def _write_review_report(
    path: Path,
    rows: list[dict[str, Any]],
    extracted_count: int,
    missing_count: int
) -> None:
    """Generate Markdown instructions and inspection list for human review."""
    lines = [
        "# BIÊN BẢN HƯỚNG DẪN VÀ DANH MỤC THẨM ĐỊNH CA BÁO GIẢ (TUẦN 6)",
        "",
        "> **Tài liệu bàn giao:** `runs/audits/week6/error_review_manifest.csv`  ",
        f"> **Tổng số ca báo giả:** {len(rows)} ca (Trích xuất từ 353 ca lỗi thực tế của 79 video âm tính)  ",
        f"> **Khung hình bằng chứng đã trích xuất:** {extracted_count} / {len(rows)} ảnh  ",
        "> **Trạng thái ban đầu:** **`PENDING_REVIEW`** (100% ca chờ người thẩm định)  ",
        "",
        "---",
        "",
        "## 1. QUY CHUẨN PHÂN LOẠI NGUYÊN NHÂN THỊ GIÁC (TAXONOMY)",
        "",
        "Người nghiên cứu / thẩm định viên (Lê Minh Thiên / Mai Quốc Việt) khi mở từng ảnh bằng chứng tại thư mục `runs/audits/week6/error_review_evidence/` cần chọn một trong các mã nguyên nhân chuẩn sau:",
        "",
        "- `VEHICLE_LIGHT`: Ánh đèn pha ô tô, xe máy, đèn chiếu rọi trên đường ban đêm.",
        "- `REFLECTION`: Ánh sáng mặt trời phản chiếu qua kính tòa nhà, vũng nước, mặt đường nhựa bóng.",
        "- `LED_DISPLAY`: Màn hình LED quảng cáo ngoài trời, biển hiệu điện tử màu đỏ/cam/vàng.",
        "- `SUNSET_OR_SUNRISE`: Ánh nắng chói chang lúc bình minh hoặc hoàng hôn chiếu xiên vào camera.",
        "- `DUST_OR_HAZE`: Khói bụi mịn đường phố, xe tải xả bụi cát, ô nhiễm không khí mù mịt.",
        "- `STEAM`: Hơi nước bốc lên từ quán ăn, ống xả nhiệt máy lạnh, nồi hơi đun nấu.",
        "- `CLOUD_OR_FOG`: Mây bay thấp tầng, sương mù dày đặc che khuất tầm nhìn xa.",
        "- `BRIGHT_OBJECT`: Quần áo, dù che, ô bạt màu cam/đỏ rực rỡ dưới ánh nắng gắt.",
        "- `CAMERA_ARTIFACT`: Nhiễu hạt cảm biến ban đêm (sensor noise), vệt sọc quét ngang, mờ nét chuyển động (motion blur).",
        "- `UNKNOWN`: Không thể xác định rõ ràng sau khi xem clip/khung hình.",
        "- `OTHER`: Nguyên nhân khác (bắt buộc phải ghi rõ chi tiết tại cột `review_notes`).",
        "",
        "---",
        "",
        "## 2. QUY TRÌNH THỰC HIỆN THẨM ĐỊNH",
        "",
        "1. Mở file `runs/audits/week6/error_review_manifest.csv` bằng trình soạn thảo hoặc Excel/CSV viewer.",
        "2. Với mỗi dòng `case_id`: Mở ảnh tại đường dẫn `evidence_path` (ví dụ `runs/audits/week6/error_review_evidence/ERR_001_VN_NEG_04_f042654.jpg`).",
        "3. Điền các trường:",
        "   - `suspected_cause`: Chọn đúng mã trong Taxonomy ở Mục 1.",
        "   - `reviewer_name`: Tên người thẩm định (ví dụ `LeMinhThien` hoặc `MaiQuocViet`).",
        "   - `review_timestamp`: Thời điểm thẩm định theo chuẩn ISO (ví dụ `2026-11-17T18:00:00Z`).",
        "   - `reviewer_status`: Chuyển từ `PENDING_REVIEW` sang `VERIFIED`.",
        "   - `review_notes`: Ghi chú bổ sung nếu chọn `OTHER` hoặc `UNKNOWN`.",
        "4. Chạy script xác thực: `python scripts/validate_week6_error_reviews.py` để bảo đảm không có trường nào bị thiếu hoặc gán sai chuẩn.",
        "",
        "---",
        "",
        "## 3. DANH SÁCH 60 CA BÁO GIẢ CHỜ THẨM ĐỊNH (MẪU 15 CA ĐẦU TIÊN)",
        "",
        "| Case ID | Video ID | Timestamp | Frame Index | Bằng chứng ảnh | Trạng thái hiện tại |",
        "| :--- | :---: | :---: | :---: | :--- | :---: |",
    ]

    for r in rows[:15]:
        lines.append(
            f"| **{r['case_id']}** | `{r['video_id']}` | {r['timestamp']} | {r['frame_index']} | "
            f"`{r['evidence_path']}` | `{r['reviewer_status']}` |"
        )

    lines.extend([
        "",
        "*(Toàn bộ 60 ca được lưu đầy đủ trong `runs/audits/week6/error_review_manifest.csv`)*."
    ])

    path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=ROOT / 'data/error_cases_catalog.csv')
    parser.add_argument('--video-dir', type=Path, default=ROOT / 'datasets/negative_videos')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/audits/week6')
    parser.add_argument('--no-extract', action='store_true', help='Skip extracting frame images')
    args = parser.parse_args()

    try:
        prepare_review_package(
            catalog_csv=args.catalog,
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            extract_frames=not args.no_extract
        )
        return 0
    except Exception as err:
        LOG.error('Failed to prepare error review package: %s', err)
        return 1


if __name__ == '__main__':
    exit(main())
