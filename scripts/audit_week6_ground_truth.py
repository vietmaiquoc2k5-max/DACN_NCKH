#!/usr/bin/env python3
"""Ground Truth Completeness and Verification Auditor for Week 6.
Audits events.jsonl against split manifests, inspects validation and test video GT,
and enforces Validation Gate blocking when validation positive GT is pending.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
LOG = logging.getLogger('gt_auditor')


@dataclass
class EventAuditRecord:
    """Audit record for a single ground truth event."""
    line_number: int
    video_id: str
    camera_id: str
    label: str
    start_sec: float
    end_sec: float
    duration_sec: float
    verification_status: str
    annotator_01: str | None
    annotator_02: str | None
    split_assignment: str
    status_classification: str
    audit_notes: str


class GroundTruthAuditor:
    """Audits ground truth events without modifying dataset or ground truth files."""

    def __init__(
        self,
        events_path: Path,
        split_manifest_path: Path,
        output_dir: Path
    ) -> None:
        self.events_path = events_path
        self.split_manifest_path = split_manifest_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.video_metadata: dict[str, dict[str, Any]] = {}
        self._load_video_metadata()

    def _load_video_metadata(self) -> None:
        """Load video durations and proposed splits from preannotation split manifest."""
        if not self.split_manifest_path.exists():
            LOG.warning('Split manifest not found: %s', self.split_manifest_path)
            return

        data = json.loads(self.split_manifest_path.read_text(encoding='utf-8'))
        for v in data.get('videos', []):
            vid = v.get('video_id')
            # Normalize ID: e.g. VN_POS_01 -> POS_01 and VN_POS_01
            norm_id = vid.replace('VN_', '') if vid.startswith('VN_') else vid
            info = {
                'video_id': vid,
                'normalized_id': norm_id,
                'duration_seconds': v.get('duration_seconds', 0.0),
                'proposed_split': v.get('proposed_split', 'UNASSIGNED'),
                'source_available': v.get('source_available', False)
            }
            self.video_metadata[vid] = info
            self.video_metadata[norm_id] = info

    def audit_events(self) -> tuple[list[EventAuditRecord], dict[str, Any]]:
        """Run comprehensive audit on all ground truth records."""
        if not self.events_path.exists():
            raise FileNotFoundError(f'Ground truth events file not found: {self.events_path}')

        lines = [l.strip() for l in self.events_path.read_text(encoding='utf-8').splitlines() if l.strip()]
        records: list[EventAuditRecord] = []
        seen_events: set[tuple[str, str, float, float]] = set()

        total_events = len(lines)
        cross_verified_count = 0
        pending_count = 0
        error_count = 0

        for line_no, line in enumerate(lines, start=1):
            try:
                ev = json.loads(line)
            except Exception as e:
                error_count += 1
                records.append(EventAuditRecord(
                    line_number=line_no, video_id='CORRUPT_JSON', camera_id='UNKNOWN',
                    label='UNKNOWN', start_sec=0.0, end_sec=0.0, duration_sec=0.0,
                    verification_status='ERROR', annotator_01=None, annotator_02=None,
                    split_assignment='UNKNOWN', status_classification='CORRUPT_RECORD',
                    audit_notes=f'JSON parse error: {e}'
                ))
                continue

            vid = ev.get('video_id', '')
            label = ev.get('label', '')
            start = float(ev.get('start_sec', 0.0))
            end = float(ev.get('end_sec', 0.0))
            duration = end - start
            v_status = ev.get('verification_status', 'unknown')
            ann1 = ev.get('annotator_01')
            ann2 = ev.get('annotator_02')

            vid_meta = self.video_metadata.get(vid, {})
            split = vid_meta.get('proposed_split', 'UNASSIGNED')
            video_dur = vid_meta.get('duration_seconds', 0.0)

            # Integrity checks
            notes = []
            classification = v_status.upper()

            if not vid:
                classification = 'INVALID_REFERENCE'
                notes.append('Missing video_id')
            elif vid not in self.video_metadata:
                notes.append('Video ID not found in split manifest')

            if start < 0:
                classification = 'INCONSISTENT_METADATA'
                notes.append(f'Negative start_sec ({start})')
            if end <= start:
                classification = 'INCONSISTENT_METADATA'
                notes.append(f'Invalid interval (start {start} >= end {end})')
            if video_dur > 0 and end > (video_dur + 1.0):
                classification = 'INCONSISTENT_METADATA'
                notes.append(f'Event end ({end}s) exceeds video duration ({video_dur}s)')

            # Duplicate check
            ev_key = (vid, label, round(start, 2), round(end, 2))
            if ev_key in seen_events:
                classification = 'DUPLICATE_RECORD'
                notes.append('Duplicate event record with identical video, label, interval')
            else:
                seen_events.add(ev_key)

            if v_status == 'cross_verified':
                cross_verified_count += 1
                if not ann2:
                    notes.append('Cross_verified but annotator_02 field is empty')
            elif v_status == 'pending_verification':
                pending_count += 1
                notes.append('Pending verification by Annotator 02')
            else:
                classification = 'UNKNOWN_STATUS'
                notes.append(f'Unrecognized verification status: {v_status}')

            records.append(EventAuditRecord(
                line_number=line_no,
                video_id=vid,
                camera_id=ev.get('camera_id', ''),
                label=label,
                start_sec=start,
                end_sec=end,
                duration_sec=duration,
                verification_status=v_status,
                annotator_01=ann1,
                annotator_02=ann2,
                split_assignment=split,
                status_classification=classification,
                audit_notes='; '.join(notes) if notes else 'OK'
            ))

        # Check Validation Gate
        val_positive_vids = ['POS_07', 'POS_08', 'POS_17', 'POS_18', 'VN_POS_07', 'VN_POS_08', 'VN_POS_17', 'VN_POS_18']
        val_records = [r for r in records if r.video_id in val_positive_vids]
        val_verified = [r for r in val_records if r.verification_status == 'cross_verified']
        val_pending = [r for r in val_records if r.verification_status == 'pending_verification']

        validation_gate_blocked = len(val_records) == 0 or len(val_verified) == 0
        validation_gate_reason = (
            f"VALIDATION_GATE_BLOCKED: All {len(val_records)} validation events across "
            f"{len(set(r.video_id for r in val_records))} videos are pending verification. "
            "Event-level tuning and official validation acceptance cannot proceed."
        ) if validation_gate_blocked else "VALIDATION_GATE_PASSED"

        # Check Test Set Completeness
        test_vids = [
            'POS_09', 'POS_10', 'POS_19', 'POS_20',
            'POS_69', 'POS_70', 'POS_71', 'POS_72', 'POS_73', 'POS_74',
            'POS_75', 'POS_76', 'POS_77', 'POS_78', 'POS_79', 'POS_80'
        ]
        test_audit: dict[str, Any] = {}
        annotated_test_vids = set()
        for tv in test_vids:
            norm_tv = tv.replace('VN_', '')
            matched_records = [r for r in records if r.video_id in (tv, f'VN_{tv}', norm_tv)]
            if matched_records:
                annotated_test_vids.add(tv)
                test_audit[tv] = {
                    'annotation_status': 'ANNOTATED',
                    'event_count': len(matched_records),
                    'verification_statuses': list(set(r.verification_status for r in matched_records)),
                    'events': [asdict(r) for r in matched_records]
                }
            else:
                test_audit[tv] = {
                    'annotation_status': 'MISSING_ANNOTATION',
                    'event_count': 0,
                    'verification_statuses': [],
                    'events': []
                }

        summary = {
            'total_events_in_manifest': total_events,
            'cross_verified_events': cross_verified_count,
            'pending_verification_events': pending_count,
            'error_records_count': error_count,
            'validation_gate': {
                'status': 'BLOCKED' if validation_gate_blocked else 'PASSED',
                'reason': validation_gate_reason,
                'total_val_events': len(val_records),
                'verified_val_events': len(val_verified),
                'pending_val_events': len(val_pending),
                'validation_videos_checked': sorted(list(set(r.video_id for r in val_records)))
            },
            'test_gt_completeness': {
                'total_test_videos_target': len(test_vids),
                'annotated_test_videos_count': len(annotated_test_vids),
                'missing_test_videos_count': len(test_vids) - len(annotated_test_vids),
                'annotated_test_videos': sorted(list(annotated_test_vids)),
                'missing_test_videos': sorted([v for v in test_vids if v not in annotated_test_vids]),
                'details': test_audit
            }
        }

        # Export CSV
        csv_path = self.output_dir / 'ground_truth_audit.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'line_number', 'video_id', 'camera_id', 'label', 'start_sec',
                'end_sec', 'duration_sec', 'verification_status', 'annotator_01',
                'annotator_02', 'split_assignment', 'status_classification', 'audit_notes'
            ])
            writer.writeheader()
            for r in records:
                writer.writerow(asdict(r))

        # Export JSON
        json_path = self.output_dir / 'ground_truth_audit.json'
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

        # Export Markdown Report
        md_path = self.output_dir / 'ground_truth_completeness_report.md'
        self._write_markdown_report(md_path, summary, records)

        LOG.info('Ground truth audit completed: %d events audited. Saved to %s', len(records), self.output_dir)
        return records, summary

    def _write_markdown_report(self, path: Path, summary: dict[str, Any], records: list[EventAuditRecord]) -> None:
        """Write human-readable completeness report."""
        val_gate = summary['validation_gate']
        test_gt = summary['test_gt_completeness']

        lines = [
            "# BÁO CÁO KIỂM TOÁN TÍNH ĐẦY ĐỦ VÀ XÁC MINH GROUND TRUTH (TUẦN 6)",
            "",
            "> **Tài liệu nguồn:** `data/ground_truth/events.jsonl`  ",
            f"> **Tổng số sự kiện:** {summary['total_events_in_manifest']} sự kiện  ",
            f"> **Trạng thái Validation Gate:** **{val_gate['status']}**  ",
            "",
            "---",
            "",
            "## 1. TỔNG HỢP TRẠNG THÁI GROUND TRUTH TOÀN BỘ SỰ KIỆN",
            "",
            f"- **Tổng số sự kiện khai báo:** {summary['total_events_in_manifest']}",
            f"- **Số sự kiện đã kiểm định chéo (`cross_verified`):** **{summary['cross_verified_events']}**",
            f"- **Số sự kiện đang chờ kiểm định chéo (`pending_verification`):** **{summary['pending_verification_events']}**",
            f"- **Số sự kiện lỗi/bất thường:** {summary['error_records_count']}",
            "",
            "---",
            "",
            "## 2. KIỂM ĐỊNH VALIDATION GATE (CHO PHÉP TỐI ƯU EVENT-LEVEL HAY KHÔNG)",
            "",
            f"**Trạng thái cổng:** **`{val_gate['status']}`**  ",
            f"**Lý do:** {val_gate['reason']}  ",
            "",
            "| Video Validation | Số sự kiện khai báo | Trạng thái hiện tại | Yêu cầu hành động |",
            "| :--- | :---: | :---: | :--- |",
        ]

        val_positive_vids = ['POS_07', 'POS_08', 'POS_17', 'POS_18']
        for vid in val_positive_vids:
            matching = [r for r in records if r.video_id in (vid, f'VN_{vid}')]
            st = matching[0].verification_status if matching else 'MISSING'
            lines.append(f"| **{vid}** | {len(matching)} | `{st}` | Cần Annotator 02 kiểm định chéo mốc onset/offset |")

        lines.extend([
            "",
            "> [!IMPORTANT]",
            "> Vì toàn bộ các video Validation dương tính đều đang ở trạng thái `pending_verification`, hạ tầng thực nghiệm Tuần 6 **CHẶN ĐỨNG (BLOCKED)** việc tối ưu hóa Event-Level Recall hoặc tính Detection Delay chính thức để bảo đảm tính khách quan.",
            "",
            "---",
            "",
            "## 3. KIỂM ĐỊNH TÍNH ĐẦY ĐỦ CỦA GROUND TRUTH TẬP TEST (TEST SET AUDIT)",
            "",
            f"- **Mục tiêu số video Test dương tính:** {test_gt['total_test_videos_target']} videos",
            f"- **Số video Test đã có nhãn sự kiện trong manifest:** {test_gt['annotated_test_videos_count']} / {test_gt['total_test_videos_target']}",
            f"- **Số video Test THIẾU nhãn sự kiện (`MISSING_ANNOTATION`):** **{test_gt['missing_test_videos_count']}** videos",
            "",
            "| Video Test ID | Trạng thái nhãn sự kiện | Trạng thái kiểm định | Số sự kiện | Ghi chú khoa học |",
            "| :--- | :---: | :---: | :---: | :--- |",
        ])

        for vid, details in sorted(test_gt['details'].items()):
            ann_st = details['annotation_status']
            ver_st = ', '.join(details['verification_statuses']) if details['verification_statuses'] else 'N/A'
            cnt = details['event_count']
            note = 'Đạt chuẩn' if 'cross_verified' in ver_st else ('Chờ Annotator 02' if cnt > 0 else 'CHƯA GÁN NHÃN SỰ KIỆN')
            lines.append(f"| **{vid}** | `{ann_st}` | `{ver_st}` | {cnt} | {note} |")

        lines.extend([
            "",
            "---",
            "",
            "## 4. DANH SÁCH CÔNG VIỆC CẦN ANNOTATOR XỬ LÝ (ACTION ITEMS)",
            "",
            "1. **Validation Set (Ưu tiên 1):** Annotator 02 (Lê Minh Thiên) cần xác nhận chéo 5 sự kiện trên 4 video (`POS_07`, `POS_08`, `POS_17`, `POS_18`) trong file `data/ground_truth/pending_events_audit_list.json` để mở khóa Validation Gate.",
            "2. **Test Set (Ưu tiên 2):** Cần bổ sung nhãn sự kiện thời gian (start_sec, end_sec, label) cho 12 video test còn thiếu (`POS_69` đến `POS_80`) trước khi thực hiện đánh giá chính thức trên tập Test.",
            "3. **Liêm chính khoa học:** Tuyệt đối không tự suy diễn hoặc dùng mô hình sinh nhãn tự động thay thế cho các video thiếu annotation."
        ])

        path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, default=ROOT / 'data/ground_truth/events.jsonl')
    parser.add_argument(
        '--split-manifest',
        type=Path,
        default=ROOT / 'data/vn_pos_preannotation_split_manifest.json'
    )
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/audits/week6')
    args = parser.parse_args()

    try:
        auditor = GroundTruthAuditor(args.events, args.split_manifest, args.output_dir)
        auditor.audit_events()
        return 0
    except Exception as err:
        LOG.error('Ground truth audit failed: %s', err)
        return 1


if __name__ == '__main__':
    exit(main())
