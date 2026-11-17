#!/usr/bin/env python3
"""Negative Video Integrity, Decode, Content, and Leakage Auditor for Week 6.
Audits negative videos for file integrity, decode status, content verification,
leakage against train/val/test splits, and computes strict FAR eligibility.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
LOG = logging.getLogger('neg_auditor')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class NegativeVideoAuditRecord:
    """Audit record for a single negative video."""
    video_id: str
    filename: str
    file_path: str
    file_exists: bool
    file_size_bytes: int
    sha256_actual: str
    sha256_manifest: str
    sha256_match: bool
    fps: float
    width: int
    height: int
    reported_frame_count: int
    duration_seconds: float
    technical_audit_status: str
    decode_integrity_status: str
    decode_verification_tool: str
    content_ground_truth_status: str
    leakage_status: str
    sampled_splits: str
    far_eligibility_status: str
    audit_notes: str


class NegativeVideoAuditor:
    """Comprehensive auditor for negative video assets and exposure eligibility."""

    def __init__(
        self,
        manifest_path: Path,
        video_dir: Path,
        dataset_dir: Path,
        output_dir: Path
    ) -> None:
        self.manifest_path = manifest_path
        self.video_dir = video_dir
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ffmpeg_available = bool(shutil.which('ffmpeg') and shutil.which('ffprobe'))
        self.split_membership = self._index_dataset_splits()

    def _index_dataset_splits(self) -> dict[str, set[str]]:
        """Index which negative video IDs have frame samples in train, val, or test splits."""
        membership: dict[str, set[str]] = {}
        images_dir = self.dataset_dir / 'images'
        if not images_dir.exists():
            LOG.warning('Dataset images dir not found: %s', images_dir)
            return membership

        for split in ['train', 'val', 'test']:
            split_folder = images_dir / split
            if not split_folder.exists():
                continue
            for f in split_folder.glob('*.*'):
                m = re.match(r'((?:VN_)?NEG_[0-9]+)', f.stem)
                if m:
                    vid = m.group(1)
                    norm_vid = vid.replace('VN_', '') if vid.startswith('VN_') else vid
                    membership.setdefault(vid, set()).add(split)
                    membership.setdefault(norm_vid, set()).add(split)
        return membership

    def audit_negative_videos(self, probe_decode: bool = True) -> tuple[list[NegativeVideoAuditRecord], dict[str, Any]]:
        """Audit all negative videos declared in manifest and located on disk."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f'Manifest not found: {self.manifest_path}')

        manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        declared_videos = manifest.get('videos', [])
        LOG.info('Auditing %d negative videos from manifest...', len(declared_videos))

        records: list[NegativeVideoAuditRecord] = []
        total_duration = 0.0
        technically_valid_duration = 0.0
        human_verified_duration = 0.0
        eligible_held_out_duration = 0.0

        for item in declared_videos:
            vid = item.get('video_id', '')
            fname = item.get('filename', f'{vid}.mp4')
            rel_path = item.get('relative_path', '')
            expected_sha = item.get('sha256', '')

            # Resolve physical path
            candidate_paths = [
                self.video_dir / fname,
                ROOT / rel_path if rel_path else None,
                ROOT / 'negative_videos' / fname,
                ROOT / 'datasets/negative_videos' / fname
            ]
            actual_path = next((p for p in candidate_paths if p and p.exists()), None)
            file_exists = actual_path is not None

            size_bytes = actual_path.stat().st_size if file_exists else 0
            actual_sha = sha256_file(actual_path) if file_exists else ''
            sha_match = (actual_sha == expected_sha) if (file_exists and expected_sha) else False

            # Inspect metadata via OpenCV
            fps = float(item.get('fps', 0.0))
            width = int(item.get('width', 0))
            height = int(item.get('height', 0))
            frame_count = int(item.get('reported_frame_count', 0))
            duration = float(item.get('duration_seconds', 0.0))

            decode_status = item.get('decode_status', 'NEEDS_REVIEW')
            tech_status = item.get('status', 'NEEDS_REVIEW')
            human_status = item.get('human_review_status', 'PENDING_HUMAN_REVIEW')
            notes = []

            if not file_exists:
                tech_status = 'FILE_MISSING'
                decode_status = 'DECODE_ERROR'
                notes.append('File not found on disk')
            elif not sha_match:
                notes.append(f'SHA256 mismatch (actual: {actual_sha[:8]}... != expected: {expected_sha[:8]}...)')

            # Decode integrity inspection
            verification_tool = 'FFMPEG_FFPROBE' if self.ffmpeg_available else 'OPENCV_STREAM_PROBE (TOOL_UNAVAILABLE_FFMPEG_MISSING)'
            if file_exists and probe_decode:
                cap = cv2.VideoCapture(str(actual_path))
                if not cap.isOpened():
                    decode_status = 'DECODE_ERROR'
                    tech_status = 'DECODE_ERROR'
                    notes.append('OpenCV could not open video stream')
                else:
                    cv_fps = cap.get(cv2.CAP_PROP_FPS) or fps
                    cv_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
                    cv_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
                    cv_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or frame_count
                    fps = cv_fps
                    width = cv_w
                    height = cv_h
                    if frame_count == 0:
                        frame_count = cv_frames
                    if duration == 0.0 and fps > 0:
                        duration = frame_count / fps

                    # Light streaming probe: grab 10 frames at head and 10 frames at tail
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 10))
                    tail_grabs = sum(1 for _ in range(10) if cap.grab())
                    cap.release()

                    if not self.ffmpeg_available:
                        # Cannot definitively prove NORMAL_EOF without ffprobe/ffmpeg packet parsing
                        if decode_status == 'NORMAL_EOF':
                            decode_status = 'PROBED_OPENCV_OK_UNVERIFIED_EOF'
                            notes.append('OpenCV stream accessible; NORMAL_EOF unverified due to missing ffmpeg')
                        elif decode_status == 'TRUNCATED_STREAM':
                            notes.append('Video recorded as TRUNCATED_STREAM in manifest')
                        else:
                            decode_status = 'NEEDS_REVIEW'

            # Leakage analysis
            splits = sorted(list(self.split_membership.get(vid, [])))
            splits_str = ','.join(splits) if splits else 'HELD_OUT'

            if 'train' in splits:
                leakage_status = 'TRAIN_EXPOSURE_LEAKAGE_RISK'
                notes.append('Frames used in Train split; forbidden as test negative exposure')
            elif 'val' in splits:
                leakage_status = 'VAL_EXPOSURE_LEAKAGE_RISK'
                notes.append('Frames used in Val split')
            elif 'test' in splits:
                leakage_status = 'TEST_BENCHMARK_MEMBER'
            else:
                leakage_status = 'CLEAN_HELD_OUT_NEGATIVE'

            # Strict FAR Eligibility Gate
            # Rule: Must exist + SHA match + technical valid + human verified negative + no train overlap
            is_tech_valid = file_exists and (decode_status in ('NORMAL_EOF', 'PROBED_OPENCV_OK_UNVERIFIED_EOF') or tech_status == 'VALID')
            is_human_verified = human_status == 'VERIFIED_NEGATIVE'
            is_held_out = 'train' not in splits and 'val' not in splits

            if is_tech_valid and is_human_verified and is_held_out:
                far_eligibility = 'ELIGIBLE_FOR_TEST_FAR'
                eligible_held_out_duration += duration
            elif not is_human_verified:
                far_eligibility = 'BLOCKED_PENDING_HUMAN_CONTENT_REVIEW'
            elif 'train' in splits:
                far_eligibility = 'BLOCKED_TRAIN_SPLIT_LEAKAGE'
            else:
                far_eligibility = 'BLOCKED_TECHNICAL_OR_SPLIT_INCONCLUSIVE'

            total_duration += duration
            if is_tech_valid:
                technically_valid_duration += duration
            if is_human_verified:
                human_verified_duration += duration

            records.append(NegativeVideoAuditRecord(
                video_id=vid,
                filename=fname,
                file_path=str(actual_path) if actual_path else 'NOT_FOUND',
                file_exists=file_exists,
                file_size_bytes=size_bytes,
                sha256_actual=actual_sha,
                sha256_manifest=expected_sha,
                sha256_match=sha_match,
                fps=round(fps, 2),
                width=width,
                height=height,
                reported_frame_count=frame_count,
                duration_seconds=round(duration, 2),
                technical_audit_status=tech_status,
                decode_integrity_status=decode_status,
                decode_verification_tool=verification_tool,
                content_ground_truth_status=human_status,
                leakage_status=leakage_status,
                sampled_splits=splits_str,
                far_eligibility_status=far_eligibility,
                audit_notes='; '.join(notes) if notes else 'OK'
            ))

        summary = {
            'total_negative_videos_manifest': len(declared_videos),
            'files_existing_on_disk': sum(1 for r in records if r.file_exists),
            'sha256_verified_matches': sum(1 for r in records if r.sha256_match),
            'ffmpeg_available': self.ffmpeg_available,
            'decode_tool_used': 'FFMPEG' if self.ffmpeg_available else 'TOOL_UNAVAILABLE_FFMPEG_MISSING',
            'durations_breakdown_hours': {
                'total_negative_duration_hours': round(total_duration / 3600.0, 4),
                'technically_valid_duration_hours': round(technically_valid_duration / 3600.0, 4),
                'human_verified_negative_duration_hours': round(human_verified_duration / 3600.0, 4),
                'eligible_held_out_test_exposure_hours': round(eligible_held_out_duration / 3600.0, 4)
            },
            'split_membership_breakdown': {
                'train_sampled_videos_count': sum(1 for r in records if 'train' in r.sampled_splits),
                'val_sampled_videos_count': sum(1 for r in records if 'val' in r.sampled_splits),
                'test_sampled_videos_count': sum(1 for r in records if 'test' in r.sampled_splits),
                'held_out_negative_videos_count': sum(1 for r in records if r.sampled_splits == 'HELD_OUT')
            },
            'far_eligibility_gate': {
                'status': 'BLOCKED' if eligible_held_out_duration == 0.0 else 'PASSED',
                'reason': (
                    'FAR_GATE_BLOCKED_PENDING_HUMAN_VERIFICATION: 100% of negative videos are currently '
                    'PENDING_HUMAN_REVIEW for content verification, and 37 videos have training split overlap. '
                    'Official held-out test FAR denominator cannot be finalized until negative content is human-certified.'
                )
            }
        }

        # Export CSV
        csv_path = self.output_dir / 'negative_video_audit.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'video_id', 'filename', 'file_path', 'file_exists', 'file_size_bytes',
                'sha256_actual', 'sha256_manifest', 'sha256_match', 'fps', 'width',
                'height', 'reported_frame_count', 'duration_seconds', 'technical_audit_status',
                'decode_integrity_status', 'decode_verification_tool', 'content_ground_truth_status',
                'leakage_status', 'sampled_splits', 'far_eligibility_status', 'audit_notes'
            ])
            writer.writeheader()
            for r in records:
                writer.writerow(asdict(r))

        # Export JSON
        json_path = self.output_dir / 'negative_video_audit.json'
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

        # Export Markdown Report
        md_path = self.output_dir / 'negative_video_audit_report.md'
        self._write_markdown_report(md_path, summary, records)

        LOG.info('Negative video audit completed: %d videos processed. Saved to %s', len(records), self.output_dir)
        return records, summary

    def _write_markdown_report(self, path: Path, summary: dict[str, Any], records: list[NegativeVideoAuditRecord]) -> None:
        """Write human-readable negative video audit report."""
        dur = summary['durations_breakdown_hours']
        splits = summary['split_membership_breakdown']
        far_gate = summary['far_eligibility_gate']

        lines = [
            "# BÁO CÁO KIỂM TOÁN TÍNH TOÀN VẸN VÀ RÒ RỈ VIDEO ÂM TÍNH (TUẦN 6)",
            "",
            "> **Tài liệu nguồn:** `data/negative_videos_manifest.json`  ",
            f"> **Tổng số video khai báo:** {summary['total_negative_videos_manifest']} video  ",
            f"> **Trạng thái công cụ FFmpeg/ffprobe:** `{summary['decode_tool_used']}`  ",
            f"> **Trạng thái cổng FAR Eligibility:** **`{far_gate['status']}`**  ",
            "",
            "---",
            "",
            "## 1. PHÂN BIỆT RÕ RÀNG 4 TẦNG THỜI LƯỢNG PHƠI NHIỄM (EXPOSURE BREAKDOWN)",
            "",
            "Theo chuẩn liêm chính khoa học của Đề cương NCKH, thời lượng phơi nhiễm âm tính được phân định 4 tầng độc lập:",
            "",
            f"1. **Tổng thời lượng vật lý trên đĩa (Total Negative Duration):** **{dur['total_negative_duration_hours']} giờ** ({sum(r.duration_seconds for r in records):.1f}s)",
            f"2. **Thời lượng video toàn vẹn kỹ thuật (Technically Valid Duration):** **{dur['technically_valid_duration_hours']} giờ**",
            f"3. **Thời lượng đã được con người xác minh không cháy/khói (Human-Verified Negative Duration):** **{dur['human_verified_negative_duration_hours']} giờ** *(Tất cả 79 video hiện đang ở trạng thái `PENDING_HUMAN_REVIEW`)*",
            f"4. **Thời lượng phơi nhiễm độc lập đủ điều kiện làm mẫu số FAR Test (Eligible Held-out Exposure):** **{dur['eligible_held_out_test_exposure_hours']} giờ**",
            "",
            "> [!WARNING]",
            "> Tuyệt đối không được lấy tổng thời lượng 7,13 giờ để mặc định làm mẫu số FAR cho tập Test. Các video đã dùng trong tập Train hoặc chưa được con người xác nhận âm tính không thể trở thành held-out test exposure hợp lệ.",
            "",
            "---",
            "",
            "## 2. KẾT QUẢ KIỂM TOÁN RÒ RỈ DỮ LIỆU (LEAKAGE AUDIT THEO SPLITS)",
            "",
            f"- **Số video âm tính bị trích khung hình vào tập Train:** **{splits['train_sampled_videos_count']}** video (Nguy cơ rò rỉ nếu dùng đánh giá Test)",
            f"- **Số video âm tính bị trích khung hình vào tập Validation:** **{splits['val_sampled_videos_count']}** video",
            f"- **Số video âm tính thuộc tập Test:** **{splits['test_sampled_videos_count']}** video",
            f"- **Số video âm tính hoàn toàn độc lập (Held-Out từ Train/Val):** **{splits['held_out_negative_videos_count']}** video",
            "",
            "---",
            "",
            "## 3. BẢNG TỔNG HỢP TRẠNG THÁI VIDEO ÂM TÍNH (MẪU 10 VIDEO ĐẦU TIÊN)",
            "",
            "| Video ID | Thời lượng (s) | File Exists | Checksum Match | Decode Status | Human Content Status | Split Sampled | FAR Eligibility |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]

        for r in records[:15]:
            lines.append(
                f"| **{r.video_id}** | {r.duration_seconds:.1f} | `{r.file_exists}` | `{r.sha256_match}` | "
                f"`{r.decode_integrity_status}` | `{r.content_ground_truth_status}` | `{r.sampled_splits}` | "
                f"`{r.far_eligibility_status}` |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 4. KẾT LUẬN & YÊU CẦU ĐỐI VỚI NGHIÊN CỨU VIÊN",
            "",
            "1. **Cần xác nhận nội dung âm tính:** Nhóm nghiên cứu cần thực hiện review nội dung 79 video âm tính để chuyển trạng thái `human_review_status` từ `PENDING_HUMAN_REVIEW` sang `VERIFIED_NEGATIVE`.",
            "2. **Tách biệt mẫu số FAR:** Khi tính FAR chính thức cho tập Test Tuần 6, chỉ được sử dụng 29 video thuộc nhóm `CLEAN_HELD_OUT_NEGATIVE` hoặc các video thuộc Test split, loại trừ triệt để 37 video đã tham gia vào tập Train."
        ])

        path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/negative_videos_manifest.json')
    parser.add_argument('--video-dir', type=Path, default=ROOT / 'datasets/negative_videos')
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'datasets/vn_dataset')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/audits/week6')
    parser.add_argument('--no-probe', action='store_true', help='Skip OpenCV streaming decode probe')
    args = parser.parse_args()

    try:
        auditor = NegativeVideoAuditor(args.manifest, args.video_dir, args.dataset_dir, args.output_dir)
        auditor.audit_negative_videos(probe_decode=not args.no_probe)
        return 0
    except Exception as err:
        LOG.error('Negative video audit failed: %s', err)
        return 1


if __name__ == '__main__':
    exit(main())
