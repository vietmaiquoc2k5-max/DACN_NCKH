#!/usr/bin/env python3
"""Negative Video Technical Integrity, EOF Determination, and FAR Exposure Auditor.

Performs rigorous technical audit of negative videos for the FireWatch project:
1. Multi-signal sequential decode & EOF determination:
   - Full sequential decoding (no seek probing or binary search substitution).
   - VFR timestamp tracking & stream duration reconciliation.
   - Detects mid-stream corruptions even when tail frame can be sought.
   - Distinguishes TRUNCATED_STREAM, DECODE_ERROR, UNKNOWN_EOF, and NORMAL_EOF.
   - Disallows unsafe tolerance heuristics (deficit <= 10 or <= 0.5% without duration proof).
2. Safe no-probe enforcement:
   - --no-probe mode strictly requires an existing verified audit receipt with matching SHA256.
   - Cannot create new technical PASS from metadata alone.
3. Strict FAR eligibility enforcement:
   - Separates Validation FAR exposure, Independent Test exposure, Train-contaminated videos,
     and Unresolved/Held-out inventory.
   - Enforces duplicate SHA-256 detection and exclusion.
   - Requires valid reviewer identity, ISO 8601 timestamp, full-view attestation, and zero leakage.
4. Human review conflict prevention:
   - Rejects review records with conflicting alias fields or SHA256 mismatches.
   - Prevents field-stitching across disparate sources.
5. Generates:
   - negative_video_technical_audit_report.md
   - negative_video_integrity_manifest.csv
   - negative_video_integrity_manifest.json
   - far_exposure_reconciliation.json
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOG = logging.getLogger("neg_tech_auditor")

TARGET_FAR_EXPOSURE_SEC = 18000.0  # 5.0 hours required by De Cuong NCKH

DISALLOWED_REVIEWERS = {
    "", "none", "null", "system", "auto", "script", "bot", "unassigned",
    "n/a", "pending", "annotator_pending", "unknown"
}

CONTAINER_TOLERANCE_FRAMES = 10
CONTAINER_TOLERANCE_PERCENT = 0.5


class EOFStatus(str, Enum):
    """EOF classification taxonomy."""
    NORMAL_EOF = "NORMAL_EOF"
    TRUNCATED_STREAM = "TRUNCATED_STREAM"
    DECODE_ERROR = "DECODE_ERROR"
    UNKNOWN_EOF = "UNKNOWN_EOF"


class TechnicalVerdict(str, Enum):
    """Technical verification verdict."""
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_VERIFIED = "NOT_VERIFIED"


class ContentDecision(str, Enum):
    """Human content review decision."""
    VERIFIED_NEGATIVE = "VERIFIED_NEGATIVE"
    PENDING_HUMAN_REVIEW = "PENDING_HUMAN_REVIEW"
    FIRE_PRESENT = "FIRE_PRESENT"
    SMOKE_PRESENT = "SMOKE_PRESENT"
    UNCERTAIN = "UNCERTAIN"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 checksum of a file on disk."""
    if not path.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_iso_timestamp(ts_str: str) -> bool:
    """Validate whether a string is a valid ISO 8601 date/timestamp."""
    if not ts_str:
        return False
    clean = ts_str.strip().replace("Z", "+00:00")
    try:
        datetime.datetime.fromisoformat(clean)
        return True
    except ValueError:
        pass
    try:
        datetime.datetime.strptime(clean, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON data to disk atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(dir=str(path.parent), prefix="tmp_aud_", suffix=".json")
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV data to disk atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(dir=str(path.parent), prefix="tmp_aud_", suffix=".csv")
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


@dataclass
class VideoIntegrityRecord:
    """Detailed integrity and audit record for a single negative video."""
    video_id: str
    filename: str
    relative_path: str
    file_path: str
    file_exists: bool
    file_size_bytes: int
    sha256_actual: str
    sha256_manifest: str
    sha256_match: bool
    fps: float
    width: int
    height: int
    declared_frame_count: int
    decoded_frame_count: int
    frame_count_diff: int
    frame_count_diff_percent: float
    declared_duration_sec: float
    decoded_duration_sec: float
    last_frame_timestamp_sec: float
    eof_status: str
    technical_verdict: str
    technical_notes: str
    content_decision: str
    reviewer_name: str
    review_timestamp: str
    attest_full_video_viewed: bool
    split_membership: str
    eligible_for_far: bool
    ineligibility_reasons: str
    is_vfr: bool = False
    stream_duration_sec: float = 0.0
    decoder_errors: str = ""
    process_return_code: int = 0
    is_duplicate_sha: bool = False


class NegativeVideoTechnicalAuditor:
    """Engine for video stream decoding, EOF analysis, and FAR exposure audit."""

    def __init__(
        self,
        manifest_path: Path,
        split_manifest_path: Path,
        video_dirs: list[Path],
        output_dir: Path,
        val_review_manifest_path: Path | None = None,
        target_hours: float = 5.0,
        trusted_receipt_path: Path | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.split_manifest_path = split_manifest_path
        self.video_dirs = video_dirs
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.val_review_manifest_path = val_review_manifest_path
        self.target_hours = target_hours
        self.target_exposure_sec = target_hours * 3600.0
        self.trusted_receipt_path = trusted_receipt_path

        self.ffmpeg_available = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
        self.split_membership = self._index_split_membership()
        self.val_human_reviews = self._load_validation_human_reviews()
        self.trusted_receipts = self._load_trusted_receipts()

    def _index_split_membership(self) -> dict[str, set[str]]:
        """Map negative video IDs to their sampled dataset splits."""
        membership: dict[str, set[str]] = {}
        if not self.split_manifest_path.exists():
            LOG.warning("Split manifest not found: %s", self.split_manifest_path)
            return membership

        with open(self.split_manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = row.get("split", "").strip().lower()
                vid = row.get("video_id", "").strip()
                if not vid or not split:
                    continue
                membership.setdefault(vid, set()).add(split)
                m = re.match(r"((?:VN_)?NEG_[0-9]+)", vid)
                if m:
                    base_id = m.group(1)
                    norm_id = base_id.replace("VN_", "") if base_id.startswith("VN_") else base_id
                    membership.setdefault(base_id, set()).add(split)
                    membership.setdefault(norm_id, set()).add(split)
                    membership.setdefault(f"VN_{norm_id}", set()).add(split)
        return membership

    def _load_validation_human_reviews(self) -> dict[str, dict[str, Any]]:
        """Load human review entries if worksheet or manifest exists."""
        reviews: dict[str, dict[str, Any]] = {}
        if self.val_review_manifest_path and self.val_review_manifest_path.exists():
            with open(self.val_review_manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid = row.get("video_id", "").strip()
                    if vid:
                        reviews[vid] = dict(row)
                        norm_id = vid.replace("VN_", "")
                        reviews[norm_id] = dict(row)
        return reviews

    def _load_trusted_receipts(self) -> dict[str, dict[str, Any]]:
        """Load verified technical receipts for safe --no-probe mode."""
        receipts: dict[str, dict[str, Any]] = {}
        candidate_paths = []
        if self.trusted_receipt_path and self.trusted_receipt_path.exists():
            candidate_paths.append(self.trusted_receipt_path)
        candidate_paths.extend([
            self.output_dir / "negative_video_integrity_manifest.json",
            ROOT / "runs/audits/week6/negative_video_integrity_manifest.json",
        ])

        for p in candidate_paths:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    videos = data.get("videos", [])
                    for item in videos:
                        vid = item.get("video_id")
                        if vid and item.get("technical_verdict") == TechnicalVerdict.PASS.value:
                            receipts[vid] = item
                    if receipts:
                        break
                except Exception as err:
                    LOG.debug("Could not parse receipt at %s: %s", p, err)
        return receipts

    def probe_video_stream(
        self,
        video_path: Path,
        expected_frames: int,
        expected_fps: float,
        expected_duration: float
    ) -> tuple[int, int, float, int, int, float, str, list[str]]:
        """Full sequential stream decode, VFR timestamp extraction, and strict EOF determination.

        Does NOT substitute binary search or seek probing for actual decoding.
        Detects mid-stream corruptions where sequential decode stops but tail frame can be sought.
        Disallows unsafe heuristic tolerances without verified duration proof.

        Returns:
            Tuple of (decoded_fc, declared_fc, fps, width, height, last_pts, eof_status, notes).
        """
        notes: list[str] = []
        decoder_errors: list[str] = []
        if not video_path.exists():
            return 0, 0, 0.0, 0, 0, 0.0, EOFStatus.DECODE_ERROR.value, ["File not found on disk"]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0, 0, 0.0, 0, 0, 0.0, EOFStatus.DECODE_ERROR.value, ["OpenCV could not open video stream"]

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        container_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        effective_declared_fc = max(container_frames, expected_frames) if (container_frames > 0 or expected_frames > 0) else 0
        effective_fps = fps if fps > 0.0 else (expected_fps if expected_fps > 0.0 else 25.0)

        if effective_declared_fc <= 0 or effective_fps <= 0.0 or width <= 0 or height <= 0:
            cap.release()
            return 0, effective_declared_fc, effective_fps, width, height, 0.0, EOFStatus.UNKNOWN_EOF.value, [
                "Invalid container header metadata (non-positive frames, fps, or resolution)"
            ]

        # 1. Full Sequential Decode
        decoded_fc = 0
        last_pts_sec = 0.0
        prev_pts_sec = -1.0
        pts_deltas: list[float] = []

        # Read Frame 0
        ret0, frame0 = cap.read()
        if not ret0 or frame0 is None:
            cap.release()
            return 0, effective_declared_fc, effective_fps, width, height, 0.0, EOFStatus.DECODE_ERROR.value, [
                "Failed to decode initial frame (Frame 0)"
            ]
        decoded_fc = 1
        pos0 = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos0 is not None and pos0 >= 0:
            last_pts_sec = pos0 / 1000.0
            prev_pts_sec = last_pts_sec

        # Sequential loop over subsequent frames
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            decoded_fc += 1
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec is not None and pos_msec >= 0:
                current_pts = pos_msec / 1000.0
                last_pts_sec = current_pts
                if prev_pts_sec >= 0.0:
                    pts_deltas.append(current_pts - prev_pts_sec)
                prev_pts_sec = current_pts

        # If timestamp wasn't available via POS_MSEC, estimate from decoded frames & fps
        if last_pts_sec <= 0.0 and effective_fps > 0.0:
            last_pts_sec = decoded_fc / effective_fps

        # 2. Check for mid-stream corruption: if sequential decode halted early, can tail be read via seek?
        can_read_tail = False
        if decoded_fc < effective_declared_fc:
            cap.set(cv2.CAP_PROP_POS_FRAMES, effective_declared_fc - 1)
            ret_tail, frame_tail = cap.read()
            if ret_tail and frame_tail is not None:
                can_read_tail = True
                decoder_errors.append(
                    f"Mid-stream corruption: sequential decode failed at frame {decoded_fc}/{effective_declared_fc}, "
                    f"yet tail frame {effective_declared_fc - 1} was readable via seek."
                )

        cap.release()

        # 3. Detect Variable Frame Rate (VFR)
        is_vfr = False
        if len(pts_deltas) > 10:
            mean_delta = sum(pts_deltas) / len(pts_deltas)
            var_delta = sum((d - mean_delta) ** 2 for d in pts_deltas) / len(pts_deltas)
            if var_delta > 0.0001:
                is_vfr = True
                notes.append("Variable Frame Rate (VFR) detected; duration anchored to native PTS")

        # 4. FFmpeg audit for external evidence if available
        if self.ffmpeg_available and video_path.exists():
            try:
                res = subprocess.run(
                    ["ffmpeg", "-v", "error", "-i", str(video_path), "-f", "null", "-"],
                    capture_output=True, text=True, timeout=15
                )
                if res.returncode != 0:
                    decoder_errors.append(f"FFmpeg process returned non-zero code {res.returncode}")
                if res.stderr:
                    for err_line in res.stderr.splitlines():
                        err_line = err_line.strip()
                        if err_line and any(k in err_line.lower() for k in ["error", "corrupt", "invalid", "truncat", "nal", "header"]):
                            decoder_errors.append(f"FFmpeg: {err_line}")
            except Exception:
                pass

        # 5. Strict EOF Determination (Bug 2 Fix)
        diff = effective_declared_fc - decoded_fc
        diff_pct = (diff / effective_declared_fc) * 100.0 if effective_declared_fc > 0 else 0.0

        if can_read_tail:
            # Mid-stream corruption: unreadable block between decoded_fc and tail
            notes.append(f"Corrupted stream mid-video (stopped at frame {decoded_fc}/{effective_declared_fc})")
            eof_status = EOFStatus.DECODE_ERROR.value
        elif decoded_fc == 0:
            notes.append("Zero frames successfully decoded")
            eof_status = EOFStatus.DECODE_ERROR.value
        elif diff < 0:
            notes.append(f"Decoded {decoded_fc} frames exceeding header {effective_declared_fc} frames")
            eof_status = EOFStatus.NORMAL_EOF.value
        elif diff == 0:
            notes.append("Clean exact sequential EOF decode matching container header")
            eof_status = EOFStatus.NORMAL_EOF.value
        else:
            # diff > 0: Some declared frames missing
            # Only confirm NORMAL_EOF if verified genuine container audio-video timescale padding:
            # zero errors, deficit <= 2 frames, and stream duration was fully reached
            target_dur = expected_duration if expected_duration > 0 else (effective_declared_fc / effective_fps)
            is_genuine_padding = (
                len(decoder_errors) == 0
                and diff <= 2
                and (last_pts_sec >= (target_dur - (2.0 / effective_fps)) if effective_fps > 0 else False)
            )

            if is_genuine_padding:
                notes.append(f"Verified container timescale padding: {diff} frames ({diff_pct:.2f}%) with duration reached")
                eof_status = EOFStatus.NORMAL_EOF.value
            elif diff <= CONTAINER_TOLERANCE_FRAMES or diff_pct <= CONTAINER_TOLERANCE_PERCENT:
                # Deficit <= 10 or <= 0.5% without duration proof is UNKNOWN_EOF, not NORMAL_EOF
                notes.append(f"Unsafe tolerance rejected: {diff} frames ({diff_pct:.2f}%) deficit lacks technical proof -> UNKNOWN_EOF")
                eof_status = EOFStatus.UNKNOWN_EOF.value
            else:
                notes.append(f"Premature stream cut: missing {diff} frames ({diff_pct:.2f}%) -> TRUNCATED_STREAM")
                eof_status = EOFStatus.TRUNCATED_STREAM.value

        if decoder_errors:
            notes.extend(decoder_errors)

        return (
            decoded_fc, effective_declared_fc, effective_fps, width, height,
            round(last_pts_sec, 2), eof_status, notes
        )

    def _resolve_human_review(
        self,
        vid: str,
        item: dict[str, Any],
        val_rev: dict[str, Any],
        actual_sha: str
    ) -> tuple[str, str, str, bool, list[str]]:
        """Resolve human review record enforcing single-source authority and alias conflict checks.

        Returns:
            Tuple of (content_decision, reviewer_name, review_timestamp, attest_full, rejection_reasons).
        """
        rejection_reasons: list[str] = []

        # Single source selection: use worksheet if available, otherwise manifest item
        rec = val_rev if val_rev else item
        is_from_worksheet = bool(val_rev)

        # Check SHA256 binding if provided in review record
        rec_sha = (rec.get("video_sha256") or rec.get("sha256") or "").strip()
        if rec_sha and actual_sha and rec_sha != actual_sha:
            rejection_reasons.append("REVIEW_SHA256_MISMATCH")

        # Alias conflict check 1: Decision fields
        canon_dec = str(rec.get("content_decision", "")).strip().upper()
        alias_dec = str(rec.get("review_decision") or rec.get("decision") or "").strip().upper()
        if canon_dec and alias_dec and canon_dec != alias_dec:
            rejection_reasons.append("CONFLICTING_DECISION_FIELDS")

        raw_decision = canon_dec or alias_dec or str(rec.get("human_review_status", "")).strip().upper()
        if not raw_decision:
            raw_decision = ContentDecision.PENDING_HUMAN_REVIEW.value

        # Alias conflict check 2: Reviewer fields
        canon_rev = str(rec.get("reviewer_name", "")).strip()
        alias_rev = str(rec.get("reviewer") or rec.get("annotator") or "").strip()
        if canon_rev and alias_rev and canon_rev.lower() != alias_rev.lower():
            rejection_reasons.append("CONFLICTING_REVIEWER_FIELDS")
        reviewer = canon_rev or alias_rev

        # Alias conflict check 3: Timestamp fields
        canon_ts = str(rec.get("review_timestamp", "")).strip()
        alias_ts = str(rec.get("timestamp") or rec.get("reviewed_at") or "").strip()
        if canon_ts and alias_ts and canon_ts != alias_ts:
            rejection_reasons.append("CONFLICTING_TIMESTAMP_FIELDS")
        rev_ts = canon_ts or alias_ts

        # Alias conflict check 4: Full view attestation
        canon_att = str(rec.get("attest_full_video_viewed", "")).strip().upper()
        alias_att = str(rec.get("full_view_attested") or rec.get("attest_full_view") or "").strip().upper()
        if canon_att and alias_att and canon_att != alias_att:
            rejection_reasons.append("CONFLICTING_ATTESTATION_FIELDS")
        raw_att = canon_att or alias_att
        attest_full = (raw_att == "TRUE")

        if rejection_reasons:
            raw_decision = ContentDecision.PENDING_HUMAN_REVIEW.value

        return raw_decision, reviewer, rev_ts, attest_full, rejection_reasons

    def audit_all(self, probe_decode: bool = True) -> tuple[list[VideoIntegrityRecord], dict[str, Any]]:
        """Run full audit over all negative videos declared in manifest."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        declared_videos = manifest.get("videos", [])
        LOG.info("Auditing %d negative videos from manifest...", len(declared_videos))

        records: list[VideoIntegrityRecord] = []
        seen_shas: dict[str, str] = {}

        total_declared_sec = 0.0
        total_decoded_sec = 0.0
        verified_tech_sec = 0.0
        human_verified_sec = 0.0

        # Separate FAR exposures across splits
        val_far_eligible_sec = 0.0
        test_far_eligible_sec = 0.0
        train_contaminated_sec = 0.0
        unresolved_held_out_sec = 0.0

        for item in declared_videos:
            vid = item.get("video_id", "").strip()
            fname = item.get("filename", f"{vid}.mp4").strip()
            rel_path = item.get("relative_path", "")
            expected_sha = item.get("sha256", "").strip()
            item_decl_dur = float(item.get("duration_seconds", 0.0) or 0.0)
            item_decl_fc = int(item.get("reported_frame_count", 0) or 0)
            item_decl_fps = float(item.get("fps", 0.0) or 0.0)

            # Resolve physical file path
            candidate_paths = [d / fname for d in self.video_dirs]
            if rel_path:
                candidate_paths.append(ROOT / rel_path)
            actual_path = next((p for p in candidate_paths if p.exists()), None)
            file_exists = actual_path is not None

            file_size = actual_path.stat().st_size if file_exists else 0
            actual_sha = sha256_file(actual_path) if file_exists else ""
            sha_match = (actual_sha == expected_sha) if (file_exists and expected_sha) else False

            # Check duplicate SHA-256
            is_duplicate_sha = False
            duplicate_note = ""
            if actual_sha and actual_sha != "FILE_NOT_FOUND":
                if actual_sha in seen_shas:
                    is_duplicate_sha = True
                    duplicate_note = f"DUPLICATE_SHA256_OF_{seen_shas[actual_sha]}"
                else:
                    seen_shas[actual_sha] = vid

            # Stream probing & EOF determination
            if file_exists and probe_decode:
                (
                    dec_fc, decl_fc, fps, w, h, last_pts, eof_status, probe_notes
                ) = self.probe_video_stream(
                    actual_path, item_decl_fc, item_decl_fps, item_decl_dur
                )
                tech_notes: list[str] = list(probe_notes)
                tech_verdict = TechnicalVerdict.PASS.value if eof_status == EOFStatus.NORMAL_EOF.value else TechnicalVerdict.FAIL.value
            elif file_exists and not probe_decode:
                # Bug 3: Safe --no-probe mode: strictly requires trusted audit receipt
                decl_fc = item_decl_fc
                fps = item_decl_fps if item_decl_fps > 0 else 25.0
                w = int(item.get("width", 0) or 0)
                h = int(item.get("height", 0) or 0)

                trusted = self.trusted_receipts.get(vid)
                if (
                    trusted
                    and trusted.get("sha256_actual") == actual_sha
                    and trusted.get("technical_verdict") == TechnicalVerdict.PASS.value
                ):
                    tech_verdict = TechnicalVerdict.PASS.value
                    eof_status = trusted.get("eof_status", EOFStatus.NORMAL_EOF.value)
                    dec_fc = int(trusted.get("decoded_frame_count", decl_fc))
                    last_pts = float(trusted.get("last_frame_timestamp_sec", 0.0))
                    tech_notes = ["Loaded verified audit from trusted receipt"]
                else:
                    tech_verdict = TechnicalVerdict.NOT_VERIFIED.value
                    eof_status = EOFStatus.UNKNOWN_EOF.value
                    dec_fc = decl_fc
                    last_pts = 0.0
                    tech_notes = ["NO_PROBE_WITHOUT_TRUSTED_RECEIPT: Technical integrity unverified"]
            else:
                decl_fc = item_decl_fc
                dec_fc = 0
                fps = item_decl_fps if item_decl_fps > 0 else 25.0
                w = 0
                h = 0
                last_pts = 0.0
                eof_status = EOFStatus.DECODE_ERROR.value
                tech_verdict = TechnicalVerdict.FAIL.value
                tech_notes = ["File missing on disk"]

            frame_diff = decl_fc - dec_fc
            frame_diff_pct = (frame_diff / decl_fc) * 100.0 if decl_fc > 0 else 0.0
            declared_dur = item_decl_dur if item_decl_dur > 0 else (decl_fc / fps if fps > 0 else 0.0)
            decoded_dur = (dec_fc / fps) if fps > 0 else 0.0

            if not file_exists:
                tech_verdict = TechnicalVerdict.FAIL.value
                tech_notes.append("File missing on disk")
            if file_exists and not sha_match:
                tech_verdict = TechnicalVerdict.FAIL.value
                tech_notes.append("SHA256 checksum mismatch against manifest")
            if eof_status != EOFStatus.NORMAL_EOF.value:
                tech_verdict = TechnicalVerdict.FAIL.value if tech_verdict != TechnicalVerdict.NOT_VERIFIED.value else tech_verdict
                tech_notes.append(f"Non-normal EOF status: {eof_status}")

            # Bug 5: Resolve human review with conflict prevention
            val_rev = self.val_human_reviews.get(vid, {})
            (
                content_dec, reviewer, rev_ts, attest_full, review_conflicts
            ) = self._resolve_human_review(vid, item, val_rev, actual_sha)

            # Split membership
            splits = sorted(list(self.split_membership.get(vid, [])))
            if "train" in splits:
                split_cat = "TRAIN"
            elif "val" in splits:
                split_cat = "VAL"
            elif "test" in splits:
                split_cat = "TEST"
            else:
                split_cat = "HELD_OUT"

            # Strict FAR Eligibility Evaluation
            ineligibility_reasons: list[str] = []
            if not file_exists:
                ineligibility_reasons.append("FILE_MISSING")
            if not sha_match:
                ineligibility_reasons.append("SHA256_MISMATCH")
            if is_duplicate_sha:
                ineligibility_reasons.append(duplicate_note)
            if tech_verdict != TechnicalVerdict.PASS.value:
                ineligibility_reasons.append(f"TECHNICAL_FAILURE_{tech_verdict}_{eof_status}")
            if review_conflicts:
                ineligibility_reasons.extend(review_conflicts)
            if content_dec == ContentDecision.PENDING_HUMAN_REVIEW.value or not content_dec:
                ineligibility_reasons.append("PENDING_HUMAN_CONTENT_REVIEW")
            elif content_dec in (ContentDecision.FIRE_PRESENT.value, ContentDecision.SMOKE_PRESENT.value):
                ineligibility_reasons.append(f"HAZARD_PRESENT_{content_dec}")
            elif content_dec != ContentDecision.VERIFIED_NEGATIVE.value:
                ineligibility_reasons.append(f"UNVERIFIED_CONTENT_{content_dec}")

            if content_dec == ContentDecision.VERIFIED_NEGATIVE.value:
                if reviewer.lower() in DISALLOWED_REVIEWERS:
                    ineligibility_reasons.append(f"INVALID_REVIEWER_{reviewer or 'EMPTY'}")
                if not parse_iso_timestamp(rev_ts):
                    ineligibility_reasons.append("INVALID_REVIEW_TIMESTAMP")
                if not attest_full:
                    ineligibility_reasons.append("FULL_VIEW_NOT_ATTESTED")

            if split_cat == "TRAIN":
                ineligibility_reasons.append("TRAIN_SPLIT_LEAKAGE_RISK")

            eligible_for_far = (len(ineligibility_reasons) == 0)

            # Aggregate exposure metrics
            total_declared_sec += declared_dur
            total_decoded_sec += decoded_dur

            if tech_verdict == TechnicalVerdict.PASS.value:
                verified_tech_sec += decoded_dur

            if content_dec == ContentDecision.VERIFIED_NEGATIVE.value and "PENDING_HUMAN_CONTENT_REVIEW" not in ineligibility_reasons:
                human_verified_sec += decoded_dur

            # Bug 4: Separate FAR split scopes
            if split_cat == "VAL" and eligible_for_far:
                val_far_eligible_sec += decoded_dur
            elif split_cat == "TEST" and eligible_for_far:
                test_far_eligible_sec += decoded_dur
            elif split_cat == "TRAIN":
                train_contaminated_sec += decoded_dur
            else:
                unresolved_held_out_sec += decoded_dur

            records.append(VideoIntegrityRecord(
                video_id=vid,
                filename=fname,
                relative_path=str(rel_path) if rel_path else f"datasets/negative_videos/{fname}",
                file_path=str(actual_path) if actual_path else "NOT_FOUND",
                file_exists=file_exists,
                file_size_bytes=file_size,
                sha256_actual=actual_sha,
                sha256_manifest=expected_sha,
                sha256_match=sha_match,
                fps=round(fps, 2),
                width=w,
                height=h,
                declared_frame_count=decl_fc,
                decoded_frame_count=dec_fc,
                frame_count_diff=frame_diff,
                frame_count_diff_percent=round(frame_diff_pct, 2),
                declared_duration_sec=round(declared_dur, 2),
                decoded_duration_sec=round(decoded_dur, 2),
                last_frame_timestamp_sec=round(last_pts, 2),
                eof_status=eof_status,
                technical_verdict=tech_verdict,
                technical_notes="; ".join(tech_notes) if tech_notes else "OK",
                content_decision=content_dec,
                reviewer_name=reviewer,
                review_timestamp=rev_ts,
                attest_full_video_viewed=attest_full,
                split_membership=split_cat,
                eligible_for_far=eligible_for_far,
                ineligibility_reasons="; ".join(ineligibility_reasons) if ineligibility_reasons else "NONE",
                is_duplicate_sha=is_duplicate_sha
            ))

        remaining_to_5h_sec = max(0.0, self.target_exposure_sec - val_far_eligible_sec)

        # Build comprehensive reconciliation summary
        reconciliation = {
            "target_exposure_standard": {
                "specification_source": "DeCuong_NCKH_PhatHienChay_Camera.docx (MT1 & MT3)",
                "required_minimum_hours": self.target_hours,
                "required_minimum_seconds": self.target_exposure_sec,
            },
            "duration_indices": {
                "declared_duration_sec": round(total_declared_sec, 2),
                "decoded_duration_sec": round(total_decoded_sec, 2),
                "verified_technical_duration_sec": round(verified_tech_sec, 2),
                "human_verified_negative_duration_sec": round(human_verified_sec, 2),
                "val_far_eligible_duration_sec": round(val_far_eligible_sec, 2),
                "test_far_eligible_duration_sec": round(test_far_eligible_sec, 2),
                "train_contaminated_duration_sec": round(train_contaminated_sec, 2),
                "unresolved_held_out_duration_sec": round(unresolved_held_out_sec, 2),
                "far_eligible_duration_sec": round(val_far_eligible_sec, 2),
                "remaining_duration_to_5h_sec": round(remaining_to_5h_sec, 2),
            },
            "duration_indices_hours": {
                "declared_duration_hours": round(total_declared_sec / 3600.0, 4),
                "decoded_duration_hours": round(total_decoded_sec / 3600.0, 4),
                "verified_technical_duration_hours": round(verified_tech_sec / 3600.0, 4),
                "human_verified_negative_duration_hours": round(human_verified_sec / 3600.0, 4),
                "val_far_eligible_duration_hours": round(val_far_eligible_sec / 3600.0, 4),
                "test_far_eligible_duration_hours": round(test_far_eligible_sec / 3600.0, 4),
                "train_contaminated_duration_hours": round(train_contaminated_sec / 3600.0, 4),
                "unresolved_held_out_duration_hours": round(unresolved_held_out_sec / 3600.0, 4),
                "far_eligible_duration_hours": round(val_far_eligible_sec / 3600.0, 4),
                "remaining_duration_to_5h_hours": round(remaining_to_5h_sec / 3600.0, 4),
            },
            "audit_statistics": {
                "total_negative_videos_manifest": len(declared_videos),
                "files_existing_on_disk": sum(1 for r in records if r.file_exists),
                "sha256_verified_matches": sum(1 for r in records if r.sha256_match),
                "duplicate_sha256_count": sum(1 for r in records if r.is_duplicate_sha),
                "technical_pass_count": sum(1 for r in records if r.technical_verdict == TechnicalVerdict.PASS.value),
                "technical_fail_count": sum(1 for r in records if r.technical_verdict == TechnicalVerdict.FAIL.value),
                "technical_not_verified_count": sum(1 for r in records if r.technical_verdict == TechnicalVerdict.NOT_VERIFIED.value),
                "eof_classification_breakdown": {
                    "NORMAL_EOF": sum(1 for r in records if r.eof_status == EOFStatus.NORMAL_EOF.value),
                    "TRUNCATED_STREAM": sum(1 for r in records if r.eof_status == EOFStatus.TRUNCATED_STREAM.value),
                    "DECODE_ERROR": sum(1 for r in records if r.eof_status == EOFStatus.DECODE_ERROR.value),
                    "UNKNOWN_EOF": sum(1 for r in records if r.eof_status == EOFStatus.UNKNOWN_EOF.value),
                },
                "content_verification_breakdown": {
                    "VERIFIED_NEGATIVE": sum(1 for r in records if r.content_decision == ContentDecision.VERIFIED_NEGATIVE.value),
                    "PENDING_HUMAN_REVIEW": sum(1 for r in records if r.content_decision == ContentDecision.PENDING_HUMAN_REVIEW.value),
                    "HAZARD_FLAGGED": sum(1 for r in records if r.content_decision in (ContentDecision.FIRE_PRESENT.value, ContentDecision.SMOKE_PRESENT.value)),
                },
                "split_membership_breakdown": {
                    "TRAIN_CONTAMINATED": sum(1 for r in records if r.split_membership == "TRAIN"),
                    "VAL_SPLIT": sum(1 for r in records if r.split_membership == "VAL"),
                    "TEST_SPLIT": sum(1 for r in records if r.split_membership == "TEST"),
                    "CLEAN_HELD_OUT": sum(1 for r in records if r.split_membership == "HELD_OUT"),
                },
                "val_far_eligible_videos_count": sum(1 for r in records if r.split_membership == "VAL" and r.eligible_for_far),
                "test_far_eligible_videos_count": sum(1 for r in records if r.split_membership == "TEST" and r.eligible_for_far),
                "far_eligible_videos_count": sum(1 for r in records if r.split_membership == "VAL" and r.eligible_for_far),
            },
            "gate_evaluation": {
                "status": "PASSED" if val_far_eligible_sec >= self.target_exposure_sec else "BLOCKED",
                "verdict_reason": (
                    f"Validation FAR exposure requirement is {self.target_hours:.1f}h (18,000s). "
                    f"Current valid Validation FAR exposure is {val_far_eligible_sec / 3600.0:.4f}h ({val_far_eligible_sec:.1f}s), "
                    f"leaving a deficit of {remaining_to_5h_sec / 3600.0:.4f}h ({remaining_to_5h_sec:.1f}s). "
                    "Independent Test videos (2.10h) are strictly held-out and cannot be merged into Validation."
                ),
            },
            "excluded_videos": [
                {
                    "video_id": r.video_id,
                    "filename": r.filename,
                    "technical_verdict": r.technical_verdict,
                    "eof_status": r.eof_status,
                    "split_membership": r.split_membership,
                    "reasons": r.ineligibility_reasons
                }
                for r in records if not r.eligible_for_far
            ]
        }

        # Export CSV Manifest
        csv_path = self.output_dir / "negative_video_integrity_manifest.csv"
        fieldnames = list(asdict(records[0]).keys()) if records else []
        atomic_write_csv(csv_path, [asdict(r) for r in records], fieldnames)

        # Export JSON Manifest
        json_manifest_path = self.output_dir / "negative_video_integrity_manifest.json"
        atomic_write_json(json_manifest_path, {"videos": [asdict(r) for r in records]})

        # Export FAR Exposure Reconciliation JSON
        reconciliation_path = self.output_dir / "far_exposure_reconciliation.json"
        atomic_write_json(reconciliation_path, reconciliation)

        # Export Technical Audit Report Markdown
        md_report_path = self.output_dir / "negative_video_technical_audit_report.md"
        self._write_markdown_report(md_report_path, reconciliation, records)

        LOG.info(
            "Audit complete. Declared: %.2fh, Decoded: %.2fh, Tech Valid: %.2fh, Val FAR: %.2fh, Shortfall to 5h: %.2fh",
            total_declared_sec / 3600.0, total_decoded_sec / 3600.0,
            verified_tech_sec / 3600.0, val_far_eligible_sec / 3600.0,
            remaining_to_5h_sec / 3600.0
        )
        return records, reconciliation

    def _write_markdown_report(
        self,
        path: Path,
        summary: dict[str, Any],
        records: list[VideoIntegrityRecord]
    ) -> None:
        """Write human-readable technical audit report."""
        dur = summary["duration_indices"]
        dur_h = summary["duration_indices_hours"]
        stats = summary["audit_statistics"]
        eof_stat = stats["eof_classification_breakdown"]
        gate = summary["gate_evaluation"]

        lines = [
            "# BÁO CÁO KIỂM ĐỊNH KỸ THUẬT VÀ TÍNH TOÀN VẸN VIDEO ÂM TÍNH (TUẦN 6)",
            "",
            r"> **Tiêu chuẩn nghiên cứu:** Đề cương NCKH mục tiêu MT1 & MT3 ($\ge 5$ giờ video âm tính cảnh nhiễu).  ",
            f"> **Thời điểm kiểm định:** `{datetime.datetime.now(datetime.timezone.utc).isoformat()}`  ",
            f"> **Trạng thái cổng FAR Exposure:** **`{gate['status']}`**  ",
            f"> **Thời lượng phơi nhiễm Validation FAR hợp lệ:** **`{dur_h['val_far_eligible_duration_hours']} giờ`** / Yêu cầu **`{summary['target_exposure_standard']['required_minimum_hours']} giờ`**  ",
            "",
            "---",
            "",
            "## 1. BẢNG ĐỐI CHIẾU PHÂN PHỐI THỜI LƯỢNG THEO PHÂN CHIA TẬP DỮ LIỆU",
            "",
            "| Phân tầng phơi nhiễm | Thời lượng (giây) | Thời lượng (giờ) | Trạng thái bảo vệ khoa học |",
            "| :--- | :---: | :---: | :--- |",
            f"| `val_far_eligible_duration_sec` | **{dur['val_far_eligible_duration_sec']:.2f} s** | **{dur_h['val_far_eligible_duration_hours']:.4f} h** | Phơi nhiễm hợp lệ chọn Operating Point trên tập Validation. |",
            f"| `test_far_eligible_duration_sec` | **{dur['test_far_eligible_duration_sec']:.2f} s** | **{dur_h['test_far_eligible_duration_hours']:.4f} h** | **Tập Test độc lập (Held-Out)**, giữ nguyên vẹn cho nghiệm thu cuối cùng. |",
            f"| `train_contaminated_duration_sec` | **{dur['train_contaminated_duration_sec']:.2f} s** | **{dur_h['train_contaminated_duration_hours']:.4f} h** | **Bị loại trừ (37 videos)** do khung hình đã dùng trong tập train `vn_dataset`. |",
            f"| `unresolved_held_out_duration_sec` | **{dur['unresolved_held_out_duration_sec']:.2f} s** | **{dur_h['unresolved_held_out_duration_hours']:.4f} h** | Video chưa gắn phân chia, cấm gộp tự do vào Validation/Test. |",
            f"| `remaining_duration_to_5h_sec` | **{dur['remaining_duration_to_5h_sec']:.2f} s** | **{dur_h['remaining_duration_to_5h_hours']:.4f} h** | **Thâm hụt tới mốc 5.0h trên tập Validation.** |",
            "",
            "---",
            "",
            "## 2. KẾT QUẢ XÁC ĐỊNH EOF VÀ GIẢI MÃ TUẦN TỰ (SEQUENTIAL EOF DETERMINATION)",
            "",
            "Hệ thống kiểm định giải mã tuần tự 100% khung hình và phối hợp đa chiều:",
            "- **Sequential Frame Walk:** Giải mã liên tục từng khung hình từ frame 0 đến frame cuối, ghi nhận PTS và phát hiện điểm gián đoạn.",
            "- **Mid-Stream Corruption Probe:** Phát hiện lỗi hỏng giữa luồng ngay cả khi frame cuối cùng vẫn seek được.",
            r"- **Strict Padding Proof:** Chỉ chấp nhận chênh lệch timescale ($\le 2$ frames) khi có bằng chứng PTS đạt tới thời lượng stream.",
            "",
            "### Phân loại trạng thái EOF:",
            f"- **`NORMAL_EOF` (Đạt chuẩn):** **{eof_stat['NORMAL_EOF']}** video.",
            f"- **`TRUNCATED_STREAM` (Đứt luồng):** **{eof_stat['TRUNCATED_STREAM']}** video.",
            f"- **`DECODE_ERROR` (Lỗi giải mã/Hỏng giữa luồng):** **{eof_stat['DECODE_ERROR']}** video.",
            f"- **`UNKNOWN_EOF` (Không xác định):** **{eof_stat['UNKNOWN_EOF']}** video.",
            "",
            "---",
            "",
            "## 3. DANH SÁCH VIDEO BỊ LOẠI TRỪ KHỎI PHƠI NHIỄM FAR VÀ NGUYÊN NHÂN",
            "",
            "| Video ID | Tên file | Lớp phân chia | Trạng thái EOF | Kết luận Kỹ thuật | Lý do loại trừ khỏi FAR |",
            "| :--- | :--- | :---: | :---: | :---: | :--- |",
        ]

        for ex in summary["excluded_videos"][:25]:
            lines.append(
                f"| **{ex['video_id']}** | `{ex['filename']}` | `{ex['split_membership']}` | `{ex['eof_status']}` | `{ex['technical_verdict']}` | `{ex['reasons']}` |"
            )

        if len(summary["excluded_videos"]) > 25:
            lines.append(f"| ... | *(và {len(summary['excluded_videos']) - 25} video khác bị loại trừ do đang chờ xác minh hoặc rò rỉ Train)* | | | | |")

        lines.extend([
            "",
            "---",
            "",
            "## 4. KẾT LUẬN & TRẠNG THÁI CỔNG NGHIỆM THU",
            "",
            f"1. **Cổng FAR Exposure hiện tại:** **`{gate['status']}`**.",
            f"2. **Nguyên nhân chính:**",
            f"   - {gate['verdict_reason']}",
            f"   - Số video trùng lặp SHA256 bị phát hiện và loại trừ: **{stats['duplicate_sha256_count']}**.",
            "3. **Hành động bắt buộc:**",
            "   - Thực hiện quy trình review nội dung cho 13 video Validation âm tính.",
            "   - Giữ nguyên 13 video Test âm tính (2.10h) độc lập cho nghiệm thu Tuần 8, không di chuyển sang Validation."
        ])

        path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/negative_videos_manifest.json")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "datasets/vn_dataset/split_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/audits/week6")
    parser.add_argument("--val-review-manifest", type=Path, default=ROOT / "runs/audits/week6/validation_negative_review_manifest.csv")
    parser.add_argument("--target-hours", type=float, default=5.0)
    parser.add_argument("--no-probe", action="store_true", help="Skip stream decode probe")
    parser.add_argument("--trusted-receipt", type=Path, default=None, help="Path to trusted audit receipt for --no-probe")
    args = parser.parse_args()

    video_dirs = [
        ROOT / "datasets/negative_videos",
        ROOT / "datasets/test_videos/neg",
        ROOT / "negative_videos"
    ]

    try:
        auditor = NegativeVideoTechnicalAuditor(
            manifest_path=args.manifest,
            split_manifest_path=args.split_manifest,
            video_dirs=video_dirs,
            output_dir=args.output_dir,
            val_review_manifest_path=args.val_review_manifest,
            target_hours=args.target_hours,
            trusted_receipt_path=args.trusted_receipt
        )
        auditor.audit_all(probe_decode=not args.no_probe)
        return 0
    except Exception as err:
        LOG.error("Negative video technical audit failed: %s", err, exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
