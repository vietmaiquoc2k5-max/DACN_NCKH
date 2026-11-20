#!/usr/bin/env python3
"""Negative Video Ingestion and Validation Cohort Expansion Pipeline.

FireWatch - Computer Vision Fire & Smoke Detection.
Scientific Integrity Hardened Pipeline:
1. Non-destructive import & unique ID assignment
2. Container format & header validation
3. Full streaming SHA-256 computation
4. Full stream sequential decode (NO boundary probe substitution)
5. Multi-signal EOF determination (NORMAL_EOF, TRUNCATED_STREAM, DECODE_ERROR, UNKNOWN_EOF)
6. Duplicate detection against existing inventory AND intra-batch candidates
7. Rigorous source provenance verification (camera ID, acquisition time, environment)
8. Strict split isolation to prevent Train/Test contamination
9. Human review initialization (PENDING_HUMAN_REVIEW)
10. Final cohort exposure calculation and gate eligibility determination
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import yaml

LOG = logging.getLogger("negative_ingestion")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent

DISALLOWED_PROVENANCE_VALUES = {
    "", "none", "null", "unknown", "n/a", "cam_acquired", "surveillance camera feed",
    "unassigned", "pending", "default", "temp"
}

DISALLOWED_REVIEWERS = {
    "", "none", "null", "system", "auto", "script", "bot", "unassigned",
    "n/a", "pending", "annotator_pending", "unknown", "reviewer_pending"
}


class EOFStatus(str, Enum):
    NORMAL_EOF = "NORMAL_EOF"
    TRUNCATED_STREAM = "TRUNCATED_STREAM"
    DECODE_ERROR = "DECODE_ERROR"
    UNKNOWN_EOF = "UNKNOWN_EOF"


class SplitEligibility(str, Enum):
    ELIGIBLE_FOR_VALIDATION = "ELIGIBLE_FOR_VALIDATION"
    LEAKAGE_TRAIN_CONTAMINATED = "LEAKAGE_TRAIN_CONTAMINATED"
    LEAKAGE_TEST_CONTAMINATED = "LEAKAGE_TEST_CONTAMINATED"
    UNRESOLVED_PROVENANCE = "UNRESOLVED_PROVENANCE"


class ContentDecision(str, Enum):
    PENDING_HUMAN_REVIEW = "PENDING_HUMAN_REVIEW"
    VERIFIED_NO_FIRE_SMOKE = "VERIFIED_NO_FIRE_SMOKE"
    FIRE_PRESENT = "FIRE_PRESENT"
    SMOKE_PRESENT = "SMOKE_PRESENT"
    UNCERTAIN = "UNCERTAIN"


def parse_iso_timestamp(ts_str: str) -> bool:
    """Validate whether a string is a valid ISO 8601 date or timestamp."""
    if not ts_str or not isinstance(ts_str, str):
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


def validate_provenance(source_metadata: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate that video provenance metadata meets strict scientific authenticity rules."""
    reasons: list[str] = []
    source_origin = str(source_metadata.get("source_origin") or "").strip()
    camera_id = str(source_metadata.get("camera_id") or "").strip()
    acq_time = str(source_metadata.get("acquisition_time") or source_metadata.get("collection_time") or "").strip()
    env = str(source_metadata.get("environmental_condition") or "").strip()

    if not source_origin or source_origin.lower() in DISALLOWED_PROVENANCE_VALUES:
        reasons.append(f"Invalid or missing source_origin ('{source_origin}')")
    if not camera_id or camera_id.lower() in DISALLOWED_PROVENANCE_VALUES:
        reasons.append(f"Invalid or missing camera_id ('{camera_id}')")
    if not acq_time or acq_time.lower() in DISALLOWED_PROVENANCE_VALUES:
        reasons.append(f"Invalid or missing acquisition_time ('{acq_time}')")
    elif not parse_iso_timestamp(acq_time):
        reasons.append(f"Invalid acquisition_time format '{acq_time}'. Must be valid ISO 8601 date/timestamp")
    if not env or env.lower() in DISALLOWED_PROVENANCE_VALUES:
        reasons.append(f"Invalid or missing environmental_condition ('{env}')")

    return len(reasons) == 0, reasons


@dataclass
class VideoIngestionRecord:
    """Comprehensive tracking record for an ingested negative video."""
    video_id: str
    filename: str
    relative_path: str
    sha256: str
    file_size_bytes: int
    fps: float
    resolution: str
    width: int
    height: int
    declared_frame_count: int
    decoded_frame_count: int
    declared_duration_sec: float
    calculated_duration_sec: float
    eof_status: str
    technical_verdict: str
    source_origin: str
    camera_id: str
    environmental_condition: str
    split_eligibility: str
    current_split: str
    duplicate_flag: str
    duration_diff_sec: float = 0.0
    has_duration_mismatch: bool = False
    acquisition_time: str = ""
    human_content_verdict: str = ContentDecision.PENDING_HUMAN_REVIEW.value
    reviewer_id: str = ""
    review_timestamp: str = ""
    full_view_attestation: bool = False
    review_notes: str = "Awaiting human review"
    far_eligible: bool = False


@dataclass
class ValidationExposureSummary:
    """Consolidated metrics across all negative video cohort categories."""
    target_required_sec: float
    target_required_hours: float
    technical_valid_duration_sec: float
    human_reviewed_negative_duration_sec: float
    pending_review_duration_sec: float
    rejected_duration_sec: float
    leakage_excluded_duration_sec: float
    far_eligible_duration_sec: float
    far_eligible_duration_hours: float
    remaining_shortfall_sec: float
    remaining_shortfall_hours: float
    gate_verdict: str
    phase_3_eligible: bool
    breakdown_by_split: dict[str, Any] = field(default_factory=dict)


class NegativeVideoIngestionPipeline:
    """End-to-end ingestion and governance pipeline for negative video assets."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path or (ROOT / "configs" / "negative_video_ingestion.yaml")
        self.config = self._load_config(self.config_path)

        self.staging_dir = ROOT / self.config["storage"]["incoming_staging_dir"]
        self.dest_dir = ROOT / self.config["storage"]["negative_videos_dir"]
        self.audit_out_dir = ROOT / self.config["storage"]["audit_output_dir"]

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.audit_out_dir.mkdir(parents=True, exist_ok=True)

        self.allowed_extensions = set(self.config["validation"]["allowed_extensions"])
        self.target_exposure_sec = float(self.config["validation"]["target_exposure_seconds"])

        # Load existing inventory hashes and split memberships for leakage check
        self.existing_inventory_hashes = self._load_existing_hashes()
        self.train_split_ids, self.val_split_ids, self.test_split_ids = self._load_split_ids()

        # Intra-batch duplicate tracking: sha256 -> video_id
        self.batch_hashes: dict[str, str] = {}

    def _load_config(self, path: Path) -> dict[str, Any]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {
            "storage": {
                "incoming_staging_dir": "datasets/incoming_staging",
                "negative_videos_dir": "datasets/negative_videos",
                "audit_output_dir": "runs/audits/week6",
            },
            "validation": {
                "target_exposure_seconds": 18000.0,
                "allowed_extensions": [".mp4", ".avi", ".mkv", ".mov"],
                "container_tolerance_frames": 10,
            },
            "manifests": {
                "negative_inventory_csv": "runs/audits/week6/negative_video_inventory.csv",
                "split_manifest_csv": "datasets/vn_dataset/split_manifest.csv",
                "ingestion_manifest_csv": "runs/audits/week6/negative_video_ingestion_manifest.csv",
                "exposure_status_json": "runs/audits/week6/validation_exposure_status.json",
            }
        }

    def _load_existing_hashes(self) -> dict[str, str]:
        """Load known SHA256 hashes from existing negative video inventory."""
        hashes: dict[str, str] = {}
        inv_path = ROOT / self.config.get("manifests", {}).get("negative_inventory_csv", "runs/audits/week6/negative_video_inventory.csv")
        if inv_path.exists():
            try:
                with open(inv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        h = row.get("sha256", "").strip().lower()
                        vid = row.get("video_id", "").strip()
                        if h and h != "missing" and len(h) >= 8:
                            hashes[h] = vid
            except Exception as e:
                LOG.warning("Could not read existing inventory hashes: %s", e)
        return hashes

    def _load_split_ids(self) -> tuple[set[str], set[str], set[str]]:
        """Load Train, Val, Test memberships from split manifest."""
        train_ids: set[str] = set()
        val_ids: set[str] = set()
        test_ids: set[str] = set()
        split_path = ROOT / self.config.get("manifests", {}).get("split_manifest_csv", "datasets/vn_dataset/split_manifest.csv")
        if split_path.exists():
            try:
                with open(split_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        vid = row.get("video_id", "").strip()
                        sp = row.get("split", "").strip().lower()
                        if sp == "train":
                            train_ids.add(vid)
                        elif sp == "val":
                            val_ids.add(vid)
                        elif sp == "test":
                            test_ids.add(vid)
            except Exception as e:
                LOG.warning("Could not load split manifest: %s", e)
        return train_ids, val_ids, test_ids

    @staticmethod
    def compute_sha256(path: Path) -> str:
        """Stream SHA-256 computation in 64KB blocks."""
        if not path.exists():
            return "FILE_NOT_FOUND"
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest().lower()

    def validate_file_format(self, path: Path) -> tuple[bool, str]:
        """Validate container format and readability."""
        if not path.exists():
            return False, "FAIL_FILE_NOT_FOUND"
        if path.suffix.lower() not in self.allowed_extensions:
            return False, f"FAIL_INVALID_EXTENSION_{path.suffix}"
        if path.stat().st_size == 0:
            return False, "FAIL_ZERO_BYTE_FILE"
        return True, "PASS_FORMAT"

    def decode_full_stream(
        self, path: Path, expected_duration_sec: float | None = None
    ) -> dict[str, Any]:
        """Full stream sequential decode: reads frame-by-frame without seek substitution.

        Performs:
        - Exact sequential frame counting
        - Detection of premature stream termination (TRUNCATED_STREAM)
        - Detection of mid-stream corruptions
        - Reconciliation of declared frames, decoded frames, and duration
        """
        if not path.exists():
            return {
                "fps": 0.0, "width": 0, "height": 0, "resolution": "0x0",
                "declared_frame_count": 0, "decoded_frame_count": 0,
                "declared_duration_sec": 0.0, "calculated_duration_sec": 0.0,
                "duration_diff_sec": 0.0,
                "has_duration_mismatch": False,
                "eof_status": EOFStatus.DECODE_ERROR.value,
                "technical_verdict": "FAIL_FILE_NOT_FOUND",
                "is_readable": False, "notes": ["File does not exist on disk"]
            }

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {
                "fps": 0.0, "width": 0, "height": 0, "resolution": "0x0",
                "declared_frame_count": 0, "decoded_frame_count": 0,
                "declared_duration_sec": 0.0, "calculated_duration_sec": 0.0,
                "duration_diff_sec": 0.0,
                "has_duration_mismatch": False,
                "eof_status": EOFStatus.DECODE_ERROR.value,
                "technical_verdict": "FAIL_CANNOT_OPEN_STREAM",
                "is_readable": False, "notes": ["OpenCV could not open stream"]
            }

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        declared_fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = round(fps, 2) if fps > 0.0 else 25.0
        header_dur = round(declared_fc / fps, 2) if (declared_fc > 0 and fps > 0) else 0.0

        if expected_duration_sec is not None and expected_duration_sec > 0:
            declared_dur = round(expected_duration_sec, 2)
        else:
            declared_dur = header_dur

        if declared_fc <= 0 or width <= 0 or height <= 0:
            cap.release()
            return {
                "fps": fps, "width": width, "height": height, "resolution": f"{width}x{height}",
                "declared_frame_count": declared_fc, "decoded_frame_count": 0,
                "declared_duration_sec": declared_dur, "calculated_duration_sec": 0.0,
                "duration_diff_sec": declared_dur,
                "has_duration_mismatch": declared_dur > 0,
                "eof_status": EOFStatus.DECODE_ERROR.value,
                "technical_verdict": "FAIL_INVALID_HEADER_METADATA",
                "is_readable": False, "notes": ["Declared frame count or resolution non-positive"]
            }

        # Sequential full-stream decoding: read every frame until EOF
        decoded_fc = 0
        decoder_errors: list[str] = []

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame.size == 0:
                decoder_errors.append(f"Empty decoded frame encountered at index {decoded_fc}")
                break
            decoded_fc += 1

        # Check for mid-stream corruption: if sequential read stopped early, can tail be sought?
        can_read_tail = False
        if decoded_fc < declared_fc:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, declared_fc - 1))
            ret_tail, frame_tail = cap.read()
            if ret_tail and frame_tail is not None and frame_tail.size > 0:
                can_read_tail = True
                decoder_errors.append(
                    f"Mid-stream corruption: sequential decode stopped at {decoded_fc}/{declared_fc}, "
                    f"yet tail frame {declared_fc - 1} was reachable via seek."
                )

        cap.release()

        actual_dur = round(decoded_fc / fps, 2) if fps > 0 else 0.0
        diff = declared_fc - decoded_fc
        diff_pct = (diff / declared_fc) * 100.0 if declared_fc > 0 else 0.0

        notes = list(decoder_errors)

        # Strict EOF evaluation (independent of duration mismatch)
        if can_read_tail:
            eof_status = EOFStatus.DECODE_ERROR.value
            tech_verdict = f"FAIL_MID_STREAM_CORRUPTION_AT_FRAME_{decoded_fc}"
        elif decoded_fc == 0:
            eof_status = EOFStatus.DECODE_ERROR.value
            tech_verdict = "FAIL_ZERO_FRAMES_DECODED"
        elif diff < 0:
            eof_status = EOFStatus.NORMAL_EOF.value
            tech_verdict = "PASS_NORMAL_EOF_EXCEEDS_HEADER"
        elif diff == 0:
            eof_status = EOFStatus.NORMAL_EOF.value
            tech_verdict = "PASS_NORMAL_EOF"
        else:
            # diff > 0: Some declared frames missing
            # Only confirm NORMAL_EOF if verified genuine container timescale padding:
            is_genuine_padding = (
                len(decoder_errors) == 0
                and diff <= 2
                and abs(header_dur - actual_dur) <= (2.0 / fps)
            )
            if is_genuine_padding:
                eof_status = EOFStatus.NORMAL_EOF.value
                tech_verdict = "PASS_NORMAL_EOF_CONTAINER_PADDING"
            elif diff <= 10 or diff_pct <= 0.5:
                eof_status = EOFStatus.UNKNOWN_EOF.value
                tech_verdict = f"FAIL_UNKNOWN_EOF_MISSING_{diff}_FRAMES"
                notes.append(f"Insufficient proof of normal EOF: missing {diff} frames ({diff_pct:.2f}%)")
            else:
                eof_status = EOFStatus.TRUNCATED_STREAM.value
                tech_verdict = f"FAIL_TRUNCATED_STREAM_MISSING_{diff}_FRAMES"
                notes.append(f"Premature stream cut: missing {diff}/{declared_fc} frames ({diff_pct:.2f}%)")

        # Independent duration mismatch check
        duration_diff = round(abs(declared_dur - actual_dur), 2)
        has_duration_mismatch = (duration_diff > 1.0)
        if has_duration_mismatch:
            notes.append(
                f"Duration mismatch: declared {declared_dur:.2f}s vs decoded {actual_dur:.2f}s "
                f"(delta: {duration_diff:.2f}s)"
            )
            if tech_verdict.startswith("PASS"):
                tech_verdict = f"FAIL_DURATION_MISMATCH_{declared_dur}s_vs_{actual_dur}s"
            else:
                tech_verdict = f"{tech_verdict}; DURATION_MISMATCH_{declared_dur}s_vs_{actual_dur}s"

        return {
            "fps": fps,
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "declared_frame_count": declared_fc,
            "decoded_frame_count": decoded_fc,
            "declared_duration_sec": declared_dur,
            "calculated_duration_sec": actual_dur,
            "duration_diff_sec": duration_diff,
            "has_duration_mismatch": has_duration_mismatch,
            "eof_status": eof_status,
            "technical_verdict": tech_verdict,
            "is_readable": decoded_fc > 0,
            "notes": notes,
            "diff_frames": diff,
            "diff_pct": round(diff_pct, 2)
        }

    def extract_technical_metadata(self, path: Path) -> dict[str, Any]:
        """Backward-compatible wrapper that runs full sequential stream decode."""
        return self.decode_full_stream(path)

    def evaluate_eof_status(self, path: Path, meta: dict[str, Any]) -> tuple[EOFStatus, str]:
        """Multi-signal EOF classification based on full decode results."""
        if not meta.get("is_readable", False):
            return EOFStatus.DECODE_ERROR, meta.get("technical_verdict", "FAIL_DECODE_ERROR")

        eof_str = meta.get("eof_status", EOFStatus.UNKNOWN_EOF.value)
        verdict = meta.get("technical_verdict", "FAIL_UNKNOWN_EOF")

        try:
            return EOFStatus(eof_str), verdict
        except ValueError:
            return EOFStatus.UNKNOWN_EOF, verdict

    def detect_duplicates(
        self, sha256: str, filename: str = "", video_id: str = ""
    ) -> str:
        """Check duplicate checksums against existing inventory AND intra-batch items."""
        clean_sha = sha256.strip().lower()
        if not clean_sha or clean_sha in ("file_not_found", "n/a", ""):
            return "UNKNOWN"

        # 1. Check against existing inventory
        if clean_sha in self.existing_inventory_hashes:
            existing_vid = self.existing_inventory_hashes[clean_sha]
            return f"DUPLICATE_SHA256_OF_{existing_vid}"

        # 2. Check within current ingestion batch
        if clean_sha in self.batch_hashes:
            batch_prev_vid = self.batch_hashes[clean_sha]
            return f"DUPLICATE_BATCH_OF_{batch_prev_vid}"

        # Register hash in batch lookup
        self.batch_hashes[clean_sha] = video_id or filename
        return "UNIQUE"

    def evaluate_split_eligibility(
        self, video_id: str, sha256: str, source_metadata: dict[str, Any]
    ) -> SplitEligibility:
        """Enforce strict split isolation, Train/Test leakage guard, and provenance check."""
        clean_vid = video_id.replace("datasets/", "").replace("negative_videos/", "").replace(".mp4", "")

        # 1. Train leakage guard
        if clean_vid in self.train_split_ids:
            return SplitEligibility.LEAKAGE_TRAIN_CONTAMINATED

        # 2. Test leakage guard
        if clean_vid in self.test_split_ids:
            return SplitEligibility.LEAKAGE_TEST_CONTAMINATED

        # 3. Test pool path guard
        src_path = str(source_metadata.get("source_path", "")).lower()
        if "test_videos" in src_path or "test" in clean_vid.lower():
            if clean_vid not in self.val_split_ids:
                return SplitEligibility.UNRESOLVED_PROVENANCE

        # 4. Known official validation membership
        if clean_vid in self.val_split_ids:
            return SplitEligibility.ELIGIBLE_FOR_VALIDATION

        # 5. Candidate for new acquisition: strictly verify provenance and explicit flag
        is_new = bool(source_metadata.get("is_new_acquisition") or source_metadata.get("is_newly_acquired", False))
        if not is_new:
            return SplitEligibility.UNRESOLVED_PROVENANCE

        prov_ok, _ = validate_provenance(source_metadata)
        if not prov_ok:
            return SplitEligibility.UNRESOLVED_PROVENANCE

        return SplitEligibility.ELIGIBLE_FOR_VALIDATION

    def ingest_single_video(
        self,
        source_path: Path,
        assigned_id: str,
        source_origin: str = "",
        camera_id: str = "",
        environmental_condition: str = "",
        acquisition_time: str = "",
        declared_duration: float | None = None,
        is_new_acquisition: bool = False
    ) -> VideoIngestionRecord:
        """Execute full 11-step ingestion pipeline for a candidate video."""
        LOG.info("Ingesting video candidate: %s as ID %s", source_path, assigned_id)

        # 1. Format validation
        fmt_ok, fmt_msg = self.validate_file_format(source_path)
        if not fmt_ok:
            return VideoIngestionRecord(
                video_id=assigned_id,
                filename=source_path.name,
                relative_path=str(source_path).replace("\\", "/"),
                sha256="N/A",
                file_size_bytes=0,
                fps=0.0,
                resolution="0x0",
                width=0,
                height=0,
                declared_frame_count=0,
                decoded_frame_count=0,
                declared_duration_sec=0.0,
                calculated_duration_sec=0.0,
                eof_status=EOFStatus.DECODE_ERROR.value,
                technical_verdict=fmt_msg,
                source_origin=source_origin,
                camera_id=camera_id,
                environmental_condition=environmental_condition,
                acquisition_time=acquisition_time,
                split_eligibility=SplitEligibility.UNRESOLVED_PROVENANCE.value,
                current_split="UNRESOLVED",
                duplicate_flag="UNKNOWN",
                far_eligible=False
            )

        # 2. SHA256 computation
        sha = self.compute_sha256(source_path)
        file_size = source_path.stat().st_size

        # 3. Duplicate detection (inventory & batch)
        dup_status = self.detect_duplicates(sha, filename=source_path.name, video_id=assigned_id)

        # 4. Full stream sequential decode (NO boundary probe)
        meta = self.decode_full_stream(source_path, expected_duration_sec=declared_duration)

        # 5. Provenance & split eligibility
        src_meta = {
            "source_path": str(source_path),
            "source_origin": source_origin,
            "camera_id": camera_id,
            "environmental_condition": environmental_condition,
            "acquisition_time": acquisition_time,
            "is_new_acquisition": is_new_acquisition,
        }
        eligibility = self.evaluate_split_eligibility(assigned_id, sha, src_meta)

        current_split = "val" if eligibility == SplitEligibility.ELIGIBLE_FOR_VALIDATION else (
            "train" if eligibility == SplitEligibility.LEAKAGE_TRAIN_CONTAMINATED else (
                "test" if eligibility == SplitEligibility.LEAKAGE_TEST_CONTAMINATED else "UNRESOLVED"
            )
        )

        tech_verdict = meta["technical_verdict"]
        if dup_status != "UNIQUE":
            tech_verdict = f"{tech_verdict}; {dup_status}"

        return VideoIngestionRecord(
            video_id=assigned_id,
            filename=source_path.name,
            relative_path=str(source_path).replace("\\", "/"),
            sha256=sha,
            file_size_bytes=file_size,
            fps=meta["fps"],
            resolution=meta["resolution"],
            width=meta["width"],
            height=meta["height"],
            declared_frame_count=meta["declared_frame_count"],
            decoded_frame_count=meta["decoded_frame_count"],
            declared_duration_sec=meta["declared_duration_sec"],
            calculated_duration_sec=meta["calculated_duration_sec"],
            duration_diff_sec=meta.get("duration_diff_sec", 0.0),
            has_duration_mismatch=meta.get("has_duration_mismatch", False),
            eof_status=meta["eof_status"],
            technical_verdict=tech_verdict,
            source_origin=source_origin,
            camera_id=camera_id,
            environmental_condition=environmental_condition,
            acquisition_time=acquisition_time,
            split_eligibility=eligibility.value,
            current_split=current_split,
            duplicate_flag=dup_status,
            human_content_verdict=ContentDecision.PENDING_HUMAN_REVIEW.value,
            reviewer_id="",
            review_timestamp="",
            full_view_attestation=False,
            review_notes="Candidate ingested; awaits independent full video review",
            far_eligible=False
        )

    def calculate_cohort_exposure_summary(
        self, records: list[VideoIngestionRecord]
    ) -> ValidationExposureSummary:
        """Compute exact duration indices across all categories with strict duplicate exclusion."""
        tech_valid_sec = 0.0
        human_neg_sec = 0.0
        pending_sec = 0.0
        rejected_sec = 0.0
        leakage_sec = 0.0
        far_eligible_sec = 0.0
        split_breakdown: dict[str, float] = {}

        seen_vids: set[str] = set()
        seen_shas: set[str] = set()

        for r in records:
            dur = r.declared_duration_sec
            sp = r.current_split
            split_breakdown[sp] = round(split_breakdown.get(sp, 0.0) + dur, 2)

            # Prevent double counting by ID or SHA-256
            if r.video_id in seen_vids or (r.sha256 in seen_shas and r.sha256 not in ("N/A", "FILE_NOT_FOUND", "")):
                rejected_sec += dur
                continue
            seen_vids.add(r.video_id)
            if r.sha256 and r.sha256 not in ("N/A", "FILE_NOT_FOUND", ""):
                seen_shas.add(r.sha256)

            # Duplicate check
            if r.duplicate_flag != "UNIQUE":
                rejected_sec += dur
                continue

            # Leakage & provenance check
            if r.split_eligibility in (
                SplitEligibility.LEAKAGE_TRAIN_CONTAMINATED.value,
                SplitEligibility.LEAKAGE_TEST_CONTAMINATED.value,
                SplitEligibility.UNRESOLVED_PROVENANCE.value,
            ):
                if r.split_eligibility == SplitEligibility.UNRESOLVED_PROVENANCE.value:
                    rejected_sec += dur
                else:
                    leakage_sec += dur
                continue

            # Technical integrity check
            if not r.technical_verdict.startswith("PASS") or r.eof_status != EOFStatus.NORMAL_EOF.value:
                rejected_sec += dur
                continue

            tech_valid_sec += dur

            # Human review check
            if r.human_content_verdict == ContentDecision.PENDING_HUMAN_REVIEW.value:
                pending_sec += dur
            elif r.human_content_verdict in (
                ContentDecision.FIRE_PRESENT.value,
                ContentDecision.SMOKE_PRESENT.value,
                ContentDecision.UNCERTAIN.value
            ):
                rejected_sec += dur
            elif r.human_content_verdict == ContentDecision.VERIFIED_NO_FIRE_SMOKE.value:
                if (
                    r.full_view_attestation
                    and r.reviewer_id
                    and r.reviewer_id.lower() not in DISALLOWED_REVIEWERS
                    and r.review_timestamp
                    and r.split_eligibility == SplitEligibility.ELIGIBLE_FOR_VALIDATION.value
                ):
                    human_neg_sec += dur
                    far_eligible_sec += dur
                else:
                    pending_sec += dur

        shortfall = max(0.0, self.target_exposure_sec - far_eligible_sec)
        gate_status = "PASSED" if (far_eligible_sec >= self.target_exposure_sec and pending_sec == 0.0) else "BLOCKED"

        return ValidationExposureSummary(
            target_required_sec=self.target_exposure_sec,
            target_required_hours=round(self.target_exposure_sec / 3600.0, 2),
            technical_valid_duration_sec=round(tech_valid_sec, 2),
            human_reviewed_negative_duration_sec=round(human_neg_sec, 2),
            pending_review_duration_sec=round(pending_sec, 2),
            rejected_duration_sec=round(rejected_sec, 2),
            leakage_excluded_duration_sec=round(leakage_sec, 2),
            far_eligible_duration_sec=round(far_eligible_sec, 2),
            far_eligible_duration_hours=round(far_eligible_sec / 3600.0, 4),
            remaining_shortfall_sec=round(shortfall, 2),
            remaining_shortfall_hours=round(shortfall / 3600.0, 4),
            gate_verdict=gate_status,
            phase_3_eligible=(gate_status == "PASSED"),
            breakdown_by_split=split_breakdown
        )

    def export_ingestion_manifest(
        self, records: list[VideoIngestionRecord], out_path: Path
    ) -> None:
        """Export candidate review manifest in CSV format."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "video_id", "filename", "relative_path", "sha256", "duration_sec",
            "source", "split_status", "technical_status", "human_content_verdict",
            "reviewer_id", "review_timestamp", "full_view_attestation", "review_notes"
        ]
        rows = []
        for r in records:
            rows.append({
                "video_id": r.video_id,
                "filename": r.filename,
                "relative_path": r.relative_path,
                "sha256": r.sha256,
                "duration_sec": r.declared_duration_sec,
                "source": r.source_origin,
                "split_status": r.current_split,
                "technical_status": r.technical_verdict,
                "human_content_verdict": r.human_content_verdict,
                "reviewer_id": r.reviewer_id,
                "review_timestamp": r.review_timestamp,
                "full_view_attestation": "TRUE" if r.full_view_attestation else "FALSE",
                "review_notes": r.review_notes
            })

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        LOG.info("Saved ingestion manifest with %d rows to %s", len(rows), out_path)

    def export_exposure_status_json(
        self, summary: ValidationExposureSummary, out_path: Path
    ) -> None:
        """Export exposure status to JSON."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2, ensure_ascii=False)
        LOG.info("Saved exposure status JSON to %s", out_path)
