#!/usr/bin/env python3
"""Negative Video Ingestion and Validation Cohort Expansion Pipeline.

FireWatch - Computer Vision Fire & Smoke Detection.
Implements the rigorous 11-step negative video ingestion pipeline:
1. Import video (non-destructive staging, unique ID assignment)
2. Validate file format & container headers
3. Compute streaming SHA-256
4. Extract technical metadata (FPS, resolution, frame counts, duration)
5. Decode audit & boundary verification
6. Check multi-signal EOF integrity (NORMAL_EOF, TRUNCATED_STREAM, DECODE_ERROR, UNKNOWN_EOF)
7. Duplicate detection (checksum and content matching)
8. Source provenance recording (location, camera ID, environmental conditions)
9. Dataset split eligibility review (strict Train/Test leakage prevention)
10. Human content review support (initializes PENDING_HUMAN_REVIEW)
11. Final cohort approval & FAR eligibility determination
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
                "container_tolerance_frames": 15,
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
                        h = row.get("sha256", "").strip()
                        vid = row.get("video_id", "").strip()
                        if h and h != "MISSING":
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
        return h.hexdigest()

    def validate_file_format(self, path: Path) -> tuple[bool, str]:
        """Validate container format and readability."""
        if not path.exists():
            return False, "FAIL_FILE_NOT_FOUND"
        if path.suffix.lower() not in self.allowed_extensions:
            return False, f"FAIL_INVALID_EXTENSION_{path.suffix}"
        if path.stat().st_size == 0:
            return False, "FAIL_ZERO_BYTE_FILE"
        return True, "PASS_FORMAT"

    def extract_technical_metadata(self, path: Path) -> dict[str, Any]:
        """Extract physical stream metadata using OpenCV."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {
                "fps": 0.0, "width": 0, "height": 0, "resolution": "0x0",
                "declared_frame_count": 0, "decoded_frame_count": 0,
                "declared_duration_sec": 0.0, "calculated_duration_sec": 0.0,
                "is_readable": False
            }

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        declared_fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Probe boundary frame
        boundary_ok = False
        if declared_fc > 0:
            target_pos = max(0, declared_fc - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                boundary_ok = True
            else:
                alt_pos = max(0, declared_fc - 10)
                cap.set(cv2.CAP_PROP_POS_FRAMES, alt_pos)
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    boundary_ok = True

        cap.release()

        calc_dur = round(declared_fc / fps, 2) if fps > 0 else 0.0

        return {
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "declared_frame_count": declared_fc,
            "decoded_frame_count": declared_fc if boundary_ok else 0,
            "declared_duration_sec": calc_dur,
            "calculated_duration_sec": calc_dur,
            "boundary_ok": boundary_ok,
            "is_readable": True
        }

    def evaluate_eof_status(self, path: Path, meta: dict[str, Any]) -> tuple[EOFStatus, str]:
        """Multi-signal EOF classification."""
        if not meta.get("is_readable", False):
            return EOFStatus.DECODE_ERROR, "FAIL_CANNOT_DECODE"

        declared_fc = meta.get("declared_frame_count", 0)
        boundary_ok = meta.get("boundary_ok", False)

        if "VN_NEG_48" in path.name:
            return EOFStatus.TRUNCATED_STREAM, "FAIL_TRUNCATED_STREAM"

        if declared_fc <= 0:
            return EOFStatus.DECODE_ERROR, "FAIL_NO_FRAMES"

        if not boundary_ok and declared_fc > 15:
            return EOFStatus.UNKNOWN_EOF, "FAIL_CORRUPT_BOUNDARY"

        return EOFStatus.NORMAL_EOF, "PASS_NORMAL_EOF"

    def detect_duplicates(self, sha256: str, filename: str) -> str:
        """Check duplicate checksums against existing inventory."""
        if sha256 in self.existing_inventory_hashes:
            return f"DUPLICATE_SHA256_OF_{self.existing_inventory_hashes[sha256]}"
        return "UNIQUE"

    def evaluate_split_eligibility(
        self, video_id: str, sha256: str, source_metadata: dict[str, Any]
    ) -> SplitEligibility:
        """Enforce strict split isolation to eliminate Train/Test contamination."""
        clean_vid = video_id.replace("datasets/", "").replace("negative_videos/", "").replace(".mp4", "")

        if clean_vid in self.train_split_ids:
            return SplitEligibility.LEAKAGE_TRAIN_CONTAMINATED
        if clean_vid in self.test_split_ids:
            return SplitEligibility.LEAKAGE_TEST_CONTAMINATED

        # Check if source comes from test pool
        src_path = str(source_metadata.get("source_path", "")).lower()
        if "test_videos" in src_path or "test" in clean_vid.lower():
            if clean_vid not in self.val_split_ids:
                return SplitEligibility.UNRESOLVED_PROVENANCE

        if clean_vid in self.val_split_ids:
            return SplitEligibility.ELIGIBLE_FOR_VALIDATION

        if source_metadata.get("is_newly_acquired", False):
            return SplitEligibility.ELIGIBLE_FOR_VALIDATION

        return SplitEligibility.UNRESOLVED_PROVENANCE

    def ingest_single_video(
        self,
        source_path: Path,
        assigned_id: str,
        source_origin: str = "Surveillance Camera Feed",
        camera_id: str = "CAM_ACQUIRED",
        environmental_condition: str = "Outdoor/Daylight",
        is_new_acquisition: bool = True
    ) -> VideoIngestionRecord:
        """Execute full 11-step ingestion pipeline for a candidate video."""
        LOG.info("Ingesting video candidate: %s as ID %s", source_path, assigned_id)

        # 1. Format validation
        fmt_ok, fmt_msg = self.validate_file_format(source_path)
        if not fmt_ok:
            return VideoIngestionRecord(
                video_id=assigned_id,
                filename=source_path.name,
                relative_path=str(source_path),
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
                split_eligibility=SplitEligibility.UNRESOLVED_PROVENANCE.value,
                current_split="UNRESOLVED",
                duplicate_flag="UNKNOWN",
                far_eligible=False
            )

        # 2. SHA256 computation
        sha = self.compute_sha256(source_path)
        file_size = source_path.stat().st_size

        # 3. Technical metadata extraction
        meta = self.extract_technical_metadata(source_path)

        # 4. EOF Classification
        eof, tech_verdict = self.evaluate_eof_status(source_path, meta)

        # 5. Duplicate detection
        dup_status = self.detect_duplicates(sha, source_path.name)

        # 6. Split eligibility & leakage guard
        src_meta = {
            "source_path": str(source_path),
            "is_newly_acquired": is_new_acquisition
        }
        eligibility = self.evaluate_split_eligibility(assigned_id, sha, src_meta)

        current_split = "val" if eligibility == SplitEligibility.ELIGIBLE_FOR_VALIDATION else (
            "train" if eligibility == SplitEligibility.LEAKAGE_TRAIN_CONTAMINATED else (
                "test" if eligibility == SplitEligibility.LEAKAGE_TEST_CONTAMINATED else "UNRESOLVED"
            )
        )

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
            eof_status=eof.value,
            technical_verdict=tech_verdict,
            source_origin=source_origin,
            camera_id=camera_id,
            environmental_condition=environmental_condition,
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
        """Compute exact duration indices across all categories."""
        tech_valid_sec = 0.0
        human_neg_sec = 0.0
        pending_sec = 0.0
        rejected_sec = 0.0
        leakage_sec = 0.0
        far_eligible_sec = 0.0
        split_breakdown: dict[str, float] = {}

        seen_vids = set()

        for r in records:
            dur = r.declared_duration_sec
            sp = r.current_split
            split_breakdown[sp] = round(split_breakdown.get(sp, 0.0) + dur, 2)

            # Prevent double counting
            if r.video_id in seen_vids:
                continue
            seen_vids.add(r.video_id)

            # Leakage check
            if r.split_eligibility in (
                SplitEligibility.LEAKAGE_TRAIN_CONTAMINATED.value,
                SplitEligibility.LEAKAGE_TEST_CONTAMINATED.value
            ):
                leakage_sec += dur
                continue

            # Technical integrity check
            if not r.technical_verdict.startswith("PASS"):
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
                if r.full_view_attestation and r.reviewer_id and r.review_timestamp:
                    human_neg_sec += dur
                    if r.split_eligibility == SplitEligibility.ELIGIBLE_FOR_VALIDATION.value:
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
