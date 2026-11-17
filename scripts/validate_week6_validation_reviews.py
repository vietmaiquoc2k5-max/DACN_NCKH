#!/usr/bin/env python3
"""Validation and Reconciliation Tool for Week 6 Validation Reviews.

Audits and reconciles:
1. Positive Ground Truth Review Worksheet (runs/audits/week6/validation_gt_review_worksheet.csv):
   - Enforces double-blind protocol (no annotator 1 timestamps in worksheet).
   - Enforces dual-human rule (annotator_01 != annotator_02).
   - Rejects non-human names (SYSTEM, AUTO, NONE, etc.).
   - Compares timestamps with SOP tolerance |start_1 - start_2| <= 0.5s.
   - Enforces Validation Gate blocking until all 5 events achieve AGREED_CONSENSUS.

2. Validation Negative Video Review Manifest (runs/audits/week6/validation_negative_review_manifest.csv):
   - Enforces exact 13 Validation videos matching split provenance.
   - Enforces SHA-256 integrity (mismatch blocks gate).
   - Rejects technically defective videos (e.g. TRUNCATED_STREAM) from FAR exposure.
   - Requires genuine human reviewer and full view attestation for VERIFIED_NEGATIVE.

Outputs:
- runs/audits/week6/validation_review_summary.json
- runs/audits/week6/validation_gt_reconciliation_report.md
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOG = logging.getLogger("validation_review_validator")

TARGET_POSITIVE_EVENTS = {
    "VAL_POS_07_FIRE": {"video_id": "POS_07", "label": "FIRE", "source_line": 11},
    "VAL_POS_08_FIRE": {"video_id": "POS_08", "label": "FIRE", "source_line": 12},
    "VAL_POS_08_SMOKE": {"video_id": "POS_08", "label": "SMOKE", "source_line": 13},
    "VAL_POS_17_SMOKE": {"video_id": "POS_17", "label": "SMOKE", "source_line": 27},
    "VAL_POS_18_SMOKE": {"video_id": "POS_18", "label": "SMOKE", "source_line": 28},
}

EXPECTED_VAL_NEG_VIDEOS = {
    "VN_NEG_03", "VN_NEG_42", "VN_NEG_43", "VN_NEG_44", "VN_NEG_45",
    "VN_NEG_46", "VN_NEG_47", "VN_NEG_48", "VN_NEG_49", "VN_NEG_50",
    "VN_NEG_51", "VN_NEG_52", "VN_NEG_53"
}

DISALLOWED_REVIEWERS = {
    "", "none", "null", "system", "auto", "script", "bot", "unassigned", "n/a", "annotator_pending"
}

SOP_ONSET_TOLERANCE_SEC = 0.5


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
    """Atomically write JSON data to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(dir=str(path.parent), prefix="tmp_sum_", suffix=".json")
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def load_original_gt_timestamps(events_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load ground truth Annotator 1 timestamps from events.jsonl.

    Args:
        events_path: Path to events.jsonl.

    Returns:
        Dict mapping (clean_video_id, label) to event details.
    """
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    if not events_path.exists():
        return mapping

    with open(events_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            vid = item.get("video_id", "").replace("VN_", "")
            lbl = (item.get("label") or "").upper()
            s1 = float(item.get("start_sec_annotator_01") if item.get("start_sec_annotator_01") is not None else item.get("start_sec", 0.0))
            e1 = float(item.get("end_sec_annotator_01") if item.get("end_sec_annotator_01") is not None else item.get("end_sec", 0.0))
            ann1 = item.get("annotator_01") or item.get("annotator_id", "annotator_01")

            mapping[(vid, lbl)] = {
                "line_no": line_no,
                "video_id": vid,
                "label": lbl,
                "start_sec_annotator_01": s1,
                "end_sec_annotator_01": e1,
                "annotator_01_id": ann1,
            }
    return mapping


def validate_positive_gt_worksheet(
    worksheet_csv: Path,
    events_jsonl_path: Path
) -> dict[str, Any]:
    """Audit Annotator 2 review worksheet and perform reconciliation with Annotator 1.

    Args:
        worksheet_csv: Path to validation_gt_review_worksheet.csv.
        events_jsonl_path: Path to events.jsonl.

    Returns:
        Audit report dictionary for positive events.
    """
    if not worksheet_csv.exists():
        raise FileNotFoundError(f"GT worksheet not found: {worksheet_csv}")

    orig_events = load_original_gt_timestamps(events_jsonl_path)

    with open(worksheet_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    # Invariant check: Annotator 1 timestamps MUST NOT be exposed in this worksheet!
    forbidden_headers = {"start_sec_annotator_01", "end_sec_annotator_01", "start_sec", "end_sec"}
    exposed_headers = forbidden_headers.intersection(headers)
    blind_protocol_intact = len(exposed_headers) == 0

    reconciliation_records = []
    seen_event_ids = set()
    agreed_count = 0
    disputed_count = 0
    pending_count = 0
    invalid_count = 0
    errors: list[str] = []

    if not blind_protocol_intact:
        errors.append(f"BLIND_PROTOCOL_VIOLATION: Worksheet exposes Annotator 1 timestamps: {list(exposed_headers)}")

    for row in rows:
        eid = row.get("event_id", "").strip()
        vid = row.get("video_id", "").strip()
        lbl = row.get("target_label", "").strip().upper()
        ann1_id = row.get("annotator_01_id", "").strip()
        ann2_id = row.get("annotator_02_id", "").strip()
        decision = row.get("annotator_02_decision", "").strip().upper()
        attest_full = row.get("attest_full_view", "").strip().upper() == "TRUE"
        attest_indep = row.get("attest_independent", "").strip().upper() == "TRUE"
        review_ts = row.get("review_timestamp", "").strip()
        notes = row.get("review_notes", "").strip()

        if eid in seen_event_ids:
            errors.append(f"DUPLICATE_EVENT_ID: {eid} appears multiple times")
            invalid_count += 1
            continue
        seen_event_ids.add(eid)

        if eid not in TARGET_POSITIVE_EVENTS:
            errors.append(f"UNRECOGNIZED_EVENT_ID: {eid} is not a valid validation event")
            invalid_count += 1
            continue

        orig_info = orig_events.get((vid, lbl), {})
        s1 = orig_info.get("start_sec_annotator_01", 0.0)
        e1 = orig_info.get("end_sec_annotator_01", 0.0)

        # Parse Annotator 2 values if submitted
        s2_str = row.get("annotator_02_start_sec", "").strip()
        e2_str = row.get("annotator_02_end_sec", "").strip()

        is_submitted = (
            ann2_id != "" and
            decision != "PENDING_REVIEW" and
            decision != "" and
            s2_str != "" and
            e2_str != ""
        )

        if not is_submitted:
            pending_count += 1
            reconciliation_records.append({
                "event_id": eid,
                "video_id": vid,
                "label": lbl,
                "reconciliation_status": "PENDING_ANNOTATOR_02",
                "delta_start_sec": None,
                "annotator_01": ann1_id,
                "annotator_02": ann2_id or "UNASSIGNED",
                "onset_01": s1,
                "offset_01": e1,
                "onset_02": None,
                "offset_02": None,
                "notes": "Awaiting Annotator 02 independent review"
            })
            continue

        # Check validity of submitted review
        case_errors = []
        if ann2_id.lower() in DISALLOWED_REVIEWERS:
            case_errors.append(f"Invalid reviewer name '{ann2_id}'")

        if ann1_id.lower() == ann2_id.lower():
            case_errors.append("SELF_VERIFICATION_FORBIDDEN: Reviewer cannot verify their own annotation")

        if not parse_iso_timestamp(review_ts):
            case_errors.append(f"Invalid review timestamp '{review_ts}'. ISO 8601 required")

        if not attest_full or not attest_indep:
            case_errors.append("Reviewer must attest both full_view and independent evaluation")

        try:
            s2 = float(s2_str.replace(",", "."))
            e2 = float(e2_str.replace(",", "."))
            if s2 < 0 or e2 <= s2:
                case_errors.append(f"Invalid timestamp interval: [{s2}, {e2}]")
        except ValueError:
            s2 = 0.0
            e2 = 0.0
            case_errors.append(f"Cannot parse start/end timestamps: '{s2_str}', '{e2_str}'")

        if case_errors:
            errors.extend([f"{eid}: {ce}" for ce in case_errors])
            invalid_count += 1
            reconciliation_records.append({
                "event_id": eid,
                "video_id": vid,
                "label": lbl,
                "reconciliation_status": "INVALID_SUBMISSION",
                "delta_start_sec": None,
                "annotator_01": ann1_id,
                "annotator_02": ann2_id,
                "onset_01": s1,
                "offset_01": e1,
                "onset_02": s2,
                "offset_02": e2,
                "notes": "; ".join(case_errors)
            })
            continue

        # Reconcile timestamps against SOP tolerance
        delta_start = abs(s1 - s2)
        if decision == "CONFIRM_EVENT" and delta_start <= SOP_ONSET_TOLERANCE_SEC:
            rec_status = "AGREED_CONSENSUS"
            agreed_count += 1
            rec_note = f"Agreed within tolerance ({delta_start:.2f}s <= {SOP_ONSET_TOLERANCE_SEC}s)"
        else:
            rec_status = "DISPUTED_NEEDS_ADJUDICATION"
            disputed_count += 1
            if decision != "CONFIRM_EVENT":
                rec_note = f"Decision mismatch: Annotator 2 selected {decision}"
            else:
                rec_note = f"Tolerance exceeded: Delta start {delta_start:.2f}s > {SOP_ONSET_TOLERANCE_SEC}s"

        reconciliation_records.append({
            "event_id": eid,
            "video_id": vid,
            "label": lbl,
            "reconciliation_status": rec_status,
            "delta_start_sec": round(delta_start, 2),
            "annotator_01": ann1_id,
            "annotator_02": ann2_id,
            "onset_01": s1,
            "offset_01": e1,
            "onset_02": s2,
            "offset_02": e2,
            "notes": rec_note
        })

    # Completeness check
    missing_events = [eid for eid in TARGET_POSITIVE_EVENTS if eid not in seen_event_ids]
    if missing_events:
        errors.append(f"MISSING_TARGET_EVENTS: Worksheet is missing {missing_events}")

    # Validation Gate Status
    val_gate_status = "PASSED" if (agreed_count == 5 and len(errors) == 0) else "BLOCKED"

    return {
        "total_target_events": len(TARGET_POSITIVE_EVENTS),
        "total_rows_worksheet": len(rows),
        "blind_protocol_intact": blind_protocol_intact,
        "agreed_consensus_count": agreed_count,
        "disputed_count": disputed_count,
        "pending_count": pending_count,
        "invalid_count": invalid_count,
        "validation_gate_status": val_gate_status,
        "errors": errors,
        "records": reconciliation_records,
    }


def validate_negative_review_manifest(
    manifest_csv: Path,
    split_manifest_csv: Path
) -> dict[str, Any]:
    """Audit Validation negative video review manifest and compute eligible FAR exposure.

    Args:
        manifest_csv: Path to validation_negative_review_manifest.csv.
        split_manifest_csv: Path to datasets/vn_dataset/split_manifest.csv.

    Returns:
        Audit report dictionary for negative videos.
    """
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Negative review manifest not found: {manifest_csv}")

    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    seen_vids = set()
    verified_negative_count = 0
    pending_count = 0
    hazard_count = 0
    technical_fail_count = 0
    errors: list[str] = []
    records: list[dict[str, Any]] = []

    total_duration_sec = 0.0
    verified_negative_exposure_sec = 0.0

    for row in rows:
        vid = row.get("video_id", "").strip()
        fname = row.get("filename", "").strip()
        sha_match = row.get("sha256_match", "").strip().upper() == "TRUE"
        tech_verdict = row.get("technical_verdict", "").strip()
        decision = row.get("content_decision", "").strip().upper()
        reviewer = row.get("reviewer_name", "").strip()
        review_ts = row.get("review_timestamp", "").strip()
        attest_full = row.get("attest_full_video_viewed", "").strip().upper() == "TRUE"
        dur = float(row.get("duration_seconds", 0.0) or 0.0)

        total_duration_sec += dur

        # Duplicate check
        if vid in seen_vids:
            errors.append(f"DUPLICATE_VIDEO_ID: {vid} appears multiple times")
            continue
        seen_vids.add(vid)

        # Split provenance check: Must be in EXPECTED_VAL_NEG_VIDEOS
        if vid not in EXPECTED_VAL_NEG_VIDEOS:
            errors.append(f"SPLIT_LEAKAGE_DETECTED: Video {vid} does not belong to Validation negative split")
            continue

        # Technical integrity check
        tech_ok = sha_match and tech_verdict.startswith("PASS")
        if not tech_ok:
            technical_fail_count += 1
            if not sha_match:
                errors.append(f"SHA256_MISMATCH: Video {vid} hash mismatch against manifest")

        if decision == "PENDING_HUMAN_REVIEW" or not decision:
            pending_count += 1
            far_admissible = False
            notes = "Pending human content review"
        elif decision in ("FIRE_PRESENT", "SMOKE_PRESENT", "UNCERTAIN"):
            hazard_count += 1
            far_admissible = False
            notes = f"Content flagged as {decision}; inadmissible as negative exposure"
        elif decision == "VERIFIED_NEGATIVE":
            # Strict human verification criteria
            case_errors = []
            if reviewer.lower() in DISALLOWED_REVIEWERS:
                case_errors.append(f"Invalid reviewer name '{reviewer}'")
            if not parse_iso_timestamp(review_ts):
                case_errors.append(f"Invalid review timestamp '{review_ts}'")
            if not attest_full:
                case_errors.append("Reviewer must attest viewing full video")
            if not tech_ok:
                case_errors.append(f"Cannot admit video with technical failure ({tech_verdict})")

            if case_errors:
                errors.extend([f"{vid}: {ce}" for ce in case_errors])
                far_admissible = False
                notes = "; ".join(case_errors)
            else:
                verified_negative_count += 1
                far_admissible = True
                verified_negative_exposure_sec += dur
                notes = "Human certified negative and technically intact"
        else:
            errors.append(f"{vid}: Unknown decision '{decision}'")
            far_admissible = False
            notes = f"Unknown decision: {decision}"

        records.append({
            "video_id": vid,
            "filename": fname,
            "duration_seconds": dur,
            "technical_verdict": tech_verdict,
            "content_decision": decision,
            "far_admissible": far_admissible,
            "notes": notes
        })

    missing_vids = EXPECTED_VAL_NEG_VIDEOS - seen_vids
    if missing_vids:
        errors.append(f"MISSING_VALIDATION_VIDEOS: Manifest is missing {sorted(list(missing_vids))}")

    far_gate_status = "PASSED" if (verified_negative_count >= 10 and len(errors) == 0) else "BLOCKED"

    return {
        "expected_videos_count": len(EXPECTED_VAL_NEG_VIDEOS),
        "audited_videos_count": len(rows),
        "verified_negative_count": verified_negative_count,
        "pending_count": pending_count,
        "hazard_flagged_count": hazard_count,
        "technical_fail_count": technical_fail_count,
        "total_duration_hours": round(total_duration_sec / 3600.0, 4),
        "verified_exposure_hours": round(verified_negative_exposure_sec / 3600.0, 4),
        "far_gate_status": far_gate_status,
        "errors": errors,
        "records": records
    }


def validate_all_validation_reviews(
    gt_worksheet_csv: Path,
    events_jsonl_path: Path,
    neg_manifest_csv: Path,
    split_manifest_csv: Path,
    output_dir: Path
) -> dict[str, Any]:
    """Execute complete validation suite on both GT worksheet and negative review manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Starting scientific validation of Week 6 reviews...")

    gt_report = validate_positive_gt_worksheet(gt_worksheet_csv, events_jsonl_path)
    neg_report = validate_negative_review_manifest(neg_manifest_csv, split_manifest_csv)

    blockers: list[str] = []
    if gt_report["validation_gate_status"] != "PASSED":
        blockers.append(
            f"Validation Positive GT Gate is BLOCKED: Only {gt_report['agreed_consensus_count']}/5 events "
            f"verified. ({gt_report['pending_count']} pending, {gt_report['disputed_count']} disputed)."
        )

    if neg_report["verified_negative_count"] == 0:
        blockers.append(
            f"Validation Negative Verification is PENDING: 0/{neg_report['expected_videos_count']} videos "
            f"have been human-verified as negative. Eligible FAR exposure = 0.0h."
        )

    if neg_report["technical_fail_count"] > 0:
        blockers.append(
            f"Negative Video Technical Faults: {neg_report['technical_fail_count']} video(s) failed technical integrity."
        )

    overall_verdict = "BLOCKED" if blockers else "PASSED"

    summary = {
        "audit_timestamp": datetime.datetime.now().isoformat(),
        "positive_ground_truth": gt_report,
        "negative_videos": neg_report,
        "overall_verdict": overall_verdict,
        "blockers": blockers
    }

    # Save summary JSON
    summary_path = output_dir / "validation_review_summary.json"
    atomic_write_json(summary_path, summary)
    LOG.info("Saved review summary JSON to %s", summary_path)

    # Save Reconciliation Report MD
    rec_md = output_dir / "validation_gt_reconciliation_report.md"
    _write_reconciliation_report(rec_md, summary)
    LOG.info("Saved reconciliation report to %s", rec_md)

    return summary


def _write_reconciliation_report(path: Path, data: dict[str, Any]) -> None:
    """Write human-readable reconciliation Markdown report."""
    gt = data["positive_ground_truth"]
    neg = data["negative_videos"]
    verdict = data["overall_verdict"]
    blockers = data["blockers"]

    lines = [
        "# BÁO CÁO ĐỐI SOÁT KIỂM ĐỊNH GROUND TRUTH VÀ VIDEO ÂM TÍNH (TUẦN 6)",
        "## FireWatch – Dual-Annotator Reconciliation & Exposure Certification",
        "",
        "> **Tiêu chuẩn khoa học:** Đề cương NCKH & Nghiệm thu Tuần 6 (Độc lập đánh giá, SOP sai số <= 0.5s)  ",
        f"> **Thời điểm đối soát:** `{data['audit_timestamp']}`  ",
        f"> **Phán quyết tổng thể:** **`{verdict}`**  ",
        "",
        "---",
        "",
        "## 1. KẾT QUẢ ĐỐI SOÁT 5 SỰ KIỆN GROUND TRUTH TẬP VALIDATION",
        "",
        f"- **Mục tiêu:** {gt['total_target_events']} sự kiện (`POS_07`, `POS_08` (x2), `POS_17`, `POS_18`)",
        f"- **Giao thức mù đôi (Blind Protocol):** `{'ĐẠT (Không lộ mốc giờ)' if gt['blind_protocol_intact'] else 'VI PHẠM'}`",
        f"- **Số sự kiện đạt đồng thuận (`AGREED_CONSENSUS`):** **{gt['agreed_consensus_count']} / 5**",
        f"- **Số sự kiện chờ Annotator 2 (`PENDING`):** **{gt['pending_count']} / 5**",
        f"- **Số sự kiện bất đồng (`DISPUTED`):** {gt['disputed_count']} / 5",
        f"- **Trạng thái cổng Validation Gate:** **`{gt['validation_gate_status']}`**",
        "",
        "| Event ID | Video ID | Nhãn | Onset Annotator 1 | Onset Annotator 2 | Sai số $\\Delta start$ | Trạng thái đối soát | Ghi chú |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for r in gt["records"]:
        s1_str = f"{r['onset_01']}s" if r['onset_01'] is not None else "N/A"
        s2_str = f"{r['onset_02']}s" if r['onset_02'] is not None else "CHƯA NỘP"
        delta_str = f"{r['delta_start_sec']}s" if r['delta_start_sec'] is not None else "N/A"
        lines.append(
            f"| `{r['event_id']}` | **{r['video_id']}** | `{r['label']}` | "
            f"{s1_str} | {s2_str} | {delta_str} | `{r['reconciliation_status']}` | {r['notes']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. KẾT QUẢ THẨM ĐỊNH 13 VIDEO ÂM TÍNH TẬP VALIDATION",
        "",
        f"- **Tổng số video:** {neg['expected_videos_count']} video ({neg['total_duration_hours']:.2f} giờ)",
        f"- **Video đạt tiêu chuẩn kỹ thuật:** {neg['expected_videos_count'] - neg['technical_fail_count']} / {neg['expected_videos_count']}",
        f"- **Video đã được thẩm định âm tính thật (`VERIFIED_NEGATIVE`):** **{neg['verified_negative_count']} video**",
        f"- **Video chờ thẩm định (`PENDING_HUMAN_REVIEW`):** **{neg['pending_count']} video**",
        f"- **Thời lượng âm tính đủ điều kiện tính FAR:** **{neg['verified_exposure_hours']:.2f} giờ**",
        f"- **Trạng thái cổng FAR Validation:** **`{neg['far_gate_status']}`**",
        "",
        "| Video ID | Tệp tin | Thời lượng | Kỹ thuật | Thẩm định nội dung | Đủ điều kiện FAR? | Ghi chú |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
    ])

    for r in neg["records"]:
        far_icon = "ĐỦ ĐIỀU KIỆN" if r['far_admissible'] else "CHƯA ĐỦ ĐIỀU KIỆN"
        lines.append(
            f"| **{r['video_id']}** | `{r['filename']}` | {r['duration_seconds']}s | "
            f"`{r['technical_verdict']}` | `{r['content_decision']}` | **{far_icon}** | {r['notes']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. CÁC RÀO CẢN BẮT BUỘC ĐANG CHẶN NGHIỆM THU",
        "",
    ])

    for idx, b in enumerate(blockers, start=1):
        lines.append(f"{idx}. **{b}**")

    if not blockers:
        lines.append("Toàn bộ các rào cản khoa học đã được giải tỏa thành công!")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Week 6 Validation Reviews")
    parser.add_argument("--gt-worksheet", type=Path, default=ROOT / "runs/audits/week6/validation_gt_review_worksheet.csv")
    parser.add_argument("--events-jsonl", type=Path, default=ROOT / "data/ground_truth/events.jsonl")
    parser.add_argument("--neg-manifest", type=Path, default=ROOT / "runs/audits/week6/validation_negative_review_manifest.csv")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "datasets/vn_dataset/split_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/audits/week6")
    args = parser.parse_args()

    try:
        summary = validate_all_validation_reviews(
            args.gt_worksheet, args.events_jsonl, args.neg_manifest, args.split_manifest, args.output_dir
        )
        LOG.info("Validation complete. Verdict: %s", summary["overall_verdict"])
        return 0
    except Exception as err:
        LOG.error("Validation failed: %s", err, exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
