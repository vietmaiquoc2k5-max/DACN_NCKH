# Week 4 Final Reliability Report

Generated: 2026-11-12T10:26:40.338264+00:00
Base commit: `9fbdaa6b9179a7ba297ca9816e720741be18eba2`; the audit ran with a modified worktree and remains uncommitted.
**Acceptance: BLOCKED** — current workspace is missing most manifest video sources; no full-set strict audit or independently verified negative FAR is possible.

## Environment and preflight

Python: `3.9.13 (tags/v3.9.13:6de2ca5, May 17 2022, 16:36:42) [MSC v.1929 64 bit (AMD64)]`; executable `F:\Dự án cá nhân\phan mem canh bao chay\.venv\Scripts\python.exe`.
OpenCV: `5.0.0`; FFmpeg: `ffmpeg version 9.0.1-essentials_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers` at `C:\Users\vietm\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.EXE`; FFprobe: `ffprobe version 9.0.1-essentials_build-www.gyan.dev Copyright (c) 2007-2026 the FFmpeg developers`.
Checkpoint SHA-256: `f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6` (run match: True).
Manifest: 16/79 source files present; 63 missing.
The old preflight artifact reports Python 3.10.11 and no FFmpeg path, while this run observes Python 3.14 globally, Python 3.9 in `.venv`, and FFmpeg 9.0.1. Its recorded timestamp (21:45 UTC) is later than the current preflight timestamp (2026-11-12T10:26:14.193920+00:00), so its chronology is unverified. The old audit log records clean FFmpeg frame counts but not executable path/version. This is an environment/metadata-lineage difference; the old null path is not current-host evidence.

## Strict video audit

Audit policy `strict_full_stream_zero_tolerance_v1`; attempted 16/79; strict valid 16; statuses `{"BLOCKED_INPUT_MISSING": 63, "STRICT_VALID": 16}`.
Strict verified exposure: 0.162292 h. Missing sources contribute zero exposure and retain `BLOCKED_INPUT_MISSING`.
Per-video FFmpeg exit/stderr/count/progress, OpenCV count, container metadata, codec, frame rates, packet PTS/DTS presence/ranges and DTS order, SHA-256 and status are in `video_audit_verified.json` and the CSVs. PTS/DTS is recorded; B-frame PTS reordering is not treated as an error.
Tolerance policy: no experiment-backed allowance was found for 5 frames, 10 frames, 0.25 seconds, or 0.35 seconds. Production audit uses zero count tolerance; 5-frame/0.25-second legacy values remain disabled, and the 10-frame text is only a synthetic truncation-test description. No codec/FPS/PTS-based tolerance is inferred. The prior 19 NORMAL_EOF / 58 NEEDS_REVIEW / 2 TRUNCATED_STREAM records are historical only and were not treated as current-valid evidence.

## Inference reconciliation

Historical run `run_20261111_233035_neg_full`: 79 rows, 929939 decoded frames, 128291 sampled frames; sampling ratio 0.13795636057848956; timeline coverage 100% is **UNVERIFIED** because per-frame timestamps are not available.
Event totals from event files: {'pipeline_a': 988, 'pipeline_b1': 547, 'pipeline_b2': 421}; summary reconciliation: B1=True, B2=True. Pipeline A raw frame alarms (9133) are kept distinct from its event list count (988).

## FAR / alert-event rates

- A_ALL_HISTORICALLY_EVALUATED: BLOCKED; no verified negative scope.
- B_STRICT_CURRENT_AUDIT_AND_INFERENCE_MATCH: 16 videos, 0.16229167 h; B2=5 events, 30.808729139922978/h, exact 95% CI [10.0035, 71.8973]; reduction=81.4814814814815%.
- C_INDEPENDENTLY_VERIFIED_NEGATIVE: BLOCKED; no verified negative scope.
- Historical requested Scope 1 arithmetic from matching per-video rows: 79 rows, 7.13115 h, B2=421, 59.0368/h, exact Poisson 95% CI [53.5314, 64.9547]; 60 rows lack NORMAL_EOF. Status: `REJECTED_SCOPE_CONTAINS_NEEDS_REVIEW_ROWS`.
- Historical requested Scope 2 arithmetic from matching per-video rows: 77 rows, 6.98463056 h, B2=417, 59.7025/h, exact Poisson 95% CI [54.1091, 65.7172]; 58 rows lack NORMAL_EOF. Status: `REJECTED_SCOPE_CONTAINS_NEEDS_REVIEW_ROWS`.

These are alert-event rates on the recorded nominal-negative batch, not scientifically confirmed false-alarm rates: negative semantics remain pending independent review. Scope B exclusions and removed events/exposure are enumerated explicitly in `far_recomputed.json`.
The requested historical rows pair their event counts and durations internally, but their FAR claims fail strict exposure validation: Scope 2 excludes only the 2 truncated rows and retains 58 NEEDS_REVIEW / unverified rows. The 16 current strict-valid files yield the only current recomputed scope; the historical Scope 1/2 arithmetic above is not accepted FAR.
Stored reduction artifacts disagree (57.05% versus 56.48%). The rerun computes a common formula from event lists; no historical output was edited to force agreement.

## Provenance, Ground Truth and Operating Point

Run ID `run_20261111_233035_neg_full`; inference source base `256065e7e54c35c335e6953d0bc373829ac94997`; packaged commit `62785d7b4cdd1b499c99abbd016ecedf6fa1ee35`.
Current audit base `9fbdaa6b9179a7ba297ca9816e720741be18eba2`; audit commit is unset (`UNCOMMITTED_WORKTREE`). Current manifest raw and normalized hashes are recorded separately. Historical run manifest hash match: False.
File modification times (UTC), sizes and SHA-256 values for generated outputs and key source artifacts are recorded in `artifact_timestamp_audit.json`; artifact-level lineage timestamps are in `artifact_lineage.csv`.
Ground Truth queue records 11 verified rows and 24 pending; no labels changed.
Operating Point: **PROVISIONAL**; official promotion is false. Recall and latency remain unverified while GT is pending.
RTSP: **NOT_TESTED**; no live RTSP source was available in this audit environment.

## Tests and CI

Local suite: `.venv\Scripts\python.exe -m unittest discover -s tests -v` — PASS (185 tests, 0 failures, 0 errors; 21.58 s).
Requested real-pipeline scenarios and their unittest method names/results are in `runs/benchmarks/week4_consistency_fix/scenario_test_results.csv`; PASS is derived only from the recorded full-suite run with zero failures/errors.
Remote combined status: UNVERIFIED_NO_STATUS_CHECKS for base commit `9fbdaa6b9179a7ba297ca9816e720741be18eba2`; 0 status checks. No CI PASS is claimed.

## Unresolved blockers

- Restore 63 missing original source videos and rerun the strict audit; do not substitute generated fixtures.
- Retrieve the exact manifest snapshot referenced by the historical inference run, or mark that run as unverified and rerun inference after input hashes are verified.
- Complete independent review for negative semantics before reporting FAR; preserve 24 pending positive Ground Truth events until Annotator 02 confirms.
- Reconcile the historical Scope B reduction discrepancy from common raw event inputs; preserve PROVISIONAL status.
- Obtain an actual GitHub Actions workflow result before marking CI as PASS.

## Acceptance

**BLOCKED.** Evidence paths are the JSON/CSV/log files in `runs/benchmarks/week4_consistency_fix/`. No `TECHNICAL_VALIDATION_PASS` or `WEEK4_FULL_ACCEPTANCE` is claimed.
