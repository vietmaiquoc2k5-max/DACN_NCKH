# WEEK 4 FINAL RELIABILITY REPORT

**Audit time:** 2026-11-12 17:45 +07:00
**Result:** **NOT ACCEPTED — STRICT MEDIA RE-AUDIT REQUIRED**
**Audit code base:** `3ebd4f45169380c33409faf2e9899ba167d648d7`; local changes are uncommitted, so there is no audit commit SHA.

This report documents code hardening and tests against locally generated media. It does not claim the 79-video source set was re-audited, does not retrain or alter model weights, does not verify Ground Truth, and does not promote the Operating Point.

## 1. Defects found

1. The audit incremented its frame count with `VideoCapture.grab()` and treated a failed grab/read as the end of decoding. That result alone cannot distinguish normal EOF from a decoder that stopped silently.
2. FFprobe was allowed to fall back to `-count_frames`; counting frames is not evidence that a decoder completed cleanly.
3. Fixed 5/10-frame and 0.25/0.35-second tolerances could mark discrepancies `NORMAL_EOF` / `VALID` without a repository-backed calibration across codecs, containers, FPS, or PTS/DTS behavior.
4. Cache schema v2 could preserve earlier `VALID` decisions made under the old EOF rules.
5. Existing provenance fields mix code commits and artifact/run commits. The provenance audit says `bc636ea` is current although repository HEAD is `3ebd4f4`; the inference run was produced from a different recorded code snapshot.
6. Scope 1's 79-video exposure includes the partial decoded exposure of two streams already flagged as truncated. It is not suitable as the clean-video FAR denominator under the requested exclusion rule.

## 2. Root cause

The previous implementation inferred termination from frame-count differences and asserted empirical container behavior without local validation data. It also conflated a metadata/frame-count probe with a full decode. The cache version did not change when the validity rule changed, and artifact provenance used one `source_commit` field for outputs created from different snapshots.

## 3. Code changes

In [src/video_audit.py](../src/video_audit.py):

- Full decode now uses FFmpeg through a null output, `-xerror`, and `-progress pipe:1`. The audit records exit code, stderr, final `progress=end`, and decoded frame count.
- `NORMAL_EOF` / `VALID` now requires a clean FFmpeg completion and exact agreement among FFmpeg count, sequential OpenCV `read()` count, positive container frame count, and valid FPS.
- OpenCV stopping alone, FFmpeg absence/failure, invalid or zero FPS, zero container frame count, and any count mismatch cannot produce `NORMAL_EOF`.
- FFmpeg decode errors after frames were read produce `TRUNCATED_STREAM`; errors before any frame produces `DECODE_ERROR`. Uncorroborated stop or mismatch produces `NEEDS_REVIEW`. `CANNOT_OPEN` and `PROBE_ONLY` remain distinct.
- FFprobe metadata is not treated as a full-decode check. FFprobe alone cannot prove EOF.
- The former 5/10-frame and 0.25/0.35-second tolerances are disabled. Any count discrepancy requires review. No PTS/DTS, codec-specific, or FPS-specific tolerance is claimed without empirical evidence.
- Cache schema is now v3, invalidating prior v2 `VALID` records for re-audit.

Tests were updated/added in [tests/test_video_audit_and_consistency.py](../tests/test_video_audit_and_consistency.py).

## 4. Test results

Command:

```powershell
python -m unittest discover -s tests -p test_video_audit_and_consistency.py
```

**Result: PASS — 58 tests, 0 failures.** The media integration cases exercise the actual `audit_single_video` path. Synthetic videos are created locally for repeatability; they are not presented as source-dataset evidence.

| Scenario | Result | Evidence |
|---|---|---|
| Clean video through EOF | PASS | FFmpeg exit 0, stderr empty, `progress=end`, frame count 24; OpenCV and container also report 24 |
| Video truncated midstream | PASS | Pipeline rejects `NORMAL_EOF`; result requires review or decoder-error classification |
| Corrupt packet/frame payload | PASS | Full audit rejects `NORMAL_EOF`; FFmpeg/OpenCV decoder diagnostics are emitted |
| OpenCV silently stops early | PASS | OpenCV count disagrees with completed FFmpeg count; `NEEDS_REVIEW` |
| Wrong container frame count | PASS | FFmpeg/OpenCV count 20 versus container 19; `NEEDS_REVIEW` |
| FPS = 0 / invalid | PASS | `NEEDS_REVIEW`, not verified |
| Reported frame count = 0 | PASS | `NEEDS_REVIEW`, not verified |
| FFmpeg unavailable | PASS | `NEEDS_REVIEW`; OpenCV termination is not accepted as EOF |
| FFprobe unavailable, FFmpeg available | PASS | Clean FFmpeg decode can verify EOF; FFprobe is optional metadata |
| FFprobe returns error | PASS | FFmpeg full decode remains the EOF evidence; probe error does not establish or negate decoder completion |
| FFmpeg decode error / nonzero exit | PASS | Never `NORMAL_EOF`; partial output is classified as truncated, no output as decode error |
| Same-size content changed | PASS | Full SHA-256 mismatch invalidates cache and triggers a fresh audit |
| Old cache schema | PASS | v1/v2 entries are invalidated by v3 schema guard |

These passes establish behavior on the tested local fixtures and controlled failure injections only. They do not establish that any absent source video is valid.

## 5. EOF evidence and tolerance policy

For the clean integration fixture, the audit recorded:

- FFmpeg: `exit_code=0`, empty stderr, `progress=end`, `nb_read_frames=24`.
- OpenCV sequential decode: 24 frames.
- Container metadata: 24 frames; FPS 12.
- Result: `NORMAL_EOF`, `full_decode_verified=true`.

For all discrepancy/failure fixtures, the pipeline did not emit `NORMAL_EOF`.

The old thresholds 5 frames, 10 frames, 0.25 seconds, and 0.35 seconds had no measured validation set or codec/container/PTS/DTS stratification in this repository. They are therefore not used as acceptance limits. A future tolerance would need paired decoder counts and timestamp evidence across the supported formats, with a documented error analysis. Until then, mismatch means `NEEDS_REVIEW`.

## 6. FAR and exact Poisson 95% CI

Values below were recalculated with repository implementation `src.event_eval.exact_poisson_ci` (two-sided exact/Garwood, alpha 0.05).

| Scope | Videos | B2 alerts | Exposure | FAR / hour | Exact 95% CI / hour | Assessment |
|---|---:|---:|---:|---:|---:|---|
| 1, stored all-video reconciliation | 79 | 421 | 7.1312 h | 59.0363 | [53.5310, 64.9542] | Diagnostic only: includes partial exposure from two truncated streams |
| 2, untruncated videos | 77 | 417 | 6.9846 h | 59.7028 | [54.1093, 65.7175] | Numerator and denominator are scoped to the same 77 videos; use for clean-video FAR |

Stored reconciliation identifies `VN_NEG_10` and `VN_NEG_48` as truncated. Removing their partial decoded exposure changes 7.1312 h to 6.9846 h; removing the four B2 alerts attributed to `VN_NEG_48` changes 421 to 417. The stored manifest independently reports 79 total, 77 valid, 2 needing review, and the same two truncated IDs.

No failed-video exposure was added to Scope 2. Alerts were not silently dropped: Scope 1 retains all 421; Scope 2 explicitly excludes the 4 alerts attributed to the excluded failed stream. Do not compare a numerator from one scope with the denominator from the other.

**Limit:** these FAR values and confidence intervals are recalculations of stored run/reconciliation data. They are not a fresh model inference or a fresh audit of the 79 source files.

## 7. Provenance and artifact timestamps

| Artifact | Recorded identity / timestamp | Repository evidence | Status |
|---|---|---|---|
| Audit code | Base `3ebd4f45169380c33409faf2e9899ba167d648d7`; edited working tree | `git rev-parse HEAD`; changes not committed | No audit commit exists yet |
| Checkpoint `runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt` | SHA-256 `f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6`; mtime 2026-10-28 00:55 +07 | Hash recomputed from local checkpoint | Unchanged |
| Current negative manifest `data/negative_videos_manifest.json` | SHA-256 `5413a0328fc6aea70da0f5b22f96514b8bab780a45737f22be0dea896768bfb1`; mtime 2026-11-12 17:26 +07 | Last committed at `091352f7` | Does not match run's recorded manifest hash |
| Inference run `run_20261111_233035_neg_full` | Run manifest created `2026-11-12T00:00:34.195501`; checkpoint hash matches above; recorded manifest hash `75626b6b563bae499848543732cfb1a172e1475773674b1acbd36a80e8e0c1ed`; mtime 17:26 +07 | Run records full code commit `62785d7b4cdd1b499c99abbd016ecedf6fa1ee35` and source base `256065e7e54c35c335e6953d0bc373829ac94997`; manifest file last committed `6e490bd4` | Run and current manifest are not the same snapshot |
| Inference reconciliation | Timestamp `2026-11-12T09:13:00Z`; mtime 2026-11-12 17:26 +07 | Last committed at `091352f7`; run ID matches | Historical result; not regenerated by this audit |
| `runs/benchmarks/week4_final/data_audit.json` | mtime 2026-11-12 17:26 +07 | Last committed at `94a1d3b3` | Historical audit; not regenerated under strict EOF rules |
| `runs/benchmarks/week4_final/metrics.json` | Embedded timestamp `2026-11-12T17:35:00+07:00`; mtime 17:26 +07 | Last committed at HEAD `3ebd4f4`; embedded `source_commit` says `bc636ea1` | Historical source snapshot is not independently confirmed; timestamp disagrees with mtime |
| `runs/benchmarks/week4_final/operating_point_audit.json` | Embedded timestamp `2026-11-12T17:35:00+07:00`; mtime 17:26 +07 | Last committed at HEAD `3ebd4f4`; embedded `source_commit` says `bc636ea1`; status remains PROVISIONAL | Historical source snapshot is not independently confirmed; timestamp disagrees with mtime |
| `runs/benchmarks/week4_full_inference/run_20261111_233035_neg_full/video_inference_summary.csv` | mtime 2026-11-12 17:26 +07 | Last committed at `62785d7b` | Inference output is from the historical run snapshot |
| `runs/benchmarks/week4_final/provenance_audit.json` | Corrected audit timestamp `2026-11-12T17:45:00+07:00`; regenerated in this working tree | `source_commit` and `audit_commit` are null; `audit_base_commit=3ebd4f4`; dirty tree and historical artifact commits are recorded separately | Corrected; no uncommitted code is claimed to have a commit |
| Ground Truth master CSV | mtime 2026-11-12 17:26 +07; SHA-256 `d58cb156e10770df3a194aceb5a4629d6bb3b4b6b54857cb6861f41a4c3ef509` | Last committed at `94a1d3b3` | Existing annotations preserved; not reverified |

The current negative manifest hashes to `5413…`, while the inference run records `75626…`. Therefore those artifacts are not proven to derive from the same manifest snapshot. The metrics and operating-point JSON files each embed 17:35 +07 timestamps while their filesystem mtimes are around 17:26 +07; those timestamps do not agree. The earlier tracked provenance audit used `bc636ea1` as both source and audit commit although repository HEAD is `3ebd4f4`; `bc636ea1` may describe a historical source snapshot, but it is not verified as the source for every listed artifact or as the current audit commit. The corrected provenance artifact now records the audit base and leaves `source_commit` / `audit_commit` null because this working tree is uncommitted. No old result is relabeled as newly generated.

The checkpoint SHA-256 matches the inference run record. No checkpoint was modified.

## 8. Unresolved issues and scientific limits

- 63 of the 79 negative-video paths in the current manifest are absent from this workspace. Strict EOF re-audit of that source set cannot be completed here. The stored `full_stream_decode_verified` flags predate this stricter policy and must not be treated as new-policy proof.
- The run's manifest digest and current manifest digest differ. The old inference/FAR artifacts are therefore not synchronized with today's manifest snapshot.
- Historical metrics and operating-point files still carry `source_commit=bc636ea`; the artifact does not establish whether that is the exact source snapshot used for every result. Their generation time also conflicts with filesystem mtime. The corrected audit marks the relationship unsynchronized instead of substituting a guessed commit.
- Ground Truth stays **11/35 cross-verified; 24 pending** until Annotator 02 confirms. This audit made no Ground Truth determination.
- Operating Point stays **PROVISIONAL**.
- RTSP stays **NOT_TESTED** without a real RTSP source.
- The FAR interval is exact Poisson for the stored event counts and exposure, but re-running inference or recomputing alarms from missing source videos was not possible.
- No codec-specific tolerance is approved. PTS/DTS evidence and representative empirical validation remain outstanding.

## 9. Week 4 acceptance conclusion

**Week 4 reliability acceptance: NOT ACCEPTED / NEEDS REVIEW.**

The code hardening and all 58 targeted tests pass. The clean synthetic fixture provides direct FFmpeg EOF evidence, and the failure fixtures do not become `VALID` without corroboration. However, the 79-video negative source set was not re-audited under the stricter policy, most source files are absent, and the existing inference run's manifest digest does not match the current manifest. Those gaps prevent a PASS for the real-data Week 4 audit.

**Required before acceptance:** restore the exact inference manifest snapshot and all source videos, rerun the strict audit over the full negative set, reconcile alert counts and exposure per video using the new statuses, then regenerate provenance with the resulting audit/run identity. No commit or push was made.

