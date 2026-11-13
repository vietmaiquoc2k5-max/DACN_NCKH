# Week 4 — recovery / provenance / human Ground Truth toolkit

This package is a **standalone local assistant tool**, not a patch merged into `DACN_NCKH`. It is meant to be run on the Windows machine holding the actual video files. Python **3.10+**, standard library only. It does not run inference, fabricate annotations, modify original videos, commit or push.

Reviewed GitHub `fix-4` HEAD when package was authored: `0c13477dd77dde56effeb9b706cdbcedf2549008` (2026-11-13).

## Commands in PowerShell

Set the paths after extracting this package (do not assume the example repository path is yours):

```powershell
$Repo = 'F:\Dự án cá nhân\phan mem canh bao chay'
$Tool = 'C:\path\to\week4_recovery_toolkit\week4_recovery.py'

# 2a. Report EXACT locations of source media within project and explicit extra search roots.
python $Tool --repo $Repo inventory --search-dir 'D:\DACN_NCKH' --search-dir 'F:\Data' 

# 2b. Only IF a source matches both manifest SIZE and SHA-256: explicitly restore it
# to its declared path WITHOUT overwriting originals. First run inventory without --recover!
python $Tool --repo $Repo inventory --search-dir 'D:\DACN_NCKH' --recover copy

# 3. Inspect historical inference run, audit counts and known timeline limitations;
# records a per-video state and never automatically reruns inference.
python $Tool --repo $Repo provenance

# 4. Prepare blank human review forms in a PRIVATE directory next to the repo.
python $Tool --repo $Repo prepare-reviews

# 5. Validate signed human review forms and current video/inference evidence.
# This will currently exit code 2 with BLOCKED status until evidence is completed.
python $Tool --repo $Repo evaluate
```

Each command outputs detailed reports under a newly timestamped folder at `runs/benchmarks/week4_assisted_recovery/<timestamp>/`. A `2` exit code is **an expected evidence blocker** (not a Python crash); `3` means missing/invalid input or a runtime error. `0` indicates the given operation completed, **not that scientific acceptance has passed**.

### Important details about the 16 files

The manifest declares the 16 affected `negsVideo*.avi` in `datasets/test_videos/neg/`. Earlier audit artifacts reported all 79 present in another workspace; the `fix-4` workspace reported 63/79 accessible. **Never assume the files are deleted or replace them with unrelated media.** Scan explicitly allowed local directories, compare SHA-256, then use an operator-approved `--recover copy` or `--recover hardlink`. `copy` requires sufficient free disk space. `hardlink` works only on the same filesystem. No destructive force/overwrite options exist.

### Human Ground Truth protocol

The package creates, but does **not** fill in:

- `negative_annotator_01.csv` and `negative_annotator_02.csv` — independently observed video status, video hash, human identifier, review time, evidence and signed decision.
- `negative_adjudication.csv` — explicit human consensus; must be filled after both independent reviews.
- `positive_annotator_02.csv` — BLIND review: only video name and blank `annotation_events_json`; no annotator 01 intervals or model boxes. Reviewer records events as JSON list of `label`, `start_sec`, `end_sec` plus actual source SHA-256 and evidence.
- `positive_adjudication.csv` — kept **separate from blinded review**, mapping original GT record index to independently recorded reviewer 02 event index, with signed human adjudication.

Do not commit personally identifying annotation records unless authorized. Existing human forms are preserved on reruns. The code verifies CSV fields and source bytes; **it cannot verify whether a person truly viewed the video**. Maintain external original evidence (CVAT export, review history, sign-off), ideally with timestamps and access controls.

### Interpretation of `evaluate`

The current code deliberately produces **readiness and blockers**, not invented FAR/Recall/Delay values. A negative video is counted as eligible ONLY if (1) two independent human signed reviews and explicit adjudication exist, (2) its original file matches SHA-256, (3) strict audit is valid, and (4) per-video inference timeline has independent evidence (`REUSE_VERIFIED`). Positive events require independent blind reviewer and adjudication. It reports `null` metrics until these conditions, validation lock and adequate scope are satisfied.

Once ready, run the repository's source-controlled event evaluator using an approved **locked validation Operating Point**, independently verified GT, exactly paired event scopes, and Poisson 95% CI. The user/agent must then inspect the official evaluator run and files. This bundle intentionally **does not automatically edit** repo GT snapshot, promote `PENDING` to `VERIFIED`, or run model inference. A checked record is not a scientific PASS until independently reviewed and accepted.

For the repository's own latest evaluation code, inspect `scripts/run_phases4_to_8_recovery.py`, `src/event_eval.py`, `src/week4_acceptance.py`, and the run manifest to execute data-dependent stages on the correct host. If frame PTS were not stored, diagnose/recover actual frame evidence or rerun **only those video IDs** whose evidence is irretrievable. Mere timestamp reconstruction from FPS is not independent PTS proof.

## Tests

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
```

The included tests use **synthetic bytes solely as local testing fixtures** and never report those as experiment footage, Ground Truth or actual project benchmark results.

### Conditional metric recomputation (only after review and technical evidence are genuinely complete)

`evaluate` only returns a readiness report. `recompute` first runs the same gates, then delegates **event matching, Event Recall, Event F1, Detection Delay and Poisson CI 95%** to `src.event_eval.evaluate` / `src.event_eval.exact_poisson_ci` in the actual repository. Thus it does not introduce an alternate matching implementation. It uses paired video scopes and the existing `normalized_event_episodes.json` negative episodes, `week4_final/frame_level_events.json` and `week4_final/temporal_confirmed_events.json` positive events. When inference is actually rerun, pass matching **new** event artifacts with the flags below and make sure source hashes are from the same run.

```powershell
# ONLY after real human review, >=5h eligible negative video, verified PTS,
# authenticated positive source/run provenance, and validation lock evidence:
python $Tool --repo $Repo recompute `
  --validation-lock 'D:\evidence\validation_op_lock.json' `
  --positive-provenance 'D:\evidence\positive_inference_provenance.json' `
  --reuse-report 'D:\evidence\inference_reuse_per_video.json' `
  --negative-episodes 'D:\evidence\normalized_event_episodes.json' `
  --positive-baseline 'D:\evidence\positive_baseline_events.json' `
  --positive-temporal 'D:\evidence\positive_temporal_events.json'
```

The lock JSON must provide `selected_on="validation"`, `locked_before_test=true`, `source_commit` (the actual commit), `validation_evidence_ref` (the real validation artifacts), and validated numeric `tolerance_sec` / `early_tolerance_sec`. **These fields must be supported by independent run/commit history, not invented.** The positive provenance JSON must include `videos` with per-video `video_id`, current source SHA-256, source-bound `run_id`, checkpoint SHA-256 and verified timeline evidence; top-level `baseline_events_sha256` and `temporal_events_sha256` must match the actual prediction JSON files. Any missing/mismatched evidence aborts. The negative per-video provenance supplied by `--reuse-report` must have a verified timeline and `REUSE_VERIFIED` for each eligible video.

Even after recalculation, output is `MEASURED_REQUIRES_INDEPENDENT_PROVENANCE_AUDIT`, **not automatic `FULL_PASS`**. Independent review of the annotation log and validation lock remains mandatory. `recompute` writes `week4_metrics_recomputed.json` with the exact video IDs, exposure, counts, Event Recall/F1, delay, FAR/hour, Garwood Poisson 95% CI and SHA-256s of source prediction artifacts. No benchmark value is guessed when files are missing.
