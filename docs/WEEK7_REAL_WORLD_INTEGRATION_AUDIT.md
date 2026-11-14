# Week 7 Real-World Integration Audit

**Audit date:** 2026-11-14  
**Branch:** `main`  
**Original audit HEAD:** `56999ad3c7f5a822b121b1d7dfefebef40541ff3`
**Follow-up baseline HEAD:** `e1212272d76f787d2ae73515def13b1fe7d215cd` (`main`)
**Environment:** Windows, Python 3.9.13, pytest 8.4.2

## Original audit worktree (at 56999)

Before the audit, the worktree already contained a modified `docs/multicamera_rtsp_benchmark_report.md` and untracked `telegram_test_error.txt` and `telegram_test_result.txt`. These were preserved. The full suite updates the benchmark report timestamp; its pre-suite bytes were restored after each full run. No test tokens or text-file contents were read into the report.

Implementation/test changes made for this audit are limited to `src/camera_manager.py`, `tests/test_dynamic_camera_selection.py`, and `tests/test_week7_camera.py`; this report is the requested audit deliverable.

## Test audit

Full command:

```powershell
.venv\Scripts\python.exe -m pytest -q -rs --basetemp <dedicated-temp-directory>
```

Final result after the inference/shutdown fix: **264 passed, 4 skipped, 0 failed, 14 warnings** in 56.97 seconds. The temporary pytest directory was `firewatch-week7-audit-56999-135058\pytest-basetemp-after-fix` under the Windows user temp directory.

Skipped tests:

| Test | Reason | Week 7 impact | Safe to run here? |
|---|---|---|---|
| `tests/test_week7_integration.py::test_four_rtsp_streams_decode_and_run_detector` | Explicit opt-in requires four authorized RTSP sources (`FIREWATCH_TEST_RTSP_1`…`_4`) and `FIREWATCH_RUN_RTSP_INTEGRATION=1`. None were configured. | Blocks real multi-camera integration acceptance. | No; no authorized streams were provided. |
| `tests/test_week7_performance.py::test_measured_camera_inference_throughput[1]` | Requires performance opt-in and one configured hardware source. | Blocks real 1-camera performance evidence. | No; no source or opt-in was configured. |
| `tests/test_week7_performance.py::test_measured_camera_inference_throughput[4]` | Requires four configured hardware sources and opt-in. | Blocks real 4-camera performance evidence. | No; no sources or opt-in were configured. |
| `tests/test_week7_performance.py::test_measured_camera_inference_throughput[9]` | Requires nine configured hardware sources and opt-in. | Blocks real 9-camera performance evidence. | No; no sources or opt-in were configured. |

All four skips are deliberate external/hardware gates. No skip marker was removed. `ffprobe`-dependent tests did not skip in this environment.

## Camera lifecycle finding and fix

An audit run of the 1/4/9-view dashboard tests reproduced the reported shutdown timeout while switching 9 cameras to 4. The diagnostic identified `firewatch-inference-CAM-05` as still alive after the 1-second inference join. `CameraStreamWorker._infer_loop` called the shared detector with `non_blocking=False`; `SharedFireSmokeDetector.detect_frame` then waited on a shared inference lock held by another camera. That queued inference work could outlast the shutdown join.

The background inference call now uses `non_blocking=True`, consistent with the detector's existing bounded lock-acquisition path. Frames are skipped when another camera owns the model, avoiding an unbounded queue of inference calls. Shutdown timeout behavior and duplicate-worker refusal remain in place. Shutdown errors now log camera ID, source type, state, elapsed time, and the names/IDs of live threads without logging a source URL or credentials.

After the fix:

- `tests/test_dynamic_camera_selection.py`, `tests/test_week7_dashboard.py`, and `tests/test_week7_camera.py`: **31 passed**.
- 9-to-4 transition: **5 separate runs, 5 passed**.
- Shutdown, disconnect/release, and controlled-timeout tests were also repeated five times before the inference fix: **15 passed**; the broader 9-to-4 test subsequently exposed the lock-wait case, which is now covered by the fix and regression run.
- The dedicated regression test asserts inference uses the nonblocking detector path.

Remaining lifecycle limit: an OpenCV `cap.read()` can block in its backend. The capture handle is locally owned by the reader thread and released in that thread's `finally` block. RTSP read timeout is configured for five seconds while shutdown joins the reader for 1.5 seconds; some webcam backends may not honor read timeouts. No real camera/RTSP source was available to measure or resolve backend-specific blocking. Shutdown still reports and retains thread handles rather than claiming success when a thread is alive.

## Dashboard and pipeline audit

Code and automated tests show camera manager startup uses `auto_assign_videos=False`, camera workers begin stopped, and explicit Connect/start actions start workers. Dynamic-selection tests cover 1/4/9 layouts, 9→4→1 transitions, duplicate-start prevention, reconnect, hidden-worker continuity, and fault isolation. View mode changes only the displayed arrangement; monitoring workers are not stopped simply because a camera is hidden. These are backend/unit validations; the Streamlit UI was not launched in a browser during this audit.

The event path is `detect_frame` → per-camera ByteTrack → `TemporalVerifier` → `should_trigger_db_alert` → `EventDatabase.record_event` → `TelegramAlertDispatcher.enqueue`. Telegram tests use mocked HTTP responses and cover event deduplication, persisted delivery state, retry policy, ambiguous timeout handling, restart recovery, and worker shutdown. These establish software behavior under tests, not external Telegram delivery.

Camera lifecycle fixtures use generated local video and do not establish real-camera accuracy or independent 4/9-source throughput. No FPS, latency, CPU, GPU, RAM, VRAM, or production throughput figures are claimed.

## Acceptance status

- **Automated validation: PASS** — final Python 3.9.13 full suite passed (264 passed, 4 documented skips).
- **Camera control validation: PARTIAL** — local prerecorded fixtures and backend controls passed; actual dashboard/browser and hardware behavior were not observed.
- **Telegram mocked validation: PASS** — automated mocked integration tests passed.
- **Real Telegram delivery: NOT VERIFIED** — no operator-authorized rotated credentials or controlled destination were supplied.
- **Real camera/RTSP integration: BLOCKED** — no authorized camera source was supplied.
- **Performance validation: BLOCKED** — opt-in source requirements were unmet; no benchmark result is asserted.
- **Scientific acceptance: NOT VERIFIED** — no real camera event-to-alert evidence was collected.

**Overall Week 7 status: PARTIAL.** Automated validation passes after the targeted inference lock fix, but real-world integration and hardware performance acceptance remain unverified.

## Reproduction

```powershell
.venv\Scripts\python.exe -m pytest -q -rs --basetemp $env:TEMP\firewatch-week7-audit
.venv\Scripts\python.exe -m pytest -q tests/test_dynamic_camera_selection.py tests/test_week7_dashboard.py tests/test_week7_camera.py
```

For real integration, first set the RTSP/performance opt-in only after configuring authorized sources as documented in `tests/test_week7_integration.py` and `tests/test_week7_performance.py`. Telegram testing must use newly rotated credentials and a controlled destination; never reuse exposed credentials.

## Follow-up: inference accounting and acceptance hardening

**Follow-up date:** 2026-11-14
**Branch and baseline:** `main` at `e1212272d76f787d2ae73515def13b1fe7d215cd`; the worktree was clean at the start of the follow-up.
**Resulting changes:** local, uncommitted changes to `src/detector.py`, `src/camera_manager.py`, `web_app.py`, `.gitignore`, `tests/test_week7_camera.py`, and new `tests/test_detector_accounting.py`.

The detector now exposes `DetectionResult` with explicit `COMPLETED`, `BUSY`, `UNAVAILABLE`, and `INVALID_INPUT` outcomes. The legacy tuple API remains available and keeps its `None` busy sentinel. Per-camera lifetime counters distinguish sampled captures, requests, rate-limit skips, detector attempts, completed outputs, busy skips, unavailable/invalid inputs, exceptions, and frames passed to downstream processing. A completed empty detection list counts as completed with zero outputs; a busy lock outcome no longer appears as a completed call. The dashboard shows cumulative calls, completions, busy outcomes, and errors. Legacy `detector_inference_calls` remains an alias for detector attempts; `processed_frames` continues to count frames sent through tracker and temporal processing. These are cumulative totals for the current worker instance, not interval rates.

The model lock remains a shared, non-FIFO `threading.Lock`; the nonblocking path can wait up to 4 ms before returning `BUSY`. A finite synthetic diagnostic used one generated neutral MJPG video as the source for nine `VIDEO_TEST` workers and the repository model. During 8 seconds, every worker completed 57 to 82 inferences, each reported zero errors, and each recorded 99 to 125 busy outcomes out of 181 to 184 attempts. Busy shares ranged from 54.7% to 68.7%. No worker starved during this run, but throughput was uneven and contention was substantial. This is a scheduler liveness diagnostic on repeated synthetic input; it is not a camera benchmark, does not measure fire recall, and cannot prove fairness under other timings. A round-robin arbiter should be evaluated if authorized multi-source runs show prolonged per-camera gaps or zero progress.

Temporal features and cooldown currently receive `time.time()` after inference, while end-to-end latency uses the capture-time monotonic clock. Under heavy inference delay, temporal spacing can therefore reflect processing time rather than source-frame capture time. This remains a timing/provenance risk; no temporal thresholds or event-quality claims were changed without controlled ground truth.

### Follow-up test provenance

Focused command: `tests/test_detector_accounting.py tests/test_week7_camera.py`, using Python 3.9.13, `--basetemp` in the OS temp directory, and `-p no:cacheprovider`: **9 passed, 14 warnings**.

Full command:

```powershell
.venv\Scripts\python.exe -m pytest -q -rs --tb=short -p no:cacheprovider --basetemp="<unique OS temp directory>"
```

Result on the follow-up working tree based on `e1212272d76f787d2ae73515def13b1fe7d215cd`: **269 passed, 4 skipped, 0 failed, 14 warnings** in 43.78 seconds. The 4 skips remain the opt-in four-stream RTSP test and the 1/4/9 hardware throughput cases. No RTSP credentials, camera endpoints, or performance opt-ins were configured. The suite changed the benchmark report timestamp; the exact pre-run bytes were restored and the original SHA-256 was preserved. Do not attribute this result to the baseline commit by itself: it covers the uncommitted follow-up worktree.

### Telegram diagnostic artifacts

`telegram_test_error.txt` (4,048 bytes) and `telegram_test_result.txt` (145,420 bytes) are tracked in the baseline commit and are not covered by the previous ignore rules. An in-memory pattern scan found no standard Bot API token shape or labeled credential assignment; this heuristic does not certify the files as secret-free. Exact root-level ignore rules were added, but tracked files remain tracked. To remove them from the index later while retaining the working copies, use:

```powershell
git rm --cached -- telegram_test_error.txt telegram_test_result.txt
```

That command was not run; file contents were not copied into this report.

### Follow-up acceptance status

- **Inference accounting and detector status tests: PASS** — busy, completed-empty, unavailable, invalid-input, and exception paths are distinct and covered.
- **Full automated suite: PASS** — 269 passed and 4 documented hardware/external skips on the follow-up working tree.
- **Synthetic 9-worker scheduler liveness: PARTIAL** — all 9 workers made progress in this 8-second run; high busy rates and non-FIFO fairness remain concerns.
- **Real camera/RTSP integration and hardware performance: NOT VERIFIED** — no authorized sources were configured.
- **Telegram external delivery: NOT VERIFIED** — tests remain mocked; no controlled live destination was used.
- **Temporal event quality/recall: NOT VERIFIED** — no independent real-world ground truth was collected.

**Overall Week 7 status remains PARTIAL.**
