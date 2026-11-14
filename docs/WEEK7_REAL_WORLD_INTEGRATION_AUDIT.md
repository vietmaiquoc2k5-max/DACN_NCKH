# Week 7 Real-World Integration Audit

**Audit date:** 2026-11-14  
**Branch:** `main`  
**HEAD:** `56999ad3c7f5a822b121b1d7dfefebef40541ff3`  
**Environment:** Windows, Python 3.9.13, pytest 8.4.2

## Worktree

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
