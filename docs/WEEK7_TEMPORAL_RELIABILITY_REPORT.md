# Week 7 Temporal Reliability and Multi-Camera Audit

## Repository state

- Branch: `main`
- Initial HEAD: `79c48cb8c4f30aa2e6e464f73b38ef2940a1c8e7`
- Initial working tree: clean
- No commit or push was made.

## Timestamp provenance

| Timestamp | Source and use |
|---|---|
| Source/media PTS | Not currently read. OpenCV backend PTS is not assumed reliable. |
| Capture wall clock | Not retained per frame. Wall clock remains for human-facing and persisted calendar timestamps. |
| Capture monotonic | Captured after `VideoCapture.read()` and atomically stored with frame ID and frame. Used for temporal features and verification. |
| Inference start/completion | Monotonic timing. Used for frame age, inference interval, and latency diagnostics only. |
| Event confirmation | Verification now uses the inferred frame's capture monotonic timestamp for elapsed-time logic. Database event creation uses UTC wall time (`utc_now()`). |
| Alert creation/delivery | Existing UTC database timestamps and Telegram delivery timestamps remain wall-clock values. |

The prior implementation already paired the newest frame, frame ID, and capture monotonic time, but discarded that time before temporal feature extraction and `TemporalVerifier`, substituting `time.time()` after inference. The corrected path propagates the frame's capture monotonic timestamp through both components. Invalid or absent capture time uses an explicit monotonic fallback and increments `temporal_timestamp_fallbacks`. Non-increasing timestamps are discarded and counted. Frame ID and its timestamp are read together under the camera lock.

For prerecorded video, the selected time base is playback acquisition time, not media PTS. This reflects when the decoder delivered frames; it must not be interpreted as source-media duration where playback is faster/slower than real time. Using media PTS requires backend-specific validation and explicit loop/reset semantics.

## Scheduling baseline

Camera inference uses one thread per camera and a shared detector `threading.Lock`. Requests make a bounded 4 ms lock attempt and return `BUSY`; there is no FIFO or round-robin guarantee. Capture and inference FPS are separate. Per-camera snapshots now include measured BUSY ratio, age of last successful inference, largest observed success gap, and frame age at inference. The dashboard displays these values.

Synthetic lock diagnostic command:

```powershell
.\.venv-week7\Scripts\python.exe scripts/benchmark_week7_contention.py --cameras 9 --duration-seconds 8 --request-fps 20 --service-ms 20
```

Observed on this host (8.0168 seconds; synthetic lock only): per-camera completed inferences ranged 10–43, BUSY ratios ranged 72.67%–93.17%, and maximum observed success gaps ranged 0.6008–2.2567 seconds. This is reproducible as a configured workload but host thread timing affects exact results. It is not YOLO, camera, RTSP, event, or fire recall evidence. It demonstrates uneven lock access under saturation and justifies a real-source fairness comparison; it does not alone justify shipping a new scheduling architecture.

No fairness threshold was declared before acceptance measurement. Real inference fairness is therefore **NOT VERIFIED**. Run the opt-in 1/4/9 source performance tests and stability harness with authorized feeds before deciding on an arbiter. Do not interpret synthetic results as camera throughput.

## Event-level and lifecycle acceptance

- Event-level recall, precision, F1, delay, and FAR/hour: **NOT VERIFIED by this audit**. Existing scientific artifacts are not rerun here and no new independent Ground Truth was created.
- RTSP reconnect, Telegram end-to-end, and actual 9-camera monitoring: **NOT VERIFIED**; this environment had no authorized live source configuration.
- Shutdown timeout safety remains in place; no scheduler lifecycle change was made.

## Changes

- Capture-time temporal propagation and explicit invalid/out-of-order handling in `src/camera_manager.py`.
- Per-camera scheduler diagnostics exposed in snapshots and dashboard.
- Deterministic camera regression coverage for frame/timestamp propagation, fallback, and out-of-order rejection.
- Synthetic shared-lock diagnostic in `scripts/benchmark_week7_contention.py`.

## Verification and acceptance

- Focused regression: **39 passed, 4 skipped**.
- Full suite on Python 3.9 (`.venv`): **272 passed, 4 skipped, 0 failed, 14 warnings**.
- Full suite on `.venv-week7` could not collect two modules because that environment lacks PyYAML and FastAPI; no package was installed. The Python 3.9 environment had both dependencies and completed the full suite.
- `git diff --check` and Python compilation of the modified Python files completed without errors.
- Post-change branch and HEAD remain `main` / `79c48cb8c4f30aa2e6e464f73b38ef2940a1c8e7`. The working tree contains only the audit implementation, regression, benchmark, and documentation changes listed above.

| Criterion | Status | Evidence/limit |
|---|---|---|
| Temporal correctness | **PASS** | Capture timestamp semantics documented; deterministic tests confirm same frame/time reaches feature extraction and verification, plus fallback and out-of-order handling. |
| Inference fairness | **PARTIAL / NOT VERIFIED** | Synthetic contention proves uneven access under one saturated workload; no declared acceptance threshold or live 1/4/9-source measurements. |
| Camera lifecycle | **PARTIAL** | Existing lifecycle suite passes; no authorized RTSP source was available for actual disconnect/reconnect and busy-shutdown integration. |
| Event-level quality | **NOT VERIFIED** | No event metrics were newly evaluated; independent Ground Truth and equivalent baseline/candidate run are required. |
| Real-world integration | **NOT VERIFIED** | RTSP and Telegram were not executed. |

Overall Week 7 acceptance: **PARTIAL**. No temporal thresholds, operating points, checkpoint, dataset, or Ground Truth were changed.
