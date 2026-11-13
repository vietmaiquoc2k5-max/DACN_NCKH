# Week 7 Repair Audit

## Repository state

- Branch: `main`
- Initial HEAD: `1f9aab396ba7cb9a3a671e174700f7f474560897`
- Initial working tree: clean (`git status --short` returned no entries)
- Reference commit: same as HEAD; no code or history delta from the supplied reference.
- Recent history: Week 7 API/deployment commit atop Week 4 validation and evidence commits.

## Existing functionality

- `MultiCameraManager` and `CameraStreamWorker` implement manual camera startup, one worker per configured camera, OpenCV decoding, shared detector access, per-camera ByteTrack/temporal verifier state, snapshot generation, and local SQLite event records.
- Detector and Week 4 scientific modules are present. Existing Streamlit UI reads a cached in-process manager and implements explicit start/stop controls.
- The new FastAPI module, camera configuration table, Telegram worker, Docker files, and a small API test module exist at the reference commit.

## Confirmed defects and gaps

| Severity | Finding | Evidence / affected modules |
|---|---|---|
| P1 | API and Streamlit each create their own `MultiCameraManager`, detector and worker set. | `src/week7_api.py` lifespan and `web_app.py:get_firewatch_station` |
| P1 | WebSocket camera feed accepts the long-lived API bearer token in a query parameter. | `/ws/cameras` in `src/week7_api.py` |
| P1 | Telegram delivery state is stored, but work and the event payload live only in an in-memory queue; startup does not recover `QUEUED` work. No attempt state or ambiguity policy exists. | `src/telegram_alerts.py` |
| P1 | Camera restore validates a credential-bearing environment reference into a URL and puts that URL in the manager config. Restore errors are only logged by exception type; IDs outside the default slots can be recreated under another ID. | `_validate_source`, `_restore_camera_configs` in `src/week7_api.py` |
| P1 | Inference FPS uses an enforced cap as its only throughput metric; no processed frame count, display FPS, end-to-end latency, dropped-frame count, or queue depth is reported. The nonblocking shared-detector path is not used by camera inference. | `CameraStreamWorker._infer_loop`, `get_snapshot`, `get_system_metrics` |
| P1 | Dashboard is not a client of FastAPI and cannot share the API manager; API has no live-frame endpoint suitable for Streamlit. | `web_app.py` singleton versus FastAPI lifespan |
| P2 | Delivery rows lack a foreign key to events and event/delivery transitions are not atomic. SQLite WAL is enabled by one dispatcher, while legacy alert operations use separate connection/lock paths. | `src/telegram_alerts.py`, `alert_manager.py`, `src/database.py` |
| P2 | Event dismissal is persisted as `ACKNOWLEDGED` in the legacy database wrapper; operational acknowledgment is not the same as scientific confirmation. | `src/database.py: dismiss_event`; event schema in `alert_manager.py` |
| P2 | API coverage is narrow; camera restoration, WebSocket authentication, event integrity, queue recovery, and dashboard/backend sharing have no tests. | `tests/test_week7_api.py` |
| P2 | Compose bind-mounts `./best.pt` unconditionally; model loading falls back across candidates, which can silently select a different checkpoint. | `docker-compose.yml`, `src/config.py`, `src/detector.py` |

## Runtime and evidence

- Available in the executed Python 3.14.3 environment: OpenCV, PyTorch, Ultralytics, FastAPI, uvicorn, httpx, pytest, and psutil; checkpoint files exist.
- Streamlit is not installed in the executed Python environment. The ByteTrack `lap` dependency is also missing; Ultralytics attempted an automatic installation during the full suite, but the user site was not writable. `lap>=0.5.12` is now declared in `requirements.txt`.
- Docker CLI was not found. No running RTSP publisher or Telegram credentials were supplied.
- Four-stream RTSP, real alert delivery, CPU/GPU target throughput, container behavior, and 7,200-second stability are therefore not evidenced by this audit.

## Repair outcomes

- Streamlit now starts an embedded FastAPI service with the same cached manager; a new browser session no longer globally stops existing workers. Layout selection only changes visible cameras, and per-camera Connect/Disconnect/Reconnect controls operate on the selected worker.
- Management WebSockets use short-lived one-use tickets with exact origin checks. RTSP settings persist environment references, and camera restore errors retain the original camera ID in the error report.
- Telegram sends are queued in SQLite with delivery timestamps, attempts, retry/backoff, capacity failures, and a conservative AMBIGUOUS state after uncertain outcomes. Delivery records require an existing event.
- Metrics report observed processed-frame rate, inference and end-to-end latency, capture/inference/display counters, drops, and queue depth. The two-hour stability harness and opt-in external/hardware tests were added.
- Event operator acknowledgment/dismissal remains separate from the scientific `CONFIRMED` event state.
- The focused Week 7 suite passed: 26 passed, 4 skipped. The full suite completed with 200 passed, 4 skipped, 14 failed, and 25 errors; tracker-dependent cases were blocked by missing `lap`.
- Compose, live RTSP, real Telegram, Streamlit launch, hardware throughput, and two-hour runtime remain NOT VERIFIED or BLOCKED.

## Scientific constraints

- Week 4 Operating Point remains `PROVISIONAL`; this repair will not change thresholds, Ground Truth, test splits, checkpoint weights, or historical benchmarks.
- Existing per-camera tracking and temporal verification are reusable. Week 7 operational tests do not establish FAR/hour or scientific acceptance.
