# Week 7 Technical Report

## Repository and provenance

- Initial branch: `main`
- Initial HEAD: `1f9aab396ba7cb9a3a671e174700f7f474560897`
- Initial working tree: clean
- Reference commit: `1f9aab396ba7cb9a3a671e174700f7f474560897` (same as initial HEAD)
- Final HEAD: unchanged; this work is uncommitted.
- Week 4 operating point: **PROVISIONAL**. Week 7 work did not modify datasets,
  Ground Truth, test split, model checkpoints, or historical benchmark artifacts.

## Implementation status

The existing `MultiCameraManager`, camera workers, shared detector, ByteTrack,
temporal verification, event evaluator, SQLite event store, and Streamlit UI were
retained. Changes add persistent safe camera references, a durable SQLite-backed
Telegram queue with explicit ambiguous-outcome handling, scoped short-lived
WebSocket tickets, measured per-camera processing/display metrics, and an embedded
FastAPI service that uses the same manager instance as Streamlit. Camera layout
selection is a presentation choice; individual controls perform connect,
disconnect, and reconnect actions.

Camera settings restore disconnected with stable IDs. RTSP credentials are held
in environment variables and source configurations persist references such as
`env:CAMERA_RTSP_CAM01`. Model selection no longer falls through from an explicit
missing checkpoint to an unrelated checkpoint.

## Verification evidence

| Area | Status | Evidence / limitation |
|---|---|---|
| Python syntax | TESTED | `python -m compileall src scripts web_app.py alert_manager.py tests`; passed. Two pre-existing invalid escape warnings remain in `scripts/evaluate_full_frame_far.py`. |
| Week 7 automated tests | TESTED | Targeted Week 7 command: 26 passed, 4 skipped in Python 3.14.3 / pytest 8.2.1. Skips require RTSP or hardware. |
| Full automated suite | FAIL / ENVIRONMENT LIMITED | `python -m pytest -q`: 200 passed, 4 skipped, 14 failed, 25 errors. The failures/errors predominantly require the missing `lap` ByteTrack dependency; Ultralytics attempted an automatic install but could not write into the user site. `lap>=0.5.12` is now declared in runtime requirements. |
| RTSP and detector | NOT VERIFIED | No authorized RTSP sources were available. Opt-in test is `tests/test_week7_integration.py`. |
| Telegram | NOT VERIFIED | No credentials were available; unit tests use mocks and do not imply Telegram acceptance. |
| Docker/MediaMTX | BLOCKED | Docker CLI is unavailable; compose validation, image build, service health, and simulated RTSP were not run. |
| Performance targets | NOT VERIFIED | No measured 1/4/9-camera hardware run. The performance test records observed inference FPS; configured caps are not evidence. |
| Two-hour stability | NOT VERIFIED | Harness added at `scripts/benchmark_week7_stability.py`; no runtime result is claimed. |

The available Python environment also lacks Streamlit and `lap`; the dashboard
could not be launched and repository tests requiring ByteTrack could not execute.
Compile success is syntax evidence only and does not establish runtime correctness.
See `docs/WEEK7_REPAIR_AUDIT.md` and
`docs/WEEK7_ACCEPTANCE.md` for detailed findings and outstanding criteria.

## Conclusion

**PARTIAL — NOT ACCEPTED.** Source fixes and test/benchmark tooling are present,
but the project does not yet have the four-stream, real Telegram, Docker,
hardware-throughput, and two-hour runtime evidence required for Week 7 acceptance.
