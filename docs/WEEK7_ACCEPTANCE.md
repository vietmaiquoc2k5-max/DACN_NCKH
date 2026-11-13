# Week 7 Acceptance Checklist

**Current classification: PARTIAL — NOT ACCEPTED.** Source implementation is not
equivalent to operational verification.

| Criterion | Status | Evidence / next action |
|---|---|---|
| Four concurrent RTSP feeds | NOT VERIFIED | Run four authorized MediaMTX replays or cameras and save the result. |
| Real detector inference on feeds | NOT VERIFIED | Execute `tests/test_week7_integration.py` with the required RTSP environment and a named checkpoint. |
| Tracking and temporal verification | IMPLEMENTED / NOT VERIFIED | Existing pipeline is reused; record a live inference run. |
| SQLite event persistence | IMPLEMENTED / TESTS NOT RUN | Run the Week 7 database tests in an environment with pytest. |
| Real Telegram snapshot acknowledgment | NOT VERIFIED | Supply authorized credentials, send a real snapshot in a separately recorded integration run. |
| Dashboard and API share one backend | IMPLEMENTED | Streamlit starts an embedded API over its manager instance; verify both surfaces against a live camera. |
| Cameras disconnected at startup | IMPLEMENTED / NOT VERIFIED | Manager restores disconnected; verify with a fresh process and no auto-start calls. |
| Connect, disconnect, reconnect | IMPLEMENTED / NOT VERIFIED | Exercise the dashboard controls on a permitted stream. |
| Docker deployment | BLOCKED | Docker CLI unavailable in this environment; run compose config/build/up/ps/logs on a Docker host. |
| CPU/GPU throughput targets | NOT VERIFIED | Run hardware test with 1, 4, and 9 sources; retain measured rates and resource data. |
| Alert latency | IMPLEMENTED / NOT VERIFIED | Delivery records store event-confirmed, queued, attempt-started, and Telegram acknowledgment timestamps; capture real delivery records. |
| Two-hour stability | NOT VERIFIED | Run `scripts/benchmark_week7_stability.py` for at least 7,200 seconds with four concurrent streams. |
| Week 7 focused tests | TESTED | 26 passed, 4 skipped. |
| Full test suite | FAIL / ENVIRONMENT LIMITED | 200 passed, 4 skipped, 14 failed, 25 errors; tracker-dependent cases could not import BYTETracker because `lap` is absent. Install declared requirements in a writable project environment and rerun. |
| Scientific operating point | PROVISIONAL | No Week 4 threshold, Ground Truth, test split, or benchmark artifact was modified. |

## Evidence status vocabulary

- **IMPLEMENTED**: code is present.
- **TESTED**: a test command actually ran and passed.
- **VERIFIED**: behavior was observed in the declared runtime environment.
- **ACCEPTED**: all required runtime and test criteria have evidence.

No acceptance-level claim is made for runtime conditions that were unavailable.
