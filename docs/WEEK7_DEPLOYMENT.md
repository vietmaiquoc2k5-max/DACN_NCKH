# Week 7 Deployment Guide

## Local run

1. Copy `.env.example` to `.env`. Set a random `FIREWATCH_API_TOKEN`, choose an
   existing checkpoint with `FIREWATCH_MODEL_PATH`, and configure exact
   `FIREWATCH_RTSP_HOSTS` entries for any RTSP sources.
2. Install `requirements-dev.txt` when running tests.
3. Launch the Streamlit dashboard with `streamlit run web_app.py`. It creates one
   shared `MultiCameraManager` and starts an embedded FastAPI service on
   `FIREWATCH_API_HOST:FIREWATCH_API_PORT` (default `127.0.0.1:8000`). The API and
   dashboard therefore read and control the same manager, workers, detector, and
   database within that process. Do not also launch a standalone API process on
   the same port/database expecting shared in-memory workers.
4. Dashboard startup leaves camera workers disconnected. Configure and explicitly
   Connect each camera. WebSocket clients first obtain a short-lived ticket via
   the authenticated management API; the long-lived bearer token stays out of
   WebSocket URLs.

For authenticated RTSP cameras, set a variable such as
`CAMERA_RTSP_CAM01=rtsp://user:password@camera.internal/live` in `.env`, allowlist
`camera.internal`, and save the camera source as `env:CAMERA_RTSP_CAM01`. Do not
enter credential-bearing URLs in the dashboard. Local video paths must remain
inside the repository for API registration.

## Docker and MediaMTX simulation

1. Set `FIREWATCH_MODEL_HOST_PATH` in `.env` to an existing local checkpoint path.
   Compose mounts it read-only at `/models/firewatch.pt`; the image does not assume
   a repository-root `best.pt`.
2. Set `.env` values for the API token, RTSP allowlist, origins, and optional
   Telegram integration. Then run `docker compose config`, followed by
   `docker compose up --build`.
3. The dashboard listens on 8501 and embedded API on 8000. MediaMTX listens on
   8554. SQLite data and snapshots use named volumes.
4. Publish four authorized local test-video replays using FFmpeg, for example:

   ```powershell
   ffmpeg -re -stream_loop -1 -i videovietnam\VN_POS_01.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/cam1
   ffmpeg -re -stream_loop -1 -i videovietnam\VN_POS_01.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/cam2
   ffmpeg -re -stream_loop -1 -i videovietnam\VN_POS_01.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/cam3
   ffmpeg -re -stream_loop -1 -i videovietnam\VN_POS_01.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/cam4
   ```

5. Allowlist `mediamtx`, register `rtsp://mediamtx:8554/cam1` through `cam4`,
   and explicitly connect each. These are simulated RTSP streams, not physical
   camera validation.

## CPU and GPU

The image uses its installed PyTorch/Ultralytics runtime. GPU use requires a
compatible NVIDIA container runtime and CUDA-enabled PyTorch build. Measure CPU
and GPU throughput on the target deployment; application settings and FPS caps
do not demonstrate throughput.

## Tests and stability run

Run `python -m compileall src scripts web_app.py alert_manager.py` and
`python -m pytest -q`. RTSP and performance tests are opt-in and marked in
`pytest.ini`. Real Telegram tests require authorized credentials and must be run
separately from mocked unit tests.

The stability acceptance harness requires four source references and at least
7,200 seconds:

```powershell
python scripts/benchmark_week7_stability.py --duration-seconds 7200 `
  --source env:CAMERA_RTSP_CAM01 --source env:CAMERA_RTSP_CAM02 `
  --source env:CAMERA_RTSP_CAM03 --source env:CAMERA_RTSP_CAM04
```

It writes environment metadata, camera metric samples, event timing samples, and
a summary under `runs/benchmarks/week7/`. An interrupted run is recorded as
incomplete. Do not infer physical-camera validation from replay streams or claim
two-hour completion without the generated summary.
