# Week 7 API and Demo Instructions

## Local API

1. Copy `.env.example` to `.env`; set a private random `FIREWATCH_API_TOKEN`, select
   the intended checkpoint path, and explicitly allowlist RTSP hostnames.
2. Install `requirements-dev.txt` into the project environment.
3. Start with `uvicorn src.week7_api:app --host 127.0.0.1 --port 8000`.
4. Check `/health`; use the bearer token for `/api/*` routes. Camera startup is manual.
5. Register a webcam, repository-local video, or allowlisted RTSP camera with
   `POST /api/cameras`, then call `POST /api/cameras/{camera_id}/connect`.
   Camera settings persist in SQLite and restore as disconnected after restart.

The API protects management routes with a shared bearer token. Keep it on a trusted
network or behind TLS and an identity-aware proxy. Do not put bot tokens in frontend
code. `FIREWATCH_RTSP_HOSTS` is an exact hostname allowlist, not a network range.
For authenticated RTSP cameras, submit a source such as `env:CAMERA_RTSP_CAM01` and
define that variable in `.env`. Its RTSP hostname must be allowlisted. The API holds
the resolved URL in memory and masks the complete user-info portion in responses/logs.

## Docker and MediaMTX simulation

1. Set `.env` and confirm `best.pt` is present locally. The checkpoint is bind-mounted
   read-only and is not copied into the image.
2. Run `docker compose up --build`. The API listens on port 8000; MediaMTX listens on
   port 8554. Database and snapshots use named volumes.
3. Publish four authorized local test-video replays from four terminals, replacing the
   sample path with a permitted local video:

   ```powershell
   ffmpeg -re -stream_loop -1 -i videovietnam\VN_POS_01.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/cam1
   ffmpeg -re -stream_loop -1 -i videovietnam\VN_POS_01.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/cam2
   ffmpeg -re -stream_loop -1 -i videovietnam\VN_POS_01.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/cam3
   ffmpeg -re -stream_loop -1 -i videovietnam\VN_POS_01.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/cam4
   ```

4. Register four RTSP sources as `rtsp://mediamtx:8554/cam1` through `cam4`, then
   explicitly connect each one. These are **simulated RTSP replays**, not physical
   camera tests. Observe status, capture/inference rates, reconnection, and event rows.
5. To capture a demo, record the API health check, disconnected startup, explicit
   camera connections, all four feeds, an event with its snapshot, event acknowledgment,
   and orderly disconnect. Label the recording as simulated RTSP unless physical
   cameras were used.

## CPU and GPU

The Docker image uses the installed PyTorch/Ultralytics CPU path unless a compatible
NVIDIA Container Toolkit and CUDA-enabled PyTorch runtime are configured. GPU access
must be added to the deployment host's container runtime and measured independently.
No throughput claim follows from container startup.

## Test and benchmark protocol

Install development dependencies and run `pytest`. For meaningful evidence, record
the source commit, checkpoint SHA-256, hardware/runtime, input dimensions, stream type,
per-camera capture and inference FPS, latency percentiles, CPU/GPU/RAM/VRAM, drops,
reconnects, and timestamps. Run a separate uninterrupted two-hour session. The
Week 7 acceptance report must remain `PARTIAL` until four stream inference, real
Telegram acknowledgment, Docker runtime, target measurements, and two-hour stability
are demonstrated.
