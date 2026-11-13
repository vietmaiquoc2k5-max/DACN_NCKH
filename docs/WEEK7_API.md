# Week 7 API

The FastAPI service runs inside the Streamlit process by default, so API requests
and dashboard actions address the same `MultiCameraManager`, camera workers,
detector, and SQLite database. Set `FIREWATCH_API_HOST`, `FIREWATCH_API_PORT`, and
`FIREWATCH_API_TOKEN` before launch. Use HTTPS or a trusted reverse proxy when
exposing the service beyond localhost.

Management HTTP endpoints require:

```http
Authorization: Bearer <FIREWATCH_API_TOKEN>
```

`GET /health` is unauthenticated for service health checks. It returns `degraded`
when a valid detector checkpoint was not loaded.

## Routes

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/cameras` | Camera status and masked configuration summary. |
| `GET` | `/api/cameras/config-errors` | Restore failures keyed by the original camera ID. |
| `POST` | `/api/cameras` | Add a disconnected camera. |
| `PUT` | `/api/cameras/{camera_id}` | Edit camera configuration; saves remain disconnected. |
| `DELETE` | `/api/cameras/{camera_id}` | Remove a camera configuration. |
| `POST` | `/api/cameras/{camera_id}/connect` | Explicitly start one camera. |
| `POST` | `/api/cameras/{camera_id}/disconnect` | Stop one camera without affecting other feeds. |
| `POST` | `/api/cameras/{camera_id}/reconnect` | Explicitly reconnect one camera. |
| `GET` | `/api/events` | List confirmed events with optional camera/type/status filters. |
| `GET` | `/api/events/{event_id}/snapshot` | Retrieve a validated event image within the snapshot directory. |
| `POST` | `/api/events/{event_id}/acknowledge` | Record operator acknowledgment. |
| `POST` | `/api/events/{event_id}/dismiss` | Record dismissal and its reason. |
| `GET` | `/api/performance` | Capture, inference, display, latency, drops, and queue metrics. |
| `GET` | `/api/alerts/status` | Telegram configuration and dispatcher state. |
| `GET` | `/api/alerts/deliveries` | Delivery states and timestamps, including Telegram acknowledgment. |
| `POST` | `/api/ws-tickets` | Issue one 45-second, single-use camera-stream ticket. |
| WebSocket | `/ws/cameras?ticket=…` | Receive latest camera frames/status after ticket and origin validation. |

Camera creation accepts `WEBCAM`, repository-local `VIDEO_TEST`, or allowlisted
`IP_CAMERA` RTSP/RTSPS sources. Authenticated RTSP URLs must be stored in an
environment variable and referenced as `env:VARIABLE_NAME`; their resolved value
is held in memory and not returned or written to the camera configuration table.
Configure exact destination hostnames with `FIREWATCH_RTSP_HOSTS`.

WebSocket tickets are scoped to opening the camera feed and cannot replace the
HTTP bearer token for management operations. Reconnect by obtaining a new ticket.
Set allowed browser origins in `FIREWATCH_ALLOWED_ORIGINS`. Keep the ticket short
lived and do not log the query string.

Event `event_state` records scientific pipeline confirmation (`CONFIRMED`).
Operator `status` records `UNREAD`, `ACKNOWLEDGED`, or `DISMISSED`; operator action
does not rewrite event confirmation. Telegram `DELIVERED` means Telegram returned
a valid API acknowledgment and message identifier, not that a person read it.
