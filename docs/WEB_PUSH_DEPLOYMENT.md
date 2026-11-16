# FireWatch Web Push deployment

Web Push is opt-in. The private VAPID key must be supplied through the service environment or a secret manager; never commit it. Configure:

- `FIREWATCH_VAPID_PUBLIC_KEY`: URL-safe base64 VAPID public key.
- `FIREWATCH_VAPID_PRIVATE_KEY`: corresponding private key accepted by `pywebpush`.
- `FIREWATCH_VAPID_SUBJECT`: operator contact such as `mailto:ops@example.com`.
- `WEB_PUSH_ENABLED=true` after keys are installed.
- `FIREWATCH_PUSH_PUBLIC_ORIGIN`: exact public origin, including scheme and port if any.
- `FIREWATCH_PUSH_ALLOWED_HOSTS`: optional comma-separated push service host allowlist, defaulting to FCM, Mozilla, Apple and Windows push hosts.

Install the project requirements so `pywebpush` is available. Terminate HTTPS at the public reverse proxy and route the dashboard and API under the same origin. The browser subscription page is `/notifications/subscribe`; the service worker is `/service-worker.js`. The subscribe page calls the browser permission prompt only after the operator clicks **Subscribe this device**. The browser permission can be denied and cannot be overridden by FireWatch.

The default development ports (Streamlit 8501 and API 8000) are different origins. A browser service worker registered by the API cannot control the Streamlit dashboard, so device subscription is unsupported in that split-port configuration. The Web Push backend can accept subscriptions created on the API origin, but the dashboard subscribe link requires a same-origin reverse proxy and a matching `FIREWATCH_PUSH_PUBLIC_ORIGIN`.

Confirmed event jobs contain the event ID, FIRE/SMOKE label, camera name and ID, confirmation time, configured camera location snapshot, and Google Maps URL. No browser location is read. Camera snapshots remain behind the authenticated event API and are not exposed as public URLs in push payloads.

`ACCEPTED_BY_PUSH_SERVICE` means the push service accepted the request. It does not establish that a browser received or displayed it. HTTP 404/410 subscriptions are removed. Other failures are retried up to `NOTIFICATION_RETRIES` with bounded exponential delay. EmailNotifier is an extension point; SMTP delivery is not enabled until an SMTP provider is configured.

After reverse proxy and key setup, subscribe a test device, grant permission, enable notifications, send a test notification, then manually confirm receipt on a real phone. Phone delivery remains unverified until that manual check.
