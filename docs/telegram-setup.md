# Telegram alerts setup

FireWatch reads Telegram credentials and the API admin token from the environment of the process that starts Streamlit. It does not read `.env` files or save credentials to SQLite. The dashboard never shows the configured values. Temporary values entered in the test form are used only for that request.

## Windows PowerShell startup

Use newly rotated credentials if a Bot Token or admin token was previously exposed. Run these commands in the project directory in a fresh PowerShell window. Secure prompts keep the values out of command history; the values remain available to child processes in this window only.

```powershell
$tokenSecure = Read-Host 'New Telegram Bot Token' -AsSecureString
$tokenPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($tokenSecure)
try { $env:TELEGRAM_BOT_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPointer) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPointer); $tokenSecure.Dispose() }

$env:TELEGRAM_CHAT_ID = Read-Host 'Telegram Chat ID'
$adminSecure = Read-Host 'FireWatch admin API token' -AsSecureString
$adminPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($adminSecure)
try { $env:FIREWATCH_API_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($adminPointer) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($adminPointer); $adminSecure.Dispose() }

$env:ALERTS_ENABLED = 'false'
$env:FIREWATCH_API_HOST = '127.0.0.1'
$env:FIREWATCH_API_PORT = '8000'
$env:FIREWATCH_API_INTERNAL_URL = 'http://127.0.0.1:8000'
.venv\Scripts\python.exe -m streamlit run web_app.py
```

The environment values are inherited by the embedded API because it runs in the same Python process. Keep this PowerShell window open while FireWatch is running. To use another API port, update both `FIREWATCH_API_PORT` and `FIREWATCH_API_INTERNAL_URL` to match. Do not run a second FireWatch process against the same database.

## Enable alerts

Open the Telegram page. Confirm that the service Bot Token and Chat ID are configured, unlock administration with `FIREWATCH_API_TOKEN`, and submit the test form with both credential fields empty. This tests the service credentials. Wait for Telegram's acknowledgement, then enable event notifications. If credentials are entered in the form, that test is temporary and does not authorize enabling alerts.

`telegram_settings` stores only the enabled flag. The API acknowledgement timestamp means Telegram accepted the request; it does not show that a person read the message. Network timeouts can have an ambiguous outcome, so check Telegram before retrying a test that reports an unknown result.

The enabled flag persists across restarts. For safety, a restarted process does not run the delivery worker until the service credentials pass a new test in that process; a successful test using temporary credentials does not release it. Disabling preserves queued deliveries, and an already active Telegram request may finish.

## Diagnose service availability

The Telegram page reports missing credential variable names, API startup failure, test result, and enabled state without revealing credential values. A port conflict is reported as an API startup failure. Ensure no unrelated process occupies the configured API port and that `FIREWATCH_API_INTERNAL_URL` points to this FireWatch process. A missing `FIREWATCH_API_TOKEN` prevents management requests even when Telegram credentials are present.

The application does not automatically connect cameras at startup. Webcam access requires an explicit Connect action.
