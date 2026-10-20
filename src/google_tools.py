# src/google_tools.py
"""
Standalone Google Workspace integration.
- Does NOT import anything from bot.py
- Reads token from env var GOOGLE_TOKEN_JSON (Render-safe)
- Persists refreshed tokens to Qdrant point 5 so they survive restarts
- Lazy imports: the bot still starts if google libs are missing
"""
import os
import json
import threading
import datetime as dt
import logging
from typing import Optional

_log = logging.getLogger(__name__)
_google_lock = threading.Lock()

SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.readonly",
]

_GOOGLE_POINT_ID = 5
_SETTINGS_COLLECTION = os.getenv("SETTINGS_COLLECTION", "ai_employee_settings")

# ---------------------------------------------------------------------------
# Qdrant access (own client, does not touch bot.py's client)
# ---------------------------------------------------------------------------
_qdrant_client = None
_collection_ready = False
_qdrant_lock = threading.Lock()

def _get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        if not url or not api_key:
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set.")
        _qdrant_client = QdrantClient(url=url, api_key=api_key)
    return _qdrant_client

def _ensure_collection():
    global _collection_ready
    if _collection_ready:
        return
    with _qdrant_lock:
        if _collection_ready:
            return
        client = _get_qdrant()
        names = [c.name for c in client.get_collections().collections]
        if _SETTINGS_COLLECTION not in names:
            from qdrant_client.models import VectorParams, Distance
            client.create_collection(
                collection_name=_SETTINGS_COLLECTION,
                vectors_config=VectorParams(size=1, distance=Distance.DOT),
            )
        _collection_ready = True

# ---------------------------------------------------------------------------
# Token loading (Render-safe)
# ---------------------------------------------------------------------------
def _load_token_json() -> Optional[dict]:
    # 1. Qdrant first (refreshed token from previous run)
    try:
        _ensure_collection()
        points = _get_qdrant().retrieve(
            collection_name=_SETTINGS_COLLECTION,
            ids=[_GOOGLE_POINT_ID],
            with_payload=True,
        )
        if points and points[0].payload:
            tok = points[0].payload.get("google_token")
            if isinstance(tok, dict) and tok.get("refresh_token"):
                return tok
    except Exception as e:
        _log.debug(f"[Google] Qdrant token read failed: {e}")

    # 2. Env var (Render)
    raw = os.getenv("GOOGLE_TOKEN_JSON", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            try:
                import base64
                parsed = json.loads(base64.b64decode(raw).decode("utf-8"))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

    # 3. Local file (dev only)
    if os.path.exists("token.json"):
        try:
            with open("token.json", "r") as f:
                parsed = json.load(f)
                if isinstance(parsed, dict):
                    return parsed
        except Exception:
            pass

    return None

def _persist_token(info: dict) -> bool:
    try:
        _ensure_collection()
        from qdrant_client.models import PointStruct
        _get_qdrant().upsert(
            collection_name=_SETTINGS_COLLECTION,
            points=[PointStruct(id=_GOOGLE_POINT_ID, vector=[0.0], payload={"google_token": info})],
        )
        return True
    except Exception as e:
        _log.warning(f"[Google] Could not persist refreshed token: {e}")
        return False

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
def get_google_creds():
    with _google_lock:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request

        info = _load_token_json()
        if not info:
            raise RuntimeError("Google token not found. Set GOOGLE_TOKEN_JSON on Render.")

        creds = Credentials.from_authorized_user_info(info, SCOPES)

        if creds.expired and creds.refresh_token:
            _log.info("[Google] Access token expired, refreshing...")
            creds.refresh(Request())
            _persist_token(json.loads(creds.to_json()))

        if not creds.valid:
            raise RuntimeError("Google token is invalid. Re-run auth_google.py.")
        return creds

def is_google_configured() -> bool:
    try:
        get_google_creds()
        return True
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Calendar & Gmail
# ---------------------------------------------------------------------------
def get_upcoming_events(max_results: int = 10, days_ahead: int = 7) -> str:
    try:
        from googleapiclient.discovery import build
    except ImportError:
        return "(Google libraries not installed.)"

    creds = get_google_creds()
    service = build("calendar", "v3", credentials=creds)

    now = dt.datetime.now(dt.timezone.utc)
    time_max = now + dt.timedelta(days=days_ahead)

    events_result = service.events().list(
        calendarId="primary",
        timeMin=now.isoformat(),
        timeMax=time_max.isoformat(),
        maxResults=max_results,
        singleEvents=True,
        orderBy="startTime",
    ).execute()

    events = events_result.get("items", [])
    if not events:
        return f"No events in the next {days_ahead} days."

    lines = []
    for ev in events:
        start_raw = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date")
        if not start_raw:
            continue
        try:
            if "T" in start_raw:
                start_dt = dt.datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
                start_str = start_dt.astimezone().strftime("%a %d %b, %I:%M %p")
            else:
                start_str = f"{start_raw} (all day)"
        except Exception:
            start_str = start_raw
        summary = (ev.get("summary") or "(no title)").strip()
        lines.append(f"- {start_str}: {summary}")
    return "\n".join(lines)

def get_recent_emails(max_results: int = 5) -> str:
    try:
        from googleapiclient.discovery import build
    except ImportError:
        return "(Google libraries not installed.)"

    creds = get_google_creds()
    service = build("gmail", "v1", credentials=creds)
    results = service.users().messages().list(userId="me", maxResults=max_results).execute()
    messages = results.get("messages", [])
    if not messages:
        return "No recent emails found."

    lines = []
    for msg in messages:
        try:
            msg_data = service.users().messages().get(
                userId="me", id=msg["id"], format="metadata",
                metadataHeaders=["Subject", "From", "Date"],
            ).execute()
            headers = {h["name"]: h["value"] for h in msg_data["payload"]["headers"]}
            lines.append(
                f"- From: {headers.get('From', '?')} | "
                f"Date: {headers.get('Date', '?')} | "
                f"Subject: {headers.get('Subject', '(no subject)')}"
            )
        except Exception as e:
            lines.append(f"- (could not read {msg['id']}: {e})")
    return "\n".join(lines)