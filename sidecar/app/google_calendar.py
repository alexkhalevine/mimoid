"""Google Calendar: OAuth (desktop/loopback flow) plus the two operations
the twin's calendar tools use -- creating an event and listing upcoming
ones.

Implemented directly against the Calendar and OAuth REST APIs with httpx,
not google-api-python-client/google-auth-oauthlib. That dependency tree is
large and the first-run bootstrap already installs torch + coqui-tts; the
handful of calls actually needed here (token exchange, token refresh,
events insert/list) are each a single httpx request.

The OAuth flow itself has no existing precedent in this codebase to follow
(see the module docstring notes below at each step) -- it's the first place
the sidecar opens a second listening socket.
"""
import asyncio
import json
import logging
import secrets
import time
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, quote, urlencode, urlparse

import httpx

from . import config, db, network_activity

logger = logging.getLogger(__name__)

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
CALENDAR_LIST_URL = "https://www.googleapis.com/calendar/v3/users/me/calendarList"
# Create and read events only, not full calendar management -- the
# narrowest scope that covers both tools. The scope itself grants access to
# every calendar in the account (Google has no per-calendar scope), but
# _events_url()'s caller decides which calendar id actually gets used --
# reading fans out to all of them, writing is pinned to one, see
# create_event().
SCOPE = "https://www.googleapis.com/auth/calendar.events"

# The one calendar the twin may ever create an event on -- configurable
# (Tools tab), resolved by name against the account's calendar list on
# every write rather than cached, so a rename in Google Calendar can't
# silently point this somewhere stale. Reading is unrestricted (every
# calendar in the account); writing never is, regardless of what the model
# is asked to do -- there is no argument on the tool that can change this.
DEFAULT_WRITE_CALENDAR_NAME = "Tasks"


def _events_url(calendar_id: str) -> str:
    return f"https://www.googleapis.com/calendar/v3/calendars/{quote(calendar_id, safe='')}/events"


def write_calendar_name() -> str:
    return db.get_setting("tools_calendar_write_target", "") or DEFAULT_WRITE_CALENDAR_NAME


def set_write_calendar_name(name: str) -> None:
    db.set_setting("tools_calendar_write_target", name.strip())

# A token that's about to expire is treated as already expired, so a
# request never starts with one that goes stale mid-flight.
_EXPIRY_BUFFER_SECONDS = 60

_CALLBACK_PAGE = (
    "<!doctype html><html><head><title>Mimoid</title></head>"
    '<body style="font-family: -apple-system, sans-serif; text-align: center; padding: 64px 24px;">'
    "<h1>You can close this window.</h1>"
    "<p>Mimoid is finishing the connection to Google Calendar.</p>"
    "</body></html>"
)


class GoogleCalendarUnavailable(Exception):
    """Raised when there's no connected Google account to act on, or no
    client credentials configured to even start connecting one."""


def _resolve_client_credentials() -> tuple[str, str] | None:
    """In-app credentials (saved via the Tools tab) take precedence over the
    MIMOID_GOOGLE_CLIENT_ID/_SECRET env vars, mirroring openrouter.py's
    _resolve_api_key() precedence."""
    client_id = db.get_setting("google_client_id", "") or config.GOOGLE_CLIENT_ID
    client_secret = db.get_setting("google_client_secret", "") or config.GOOGLE_CLIENT_SECRET
    if not client_id or not client_secret:
        return None
    return client_id, client_secret


def set_client_credentials(client_id: str, client_secret: str) -> None:
    db.set_setting("google_client_id", client_id.strip())
    db.set_setting("google_client_secret", client_secret.strip())


# --- token storage -----------------------------------------------------

# One JSON-blob setting for the whole token bundle (access token, refresh
# token, expiry), same shape as tools_weather_locations -- a single flat
# string row rather than dedicated columns, since this is opaque data no
# SQL query ever needs to filter on.


def _load_tokens() -> dict | None:
    raw = db.get_setting("google_calendar_tokens", "")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("google_calendar_tokens setting is corrupt JSON; treating as disconnected")
        return None


def _save_tokens(tokens: dict) -> None:
    db.set_setting("google_calendar_tokens", json.dumps(tokens))


def is_connected() -> bool:
    tokens = _load_tokens()
    return bool(tokens and tokens.get("refresh_token"))


# --- the loopback OAuth flow --------------------------------------------

# At most one connect attempt in flight at a time. A module-level global
# because there's exactly one sidecar process per app instance (same
# reasoning as weather.py's _cache and ollama.py's other process-lifetime
# state) -- there's nothing to key it by.


class _ConnectSession:
    def __init__(self, server: asyncio.AbstractServer, state: str, redirect_uri: str, auth_url: str):
        self.server = server
        self.state = state
        self.redirect_uri = redirect_uri
        self.auth_url = auth_url
        self.received = asyncio.Event()
        self.query: dict[str, str] | None = None
        self.error: str | None = None
        self.done = False
        self.task: asyncio.Task | None = None


_session: _ConnectSession | None = None


async def start_connect() -> str:
    """Starts a one-shot local listener, builds the consent URL for it, and
    returns that URL immediately -- the frontend opens it in the system
    browser (via the Tauri opener plugin) while a background task waits for
    Google's redirect. Callers poll get_status() to learn when it's done.

    Ollama's chat-tools model already runs entirely locally, so this is the
    sidecar's first second listening socket. It exists only long enough to
    catch one redirect and closes itself either way (success, decline, or
    GOOGLE_OAUTH_CALLBACK_TIMEOUT).
    """
    global _session
    credentials = _resolve_client_credentials()
    if credentials is None:
        raise GoogleCalendarUnavailable("Google client ID/secret aren't configured yet.")
    client_id, _client_secret = credentials

    if _session is not None and not _session.done:
        # Already connecting -- reuse the in-flight attempt rather than
        # opening a second listener and orphaning the first (which would
        # then answer Google's actual redirect with a state nothing is
        # waiting on anymore).
        return _session.auth_url

    state = secrets.token_urlsafe(24)
    session: _ConnectSession | None = None  # closed over by handle_callback below

    async def handle_callback(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # `session` (assigned below) is always set before a real connection
        # can arrive -- see the comment at that assignment.
        try:
            request_line = await reader.readline()
            # Drain the header lines (up to the blank line) so the browser's
            # connection isn't left hanging on us mid-request -- none of the
            # headers matter, only the query string on the request line.
            while True:
                line = await reader.readline()
                if not line or line in (b"\r\n", b"\n"):
                    break
            _method, target, _version = request_line.decode("latin-1").split(" ", 2)
            query = parse_qs(urlparse(target).query)
            session.query = {key: values[0] for key, values in query.items()}
        except Exception as err:  # noqa: BLE001 -- a malformed request must not crash the listener
            session.error = f"malformed redirect: {err}"
        finally:
            body = _CALLBACK_PAGE.encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + body
            )
            try:
                await writer.drain()
            finally:
                writer.close()
            session.received.set()

    server = await asyncio.start_server(handle_callback, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    # 127.0.0.1, matching the loopback convention config.py already uses
    # for OLLAMA_BASE_URL -- this must also be registered as an authorized
    # redirect URI on the Google Cloud "Desktop app" OAuth client.
    redirect_uri = f"http://127.0.0.1:{port}/callback"
    auth_url = AUTH_URL + "?" + urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": SCOPE,
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
    )

    # No `await` occurs between here and `server`/`redirect_uri` being
    # computed above, so `handle_callback` (which only runs once a real
    # connection arrives, always after this function has returned control
    # to the event loop) can never observe `session` as still None.
    session = _ConnectSession(server, state, redirect_uri, auth_url)
    _session = session
    session.task = asyncio.create_task(_finish_connect(session))
    return auth_url


async def _finish_connect(session: _ConnectSession) -> None:
    try:
        try:
            await asyncio.wait_for(session.received.wait(), config.GOOGLE_OAUTH_CALLBACK_TIMEOUT)
        except TimeoutError:
            session.error = "Timed out waiting for Google's redirect."
            return
        if session.error:
            return
        query = session.query or {}
        if query.get("error"):
            session.error = f"Google declined: {query['error']}"
            return
        if query.get("state") != session.state:
            session.error = "State mismatch -- discarding this response for safety."
            return
        code = query.get("code")
        if not code:
            session.error = "Google's redirect didn't include an authorization code."
            return
        try:
            await _exchange_code(code, session.redirect_uri)
        except httpx.HTTPError as err:
            session.error = f"Token exchange failed: {err}"
    finally:
        session.done = True
        session.server.close()
        await session.server.wait_closed()


def disconnect() -> None:
    global _session
    db.set_setting("google_calendar_tokens", "")
    if _session is not None and not _session.done:
        _session.error = "Disconnected before Google responded."
        _session.received.set()  # wakes _finish_connect so it tears down promptly
    _session = None


def get_status() -> dict:
    connecting = _session is not None and not _session.done
    error = _session.error if _session is not None and _session.done else None
    return {
        "configured": _resolve_client_credentials() is not None,
        "connected": is_connected(),
        "connecting": connecting,
        "error": error,
    }


async def _exchange_code(code: str, redirect_uri: str) -> None:
    client_id, client_secret = _resolve_client_credentials()  # already validated in start_connect
    with network_activity.track("Google: connecting calendar account"):
        async with httpx.AsyncClient(timeout=config.GOOGLE_HTTP_TIMEOUT) as client:
            response = await client.post(
                TOKEN_URL,
                data={
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
            )
            response.raise_for_status()
    payload = response.json()
    _save_tokens(
        {
            "access_token": payload["access_token"],
            "refresh_token": payload["refresh_token"],
            "expires_at": time.time() + payload.get("expires_in", 3600),
        }
    )


async def _refresh_access_token(refresh_token: str) -> str:
    client_id, client_secret = _resolve_client_credentials() or (None, None)
    if client_id is None:
        raise GoogleCalendarUnavailable("Google client ID/secret aren't configured anymore.")
    with network_activity.track("Google: refreshing calendar access"):
        async with httpx.AsyncClient(timeout=config.GOOGLE_HTTP_TIMEOUT) as client:
            response = await client.post(
                TOKEN_URL,
                data={
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "grant_type": "refresh_token",
                },
            )
            response.raise_for_status()
    payload = response.json()
    access_token = payload["access_token"]
    _save_tokens(
        {
            "access_token": access_token,
            # A refresh response never repeats the refresh_token -- Google
            # only issues a new one on first consent -- so the existing one
            # has to be carried forward or the next refresh would have
            # nothing to use.
            "refresh_token": refresh_token,
            "expires_at": time.time() + payload.get("expires_in", 3600),
        }
    )
    return access_token


async def _access_token() -> str:
    """A currently-valid access token, refreshing first if the cached one
    is missing or close to expiry. Callers never see a stale token -- that
    would otherwise surface as an opaque 401 from Google instead of this
    module's own clear "isn't connected" message."""
    tokens = _load_tokens()
    if not tokens or not tokens.get("refresh_token"):
        raise GoogleCalendarUnavailable("Google Calendar isn't connected yet.")
    if tokens.get("access_token") and tokens.get("expires_at", 0) - _EXPIRY_BUFFER_SECONDS > time.time():
        return tokens["access_token"]
    return await _refresh_access_token(tokens["refresh_token"])


# --- calendar operations -------------------------------------------------


async def list_calendars(client: httpx.AsyncClient, access_token: str) -> list[dict]:
    """Every calendar in the connected account, as [{id, summary}]. The set
    list_upcoming_events fans reads out across, and create_event resolves
    its one permitted write-target name against."""
    response = await client.get(
        CALENDAR_LIST_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        params={"minAccessRole": "reader"},
    )
    response.raise_for_status()
    return [
        {"id": item["id"], "summary": item.get("summary", item["id"])}
        for item in response.json().get("items", [])
    ]


def _resolve_calendar_id(name: str, calendars: list[dict]) -> str | None:
    """Case-insensitive exact match on name, falling back to a substring
    match -- same matching style as weather.py's _resolve_location. Returns
    None on no match rather than guessing at a fallback: unlike weather,
    where a miss just means "geocode it instead", a miss here means the
    write-target calendar the user named doesn't exist, and writing
    anywhere else would violate the one rule this tool has to keep."""
    needle = name.strip().lower()
    for calendar in calendars:
        if needle == calendar["summary"].lower() or needle == calendar["id"].lower():
            return calendar["id"]
    for calendar in calendars:
        if needle in calendar["summary"].lower():
            return calendar["id"]
    return None


async def create_event(title: str, start_at: str, duration_minutes: int, description: str = "") -> dict:
    """Creates the event on the *one* calendar configured as the write
    target (write_calendar_name(), "Tasks" by default) -- never anywhere
    else, and never a calendar the model gets to pick. Resolved by name
    fresh on every call rather than cached, so renaming the calendar in
    Google Calendar can't silently point this at a stale id.

    start_at is an ISO 8601 datetime (no timezone offset is assumed to be
    present; Google treats a naive one as the calendar's own timezone).
    Returns Google's event resource -- callers use `htmlLink` for a receipt
    the user can click through to.
    """
    access_token = await _access_token()
    start = datetime.fromisoformat(start_at)
    end = start + timedelta(minutes=duration_minutes)
    target_name = write_calendar_name()
    with network_activity.track(f"Google Calendar: creating '{title}'"):
        async with httpx.AsyncClient(timeout=config.GOOGLE_HTTP_TIMEOUT) as client:
            calendars = await list_calendars(client, access_token)
            calendar_id = _resolve_calendar_id(target_name, calendars)
            if calendar_id is None:
                raise GoogleCalendarUnavailable(
                    f"No calendar named '{target_name}' was found in this Google account -- "
                    "check the name in the Tools tab."
                )
            response = await client.post(
                _events_url(calendar_id),
                headers={"Authorization": f"Bearer {access_token}"},
                json={
                    "summary": title,
                    "description": description,
                    "start": {"dateTime": start.isoformat()},
                    "end": {"dateTime": end.isoformat()},
                },
            )
            response.raise_for_status()
    return response.json()


async def list_upcoming_events(max_results: int = 5) -> list[dict]:
    """Upcoming events merged across *every* calendar in the account,
    soonest first. Google's API has no single "all calendars" events
    endpoint, so this fans out to each calendar's own events.list
    concurrently -- fetching N calendars one at a time would multiply the
    wait by N for no reason (same reasoning as the weather tool's
    multi-location fan-out). One calendar failing to list (e.g. a shared
    calendar whose access was revoked) is logged and skipped rather than
    failing the whole lookup.
    """
    access_token = await _access_token()

    async def _events_for(client: httpx.AsyncClient, calendar: dict) -> list[dict]:
        response = await client.get(
            _events_url(calendar["id"]),
            headers={"Authorization": f"Bearer {access_token}"},
            params={
                "timeMin": datetime.now(UTC).isoformat(),
                "maxResults": max_results,
                "singleEvents": "true",
                "orderBy": "startTime",
            },
        )
        response.raise_for_status()
        events = response.json().get("items", [])
        for event in events:
            event["_calendar"] = calendar["summary"]
        return events

    # The calendar-list lookup has to be inside the tracked block too, not
    # just the per-calendar fetches after it -- it's a real network call
    # like any other, and leaving it out would make it silently missing
    # from the footer's internet-access list.
    async with httpx.AsyncClient(timeout=config.GOOGLE_HTTP_TIMEOUT) as client:
        with network_activity.track("Google Calendar: checking upcoming events"):
            calendars = await list_calendars(client, access_token)
            results = await asyncio.gather(
                *(_events_for(client, calendar) for calendar in calendars), return_exceptions=True
            )

    all_events: list[dict] = []
    for calendar, result in zip(calendars, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning("failed to list events for calendar %r: %s", calendar["summary"], result)
            continue
        all_events.extend(result)

    def _start_key(event: dict) -> str:
        start = event.get("start", {})
        return start.get("dateTime") or start.get("date") or ""

    all_events.sort(key=_start_key)
    return all_events[:max_results]
