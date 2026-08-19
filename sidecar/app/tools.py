"""Tool calling: the small set of functions the twin can actually invoke
mid-conversation, plus the registry/dispatch layer around them.

Everything the twin says is otherwise frozen at training time or retrieved
from its own memories -- it has no way to know today's date or reach
anything live. These tools are the exception, and they're deliberately few
and narrow: the persona prompt's core promise is that the twin never states
a fact it isn't grounded in, so a tool either returns real data or says
plainly that it couldn't.

Tool results are fed back to the model as `tool` messages and it writes the
final answer itself, so results here are formatted as short readable prose
rather than JSON -- small models paraphrase prose far more reliably than
they read nested objects.
"""
import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import config, db, google_calendar, weather

logger = logging.getLogger(__name__)

# Neutral fallback for the handful of tool strings that need to name the
# owner directly (rather than as a first-person "you"), for the brief window
# before onboarding sets a real owner_name -- see config.DEFAULT_OWNER_NAME.
_FALLBACK_OWNER_NAME = "the owner"


def _owner_name() -> str:
    return db.get_setting("owner_name", config.DEFAULT_OWNER_NAME) or _FALLBACK_OWNER_NAME

# Feature key -> the settings row that enables it. Tools are grouped by
# feature rather than one toggle per function so that e.g. creating and
# reading calendar events turn on together, which is how someone thinks
# about it ("is the calendar connected?"), not as two independent switches.
FEATURE_CLOCK = "clock"
FEATURE_WEATHER = "weather"
FEATURE_CALENDAR = "calendar"


def feature_enabled(feature: str) -> bool:
    return db.get_setting(f"tools_{feature}_enabled", "1") == "1"


def set_feature_enabled(feature: str, enabled: bool) -> None:
    db.set_setting(f"tools_{feature}_enabled", "1" if enabled else "0")


# --- clock -----------------------------------------------------------------


def resolve_timezone() -> ZoneInfo | None:
    """The configured IANA timezone, or None to mean "whatever this Mac is
    set to". Returns None rather than raising on a bad name -- a typo in a
    setting shouldn't make the twin unable to tell the time."""
    name = db.get_setting("tools_timezone", "").strip()
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning("ignoring unknown tools_timezone %r; using system local time", name)
        return None


def now() -> datetime:
    """Current local time, timezone-aware."""
    zone = resolve_timezone()
    return datetime.now(zone) if zone else datetime.now().astimezone()


def format_datetime(moment: datetime) -> str:
    """The one place the twin's date/time wording is defined, shared by the
    tool result and persona.py's always-on system-prompt line so the two can
    never drift into disagreeing with each other."""
    return moment.strftime("%A, %d %B %Y, %H:%M (%Z)")


async def _get_current_datetime(_arguments: dict, _context: dict) -> str:
    moment = now()
    return (
        f"Right now it is {format_datetime(moment)}. "
        f"Today is day {moment.day} of {moment.strftime('%B')} {moment.year}."
    )


# --- weather -----------------------------------------------------------


async def _get_weather(arguments: dict, _context: dict) -> str:
    location = arguments.get("location")
    if not location:
        return "No location was given, so the weather couldn't be looked up."
    return await weather.get_weather(location)


# --- calendar ------------------------------------------------------------


async def _list_upcoming_events(_arguments: dict, _context: dict) -> str:
    if not google_calendar.is_connected():
        return (
            f"Google Calendar isn't connected yet -- ask {_owner_name()} to connect it "
            "in the Tools tab."
        )
    events = await google_calendar.list_upcoming_events()
    if not events:
        return "There's nothing coming up on any calendar."
    lines = []
    for event in events:
        summary = event.get("summary", "(untitled)")
        start = event.get("start", {}).get("dateTime") or event.get("start", {}).get("date", "?")
        calendar_name = event.get("_calendar")
        suffix = f" [{calendar_name}]" if calendar_name else ""
        lines.append(f"- {summary} at {start}{suffix}")
    return "Upcoming events, across all calendars:\n" + "\n".join(lines)


async def _create_calendar_event(arguments: dict, context: dict) -> str:
    if not google_calendar.is_connected():
        return (
            f"Google Calendar isn't connected yet -- ask {_owner_name()} to connect it "
            "in the Tools tab."
        )
    title = arguments.get("title")
    start_at = arguments.get("start_at")
    if not title or not start_at:
        return "Couldn't create the event -- both a title and a start time are needed."
    try:
        duration_minutes = int(arguments.get("duration_minutes") or 60)
    except (TypeError, ValueError):
        duration_minutes = 60
    description = arguments.get("description") or ""

    if db.get_setting("tools_calendar_confirm", "1") == "1":
        conversation_id = context.get("conversation_id")
        if not conversation_id:
            return "Couldn't draft the event -- there's no active conversation to attach it to."
        db.create_pending_calendar_event(conversation_id, title, start_at, duration_minutes, description)
        return (
            f"Drafted '{title}' at {start_at} -- {_owner_name()} needs to confirm it in the "
            "Talk tab before it's actually added to the calendar."
        )

    try:
        event = await google_calendar.create_event(title, start_at, duration_minutes, description)
    except (google_calendar.GoogleCalendarUnavailable, ValueError) as err:
        return f"Couldn't create the event: {err}"
    link = event.get("htmlLink", "")
    return f"Created '{title}' at {start_at}." + (f" {link}" if link else "")


# --- registry --------------------------------------------------------------

# Descriptions are kept terse on purpose. Every spec is re-sent with every
# single chat turn, so wordy ones are a permanent tax on the context window
# (and on prefill time) for zero accuracy gain at this small a tool count.
SPECS: list[dict] = [
    {
        "feature": FEATURE_CLOCK,
        "spec": {
            "type": "function",
            "function": {
                "name": "get_current_datetime",
                "description": (
                    "Current date and time. Use for date arithmetic, e.g. what day "
                    "of the week a date falls on or how long until something."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    },
    {
        "feature": FEATURE_WEATHER,
        "spec": {
            "type": "function",
            "function": {
                "name": "get_weather",
                # "{locations_hint}" is filled in by available_specs() with
                # the currently configured location names (or left blank
                # when none are configured yet), so the hint always matches
                # what's actually in the Tools tab.
                "description": "Current weather and today's high/low for a place.{locations_hint}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "A city name, e.g. Lisbon."},
                    },
                    "required": ["location"],
                },
            },
        },
    },
    {
        "feature": FEATURE_CALENDAR,
        "spec": {
            "type": "function",
            "function": {
                "name": "list_upcoming_events",
                # "{owner}" is filled in by available_specs() with the current
                # owner_name setting, matching "{calendar}" below.
                "description": "{owner}'s next few calendar events, soonest first, across all calendars.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    },
    {
        "feature": FEATURE_CALENDAR,
        "spec": {
            "type": "function",
            "function": {
                "name": "create_calendar_event",
                # "{calendar}" is filled in by available_specs() with the
                # currently configured write-target calendar name, so the
                # model is never left assuming this can write anywhere else.
                # "{owner}" is filled in the same way, with the owner_name
                # setting.
                "description": (
                    "Adds an event to {owner}'s '{calendar}' calendar -- never any other "
                    "calendar, regardless of what's asked. May need {owner}'s confirmation "
                    "before it's actually created -- say so if asked."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Event title."},
                        "start_at": {
                            "type": "string",
                            "description": "ISO 8601 date-time, e.g. 2026-08-10T14:00:00.",
                        },
                        "duration_minutes": {
                            "type": "integer",
                            "description": "Length in minutes. Defaults to 60.",
                        },
                        "description": {"type": "string", "description": "Optional notes."},
                    },
                    "required": ["title", "start_at"],
                },
            },
        },
    },
]

_HANDLERS: dict[str, Callable[[dict, dict], Awaitable[str]]] = {
    "get_current_datetime": _get_current_datetime,
    "get_weather": _get_weather,
    "list_upcoming_events": _list_upcoming_events,
    "create_calendar_event": _create_calendar_event,
}


def available_specs() -> list[dict]:
    """Ollama-shaped tool specs for the features currently switched on. A
    disabled tool is never offered to the model at all, rather than being
    offered and then refusing -- that saves the tokens and removes any
    chance of the model burning a round trip on something that can't work."""
    specs = []
    for entry in SPECS:
        if not feature_enabled(entry["feature"]):
            continue
        spec = json.loads(json.dumps(entry["spec"]))  # deep copy; callers must not mutate the registry
        if spec["function"]["name"] == "get_weather":
            names = ", ".join(weather.configured_location_names())
            hint = f" Configured: {names}." if names else ""
            spec["function"]["description"] = spec["function"]["description"].format(locations_hint=hint)
        elif spec["function"]["name"] == "list_upcoming_events":
            spec["function"]["description"] = spec["function"]["description"].format(owner=_owner_name())
        elif spec["function"]["name"] == "create_calendar_event":
            spec["function"]["description"] = spec["function"]["description"].format(
                owner=_owner_name(), calendar=google_calendar.write_calendar_name()
            )
        specs.append(spec)
    return specs


def spec_features() -> dict[str, str]:
    """tool name -> feature key, for callers that need to check a specific
    tool's availability (e.g. before recording a run)."""
    return {entry["spec"]["function"]["name"]: entry["feature"] for entry in SPECS}


async def dispatch(name: str, arguments: dict, context: dict | None = None) -> str:
    """Runs one tool and returns its result as prose.

    `context` carries request-scoped data a handler needs but that isn't
    part of the model's own arguments -- currently just conversation_id, for
    create_calendar_event to attach a draft to the right conversation.
    Defaulted to {} rather than required so existing call sites (tests, any
    future one-off dispatch) that don't need it are unaffected.

    Never raises: a tool that fails returns a plain explanation, which goes
    back to the model as the tool result so it can tell the user what went
    wrong in its own voice. Letting an exception escape here would kill the
    whole chat turn mid-stream over one flaky HTTP call.
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        logger.warning("model called unknown tool %r", name)
        return f"There's no tool called {name!r}, so that couldn't be looked up."
    try:
        return await handler(arguments or {}, context or {})
    except Exception as err:  # noqa: BLE001 -- see docstring
        logger.warning("tool %s failed: %s", name, err)
        return f"Couldn't use {name}: {err}"


def normalize_calls(tool_calls: list[dict]) -> list[tuple[str, dict]]:
    """Flattens Ollama's tool_calls into (name, arguments) pairs.

    Ollama hands back `arguments` already decoded as an object, but the
    OpenAI-compatible shape it's modelled on uses a JSON *string*, and
    models occasionally emit one anyway -- so accept both rather than
    crashing on the variant we didn't expect.
    """
    normalized: list[tuple[str, dict]] = []
    for call in tool_calls:
        function = call.get("function") or {}
        name = function.get("name")
        if not name:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning("tool %s got unparseable arguments %r", name, arguments)
                arguments = {}
        normalized.append((name, arguments if isinstance(arguments, dict) else {}))
    return normalized


async def dispatch_all(calls: list[tuple[str, dict]], context: dict | None = None) -> list[str]:
    """Runs a round's tool calls concurrently. A turn like "what's the
    weather in Lisbon and Nairobi" produces two independent network calls in
    one round; running them in sequence would double the stall the user
    sits through for no reason."""
    return list(await asyncio.gather(*(dispatch(name, args, context) for name, args in calls)))
