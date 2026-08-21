import asyncio
import json
import logging
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from . import (
    activity,
    audio,
    backup,
    config,
    db,
    google_calendar,
    memory,
    network_activity,
    ollama,
    openrouter,
    persona,
    signal,
    stt,
    tools,
    tts,
    vault,
    weather,
    whatsapp,
)

logger = logging.getLogger(__name__)

OLLAMA_UNAVAILABLE = HTTPException(
    status_code=503, detail="Ollama is not reachable. Start it and try again."
)
STT_UNAVAILABLE = HTTPException(
    status_code=503,
    detail="Speech-to-text isn't available on this machine. Reinstall the sidecar dependencies and try again.",
)
# On first use, Whisper downloads its model (~1-2 GB) before it can
# transcribe anything -- generous, but bounded so a stalled/blocked download
# fails with a clear message instead of hanging the request forever.
STT_TIMEOUT_SECONDS = 480
# How long the voice loop waits, after you stop talking, before it ends the turn
# and replies. Mirrors SILENCE_MS in src/vad.ts; the bounds are what the Talk
# tab's slider offers, with headroom on either side.
DEFAULT_PAUSE_MS = 2000
MIN_PAUSE_MS = 500
MAX_PAUSE_MS = 5000
TTS_UNAVAILABLE = HTTPException(
    status_code=503, detail="Text-to-speech isn't available on this machine."
)
# On first use, loading the voice model downloads it (~1.8 GB) before any
# synthesis can happen -- generous, but bounded so a stalled/blocked download
# fails with a clear message instead of hanging the request forever.
TTS_TIMEOUT_SECONDS = 600
OPENROUTER_UNAVAILABLE = HTTPException(
    status_code=400,
    detail="OpenRouter isn't configured yet. Add an API key to use style guide distillation.",
)

# Tag on every /memories/export file, checked (loosely -- see
# import_memories) on the way back in so a completely unrelated JSON file
# fails with a clear message instead of a confusing partial import.
MEMORIES_EXPORT_FORMAT = "mimoid-memories"
MEMORIES_EXPORT_VERSION = 1
INVALID_MEMORIES_EXPORT_DETAIL = (
    "That doesn't look like a Mimoid memories export -- expected a JSON file "
    'with a top-level "memories" list.'
)

# Tag on every /conversations/{id}/export file. No matching import route --
# this is a one-way transcript export, not a portable format meant to be
# read back in.
CONVERSATION_EXPORT_FORMAT = "mimoid-chat"
CONVERSATION_EXPORT_VERSION = 1


def _parse_iso_datetime(value: object) -> str | None:
    """Validates an imported entry's `created_at` without ever rejecting the
    import over it -- a missing or malformed timestamp (hand-edited file,
    export from a future format) just falls back to "now" for that one
    memory rather than failing the whole request."""
    if not isinstance(value, str):
        return None
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return None
    return value


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Kicks off loading the voice model into memory in the background, if
    # it's already downloaded -- every sidecar launch is a brand-new process
    # (see src-tauri/src/sidecar.rs), so the on-disk model cache survives a
    # restart but an in-memory one never does; without this, that ~1-minute
    # load cost lands on the user's first "hear in <name>'s voice" click
    # instead of happening quietly before they ever ask.
    #
    # Everything here -- including the is_model_downloaded() check itself --
    # must run inside the background task via run_in_threadpool, never
    # synchronously in this function. is_model_downloaded() imports coqui's
    # `trainer` package on first call, which alone can take upwards of a
    # minute; calling it directly here would block the ASGI startup phase
    # (and therefore /health and every other route) for that whole time --
    # exactly the regression this fixes. tts.warm_up() already no-ops
    # quickly on its own if the model isn't downloaded, so no outer check is
    # needed before scheduling it.
    async def _warm_up() -> None:
        with activity.track("Loading voice model…"):
            try:
                await run_in_threadpool(tts.warm_up)
            except Exception:
                logger.exception("TTS warm-up failed (a later /speak request will just retry)")

    asyncio.create_task(_warm_up())

    # One-time rebuild of the memories index into a cosine-space collection,
    # which is what makes the relevance floor in search_memories() possible
    # (see config.MEMORY_MAX_DISTANCE). Background and non-fatal for the same
    # reason as the warm-up above -- and additionally because embedding every
    # memory needs Ollama, which often isn't up yet at launch. Until this
    # succeeds the twin keeps using the legacy ungated index, i.e. the old
    # improvise-from-irrelevant-memories behavior persists; the log lines
    # below are deliberately loud enough to notice if it never runs.
    async def _reindex_memories() -> None:
        if not await run_in_threadpool(memory.needs_cosine_reindex):
            return
        stored = await run_in_threadpool(db.list_memories)
        logger.info("rebuilding the memories index for relevance filtering (%d memories)…", len(stored))
        try:
            with activity.track("Rebuilding memory index…"):
                indexed = await memory.reindex_memories_to_cosine(stored)
            logger.info("memory index rebuilt (%d memories indexed)", indexed)
        except Exception:
            logger.exception(
                "memory index rebuild failed -- retrying on the next launch. Until it "
                "succeeds, memory retrieval stays unfiltered. Is Ollama running with "
                "'%s' pulled?",
                config.EMBEDDING_MODEL,
            )

    asyncio.create_task(_reindex_memories())
    yield


app = FastAPI(title="Mimoid Sidecar", lifespan=lifespan)

# The app talks to this sidecar directly from the Tauri WebView over
# localhost HTTP; there's no other client, so a permissive CORS policy is
# fine for a machine-local, offline service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    # The frontend and this sidecar are different origins (different port in
    # dev, different scheme entirely under Tauri), so JS can only read the
    # CORS-safelisted response headers unless they're named here. Without
    # this, res.headers.get("X-Memory-Count") and friends come back null and
    # the features built on them fail silently -- there's no error, the
    # values just never arrive. Any new X- header the frontend reads has to
    # be added to this list.
    expose_headers=["X-Memory-Count", "X-History-Count", "X-Turn-Started"],
)

db.init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/activity")
def get_activity() -> dict[str, str | None]:
    return {"activity": activity.current()}


@app.get("/network-activity")
def get_network_activity() -> dict:
    return {"current": network_activity.current(), "history": network_activity.history()}


@app.get("/capabilities")
def get_capabilities() -> dict[str, bool]:
    """Whether on-device speech-to-text / text-to-speech can run here. Lets
    the UI gate voice-only features (e.g. hands-free conversation mode) up
    front rather than letting the user start something that 503s on the first
    turn. Import-only probes -- no model load/download."""
    return {"stt": stt.is_available(), "tts": tts.is_available()}


@app.get("/storage")
def get_storage() -> dict[str, int]:
    return {"chroma_bytes": memory.disk_usage_bytes()}


@app.get("/config")
def get_config() -> dict[str, str | bool]:
    return {
        "model": config.DEFAULT_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "whisper_model": stt.active_model(),
        "tts_model": config.TTS_MODEL,
        "openrouter_model": config.OPENROUTER_MODEL,
        "openrouter_configured": openrouter.get_status()["has_api_key"],
    }


def _current_language() -> str:
    return db.get_setting("language", config.DEFAULT_LANGUAGE)


def _current_pause_ms() -> int:
    try:
        return int(db.get_setting("pause_ms", str(DEFAULT_PAUSE_MS)))
    except ValueError:
        return DEFAULT_PAUSE_MS


def _current_owner_name() -> str:
    return db.get_setting("owner_name", config.DEFAULT_OWNER_NAME)


def _settings_payload() -> dict[str, str | int]:
    return {
        "language": _current_language(),
        "pause_ms": _current_pause_ms(),
        "owner_name": _current_owner_name(),
    }


@app.get("/settings")
def get_settings() -> dict[str, str | int]:
    return _settings_payload()


# Fields are optional so the language selector, the voice pause slider, and
# the owner-name field can each PUT only what they own without clobbering
# the others.
class SettingsRequest(BaseModel):
    language: str | None = None
    pause_ms: int | None = None
    owner_name: str | None = None


MAX_OWNER_NAME_LENGTH = 60


@app.put("/settings")
def update_settings(body: SettingsRequest) -> dict[str, str | int]:
    if body.language is not None:
        if body.language not in config.SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail=f"unsupported language: {body.language}")
        db.set_setting("language", body.language)
    if body.pause_ms is not None:
        if not MIN_PAUSE_MS <= body.pause_ms <= MAX_PAUSE_MS:
            raise HTTPException(
                status_code=400,
                detail=f"pause_ms must be between {MIN_PAUSE_MS} and {MAX_PAUSE_MS}",
            )
        db.set_setting("pause_ms", str(body.pause_ms))
    if body.owner_name is not None:
        stripped = body.owner_name.strip()
        if not stripped or len(stripped) > MAX_OWNER_NAME_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"owner_name must be 1-{MAX_OWNER_NAME_LENGTH} characters",
            )
        db.set_setting("owner_name", stripped)
    return _settings_payload()


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile) -> dict[str, str]:
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    label = "Transcribing…" if stt.is_model_downloaded() else "Downloading speech model… (first use, ~1-2 GB)"
    try:
        with activity.track(label):
            text = await asyncio.wait_for(
                run_in_threadpool(stt.transcribe, tmp_path, _current_language()),
                timeout=STT_TIMEOUT_SECONDS,
            )
    except stt.SpeechToTextUnavailable:
        raise STT_UNAVAILABLE from None
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                "Transcription timed out. If this is your first time using voice input, "
                "the speech model may still be downloading in the background on a slow "
                "connection -- check your network and try again shortly."
            ),
        ) from None
    finally:
        os.unlink(tmp_path)

    return {"text": text}


@app.get("/ollama/models")
async def ollama_models() -> dict[str, list[dict]]:
    try:
        return {"models": await ollama.list_models()}
    except httpx.HTTPError:
        raise OLLAMA_UNAVAILABLE from None


class PullRequest(BaseModel):
    name: str


@app.post("/ollama/pull")
async def ollama_pull(body: PullRequest) -> StreamingResponse:
    try:
        await ollama.list_models()
    except httpx.HTTPError:
        raise OLLAMA_UNAVAILABLE from None

    async def stream():
        with (
            activity.track("Downloading model…"),
            network_activity.track(f"Ollama registry: downloading {body.name}"),
        ):
            async for line in ollama.pull_model(body.name):
                yield line

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/conversations")
def create_conversation() -> dict:
    return db.create_conversation()


@app.get("/conversations")
def list_conversations() -> dict[str, list[dict]]:
    return {"conversations": db.list_conversations()}


@app.post("/conversations/reset")
def reset_conversation() -> dict:
    for conversation in db.list_conversations():
        db.archive_conversation(conversation["id"])
    return db.create_conversation()


@app.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: str) -> dict[str, list[dict]]:
    if not db.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"messages": db.get_messages(conversation_id)}


@app.get("/conversations/{conversation_id}/export")
def export_conversation(conversation_id: str) -> dict:
    """Downloadable JSON snapshot of one conversation's transcript -- both
    the owner's messages and the twin's replies, in order.

    Deliberately text messages only. Tool calls/results are never stored on
    a message row (the assistant's tool-call directives are ephemeral,
    living only in the in-memory list built for that one streaming request)
    -- the durable record is the separate tool_runs table, correlated to
    messages only by rough timestamp with no foreign key tying a run to a
    specific message. There's no clean way to interleave that into a
    transcript export, and nobody asked for it; this is a transcript of the
    conversation as the owner and the twin actually said it."""
    conversation = db.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    # Matches what the Talk view actually renders (chat.ts's history load
    # skips role == "system" the same way) -- "the conversation" means the
    # owner's and the twin's own turns, not internal bookkeeping rows.
    messages = [message for message in db.get_messages(conversation_id) if message["role"] != "system"]
    return {
        "format": CONVERSATION_EXPORT_FORMAT,
        "version": CONVERSATION_EXPORT_VERSION,
        "exported_at": datetime.now(UTC).isoformat(),
        "conversation": conversation,
        "messages": messages,
    }


class SendMessageRequest(BaseModel):
    content: str


# ~3.5 chars/token is a rough but workable estimate for English. Windowing
# history ourselves -- rather than sending the whole conversation and letting
# Ollama truncate once it overflows num_ctx -- matters because Ollama drops
# the *oldest* messages first, and message 0 is always the system prompt
# (persona identity, style guide, memories). An unbounded history therefore
# silently degrades the persona on any sufficiently long conversation; this
# keeps the system prompt intact and drops old turns instead.
CHARS_PER_TOKEN_ESTIMATE = 3.5
# Reserves headroom in num_ctx for the system prompt itself (already
# accounted for below) plus the model's reply.
HISTORY_CONTEXT_FRACTION = 0.6


def _windowed_history(history: list[dict], system_prompt: str) -> list[dict]:
    """Returns the newest-first-filled suffix of `history` that fits the
    character budget, oldest messages dropped first. Always keeps at least
    the most recent message, even if it alone exceeds the budget."""
    budget_chars = (
        config.OLLAMA_NUM_CTX * HISTORY_CONTEXT_FRACTION * CHARS_PER_TOKEN_ESTIMATE
        - len(system_prompt)
    )
    if budget_chars <= 0:
        return history[-1:]

    kept: list[dict] = []
    used_chars = 0
    for message in reversed(history):
        length = len(message["content"])
        if kept and used_chars + length > budget_chars:
            break
        kept.append(message)
        used_chars += length
    kept.reverse()
    return kept


@app.post("/conversations/{conversation_id}/messages")
async def send_message(conversation_id: str, body: SendMessageRequest) -> StreamingResponse:
    if not db.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")

    turn_started_at = datetime.now(UTC).isoformat()
    db.add_message(conversation_id, "user", body.content)
    history = db.get_messages(conversation_id)

    # Retrieval is best-effort: if it fails (e.g. the embedding model isn't
    # pulled, or a cold model swap makes the embed time out) the chat still
    # proceeds without memories rather than failing the whole request. But the
    # failure is LOGGED now -- a silently empty retrieval used to be
    # indistinguishable from "nothing relevant found", which made a
    # dropped-memories regression impossible to diagnose. The query is embedded
    # once and shared across all three collections; the embed is in its own
    # try so an embed failure is reported distinctly from a search failure.
    relevant_memories: list[dict] = []
    history_snippets: list[dict] = []
    style_examples: list[dict] = []
    try:
        query_embedding: list[float] | None = await ollama.embed(body.content)
    except httpx.HTTPError as err:
        query_embedding = None
        logger.warning(
            "retrieval skipped -- embedding the query failed (%s); replying "
            "without memories/history/style this turn. Is '%s' pulled in Ollama?",
            err,
            config.EMBEDDING_MODEL,
        )
    if query_embedding is not None:
        # The three searches share one embedding and don't depend on each
        # other, so they run concurrently -- in sequence they added three
        # round trips to the front of every single turn. return_exceptions
        # keeps the original best-effort semantics per collection: one
        # collection failing still leaves the other two's results usable,
        # where a bare gather would discard all three.
        results = await asyncio.gather(
            memory.search_memories(body.content, query_embedding=query_embedding),
            memory.search_vault(body.content, query_embedding=query_embedding),
            memory.search_style_examples(body.content, query_embedding=query_embedding),
            return_exceptions=True,
        )
        searched: list[list[dict]] = []
        for label, result in zip(("memories", "vault", "style"), results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("retrieval search failed for %s (%s); continuing without it", label, result)
                searched.append([])
            else:
                searched.append(result)
        relevant_memories, history_snippets, style_examples = searched
        logger.info(
            "retrieval for chat: %d memories, %d history snippets, %d style examples",
            len(relevant_memories),
            len(history_snippets),
            len(style_examples),
        )

    system_prompt = await persona.build_system_prompt(
        relevant_memories,
        history_snippets,
        _current_language(),
        style_examples,
        owner_name=_current_owner_name(),
        # Distinct from "retrieval returned nothing": there, the twin genuinely
        # has no relevant memories and should say so; here its recall is simply
        # unreachable, and claiming "I have no memory of that" would be a
        # confident false statement about its own past.
        memory_lookup_failed=query_embedding is None,
    )
    windowed_history = _windowed_history(history, system_prompt)
    chat_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": message["role"], "content": message["content"]} for message in windowed_history
    ]

    tool_specs = tools.available_specs() if config.model_supports_tools(config.DEFAULT_MODEL) else []

    async def stream():
        reply_tokens: list[str] = []
        with activity.track("Thinking…"):
            try:
                # One iteration per tool round. Each round appends the
                # model's tool-call message and the results to chat_messages
                # and re-enters the stream, so the next request's prompt is a
                # strict prefix-extension of the last -- Ollama's KV cache
                # then covers everything but the appended tokens, making the
                # follow-up pass mostly decode rather than a full re-prefill.
                # Rewriting or reordering earlier messages here would quietly
                # forfeit that; test_tools.py asserts the prefix property.
                for _ in range(config.TOOL_MAX_ITERATIONS):
                    pending: list[dict] = []
                    async for event in ollama.chat_stream(chat_messages, tools=tool_specs):
                        if event["type"] == "token":
                            reply_tokens.append(event["text"])
                            yield event["text"]
                        elif event["type"] == "tool_calls":
                            pending = event["calls"]
                    if not pending:
                        break
                    calls = tools.normalize_calls(pending)
                    if not calls:
                        break
                    label = "Checking " + ", ".join(name.replace("_", " ") for name, _ in calls)
                    with activity.track(f"{label}…"):
                        results = await tools.dispatch_all(calls, context={"conversation_id": conversation_id})
                    chat_messages.append({"role": "assistant", "content": "", "tool_calls": pending})
                    for (name, arguments), result in zip(calls, results, strict=True):
                        chat_messages.append({"role": "tool", "content": result})
                        db.create_tool_run(conversation_id, name, arguments, result)
            except httpx.HTTPError:
                yield "\n\n[Error: could not reach Ollama. Is it running?]"
                return
            db.add_message(conversation_id, "assistant", "".join(reply_tokens))

    return StreamingResponse(
        stream(),
        media_type="text/plain",
        headers={
            "X-Memory-Count": str(len(relevant_memories)),
            "X-History-Count": str(len(history_snippets)),
            # Echoed back to /tool-runs?after=... so the Talk tab sees only
            # this turn's tool runs. Issued here rather than taken from the
            # browser because the two produce different ISO spellings for the
            # same instant ("+00:00" vs "Z", microseconds vs milliseconds),
            # and the query compares these as strings -- a client-generated
            # value would sort wrong against stored rows and silently drop
            # them.
            "X-Turn-Started": turn_started_at,
        },
    )


@app.get("/conversations/{conversation_id}/tool-runs")
def get_tool_runs(conversation_id: str, after: str | None = None) -> dict:
    if not db.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"tool_runs": db.list_tool_runs(conversation_id, after)}


@app.get("/conversations/{conversation_id}/calendar/pending")
def get_pending_calendar_events(conversation_id: str, after: str | None = None) -> dict:
    if not db.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"pending_events": db.list_pending_calendar_events(conversation_id, after)}


@app.post("/calendar/pending/{event_id}/confirm")
async def confirm_pending_calendar_event(event_id: str) -> dict:
    pending = db.get_pending_calendar_event(event_id)
    if pending is None:
        raise HTTPException(status_code=404, detail="that draft event is gone")
    try:
        # create_event already wraps itself in network_activity.track --
        # doing it again here would double-count this call in the footer's
        # internet-access list.
        event = await google_calendar.create_event(
            pending["title"], pending["start_at"], pending["duration_minutes"], pending["description"]
        )
    except google_calendar.GoogleCalendarUnavailable as err:
        raise HTTPException(status_code=400, detail=str(err)) from None
    except (httpx.HTTPError, ValueError) as err:
        raise HTTPException(status_code=502, detail=f"Google Calendar request failed: {err}") from None
    db.delete_pending_calendar_event(event_id)
    return {"html_link": event.get("htmlLink", "")}


@app.delete("/calendar/pending/{event_id}")
def discard_pending_calendar_event(event_id: str) -> dict:
    db.delete_pending_calendar_event(event_id)
    return {"discarded": True}


class GoogleCredentialsRequest(BaseModel):
    client_id: str
    client_secret: str


@app.get("/tools/google/status")
def get_google_status() -> dict:
    return google_calendar.get_status()


@app.put("/tools/google/credentials")
def update_google_credentials(body: GoogleCredentialsRequest) -> dict:
    if not body.client_id.strip() or not body.client_secret.strip():
        raise HTTPException(status_code=400, detail="Both a client ID and client secret are needed.")
    google_calendar.set_client_credentials(body.client_id, body.client_secret)
    return google_calendar.get_status()


@app.post("/tools/google/connect")
async def connect_google() -> dict:
    try:
        auth_url = await google_calendar.start_connect()
    except google_calendar.GoogleCalendarUnavailable as err:
        raise HTTPException(status_code=400, detail=str(err)) from None
    return {"auth_url": auth_url}


@app.post("/tools/google/disconnect")
def disconnect_google() -> dict:
    google_calendar.disconnect()
    return google_calendar.get_status()


class WeatherLocation(BaseModel):
    name: str
    latitude: float
    longitude: float


class ToolSettingsRequest(BaseModel):
    clock_enabled: bool | None = None
    timezone: str | None = None
    weather_enabled: bool | None = None
    weather_locations: list[WeatherLocation] | None = None
    calendar_enabled: bool | None = None
    calendar_confirm: bool | None = None
    calendar_write_target: str | None = None


def _tools_payload() -> dict:
    google_status = google_calendar.get_status()
    return {
        "model": config.DEFAULT_MODEL,
        "model_supports_tools": config.model_supports_tools(config.DEFAULT_MODEL),
        "clock_enabled": tools.feature_enabled(tools.FEATURE_CLOCK),
        "timezone": db.get_setting("tools_timezone", ""),
        "weather_enabled": tools.feature_enabled(tools.FEATURE_WEATHER),
        "weather_locations": weather.configured_locations(),
        "calendar_enabled": tools.feature_enabled(tools.FEATURE_CALENDAR),
        "calendar_confirm": db.get_setting("tools_calendar_confirm", "1") == "1",
        "calendar_configured": google_status["configured"],
        "calendar_connected": google_status["connected"],
        # The one calendar create_calendar_event is ever allowed to write
        # to -- reading is never restricted to a subset, only writing is.
        "calendar_write_target": google_calendar.write_calendar_name(),
    }


@app.get("/tools")
def get_tools() -> dict:
    return _tools_payload()


@app.put("/tools")
def update_tools(body: ToolSettingsRequest) -> dict:
    if body.clock_enabled is not None:
        tools.set_feature_enabled(tools.FEATURE_CLOCK, body.clock_enabled)
    if body.timezone is not None:
        db.set_setting("tools_timezone", body.timezone.strip())
    if body.weather_enabled is not None:
        tools.set_feature_enabled(tools.FEATURE_WEATHER, body.weather_enabled)
    if body.weather_locations is not None:
        # At least one location must survive -- an empty list would leave
        # get_weather with nothing to prefer and every lookup falling back
        # to geocoding, silently changing behaviour rather than rejecting
        # the edit that caused it.
        if not body.weather_locations:
            raise HTTPException(status_code=400, detail="at least one weather location is required")
        db.set_setting(
            "tools_weather_locations",
            json.dumps([location.model_dump() for location in body.weather_locations]),
        )
    if body.calendar_enabled is not None:
        tools.set_feature_enabled(tools.FEATURE_CALENDAR, body.calendar_enabled)
    if body.calendar_confirm is not None:
        db.set_setting("tools_calendar_confirm", "1" if body.calendar_confirm else "0")
    if body.calendar_write_target is not None:
        # The one rule create_calendar_event has to keep -- never write
        # anywhere but this calendar -- so an empty name (which would leave
        # google_calendar.write_calendar_name() falling back to the
        # "Tasks" default silently) is rejected rather than accepted as if
        # it meant something.
        if not body.calendar_write_target.strip():
            raise HTTPException(status_code=400, detail="the write-target calendar name can't be empty")
        google_calendar.set_write_calendar_name(body.calendar_write_target)
    return _tools_payload()


class MemoryRequest(BaseModel):
    content: str
    topic: str | None = None
    occurred_at: str | None = None


@app.post("/memories")
async def create_memory(body: MemoryRequest) -> dict:
    metadata = {"topic": body.topic or "", "occurred_at": body.occurred_at or ""}
    record = db.create_memory(body.content, body.topic, occurred_at=body.occurred_at)
    try:
        with activity.track("Saving memory…"):
            await memory.index_memory(record["id"], body.content, metadata)
    except httpx.HTTPError:
        db.delete_memory(record["id"])
        raise OLLAMA_UNAVAILABLE from None
    return record


@app.get("/memories")
def list_memories(q: str | None = None) -> dict[str, list[dict]]:
    return {"memories": db.list_memories(q)}


# Registered ahead of GET /memories/{memory_id} -- Starlette matches routes
# in registration order, so "export" would otherwise be swallowed by the
# dynamic {memory_id} route below and 404 as "memory not found".
@app.get("/memories/export")
def export_memories() -> dict:
    """Downloadable JSON snapshot of every memory -- portable between
    machines and installs, and the recovery path if a local database ever
    looks empty for a reason other than actually being empty (e.g. two
    separately installed sidecars racing over the same port; see
    /memories/import). `id` and `updated_at` are deliberately left out:
    they're this database's own bookkeeping, not something a re-import
    should try to preserve."""
    memories = [
        {
            "content": row["content"],
            "topic": row["topic"],
            "occurred_at": row["occurred_at"],
            "created_at": row["created_at"],
        }
        for row in db.list_memories()
    ]
    return {
        "format": MEMORIES_EXPORT_FORMAT,
        "version": MEMORIES_EXPORT_VERSION,
        "exported_at": datetime.now(UTC).isoformat(),
        "memories": memories,
    }


@app.post("/memories/import")
async def import_memories(file: UploadFile) -> dict:
    """Imports memories from a file /memories/export produced (or anything
    matching the same shape). Best-effort and additive: invalid entries are
    skipped and counted rather than failing the whole request, and content
    that normalizes (stripped, lowercased) to an existing memory is skipped
    as a duplicate rather than creating a second copy -- importing the same
    export twice, or an export that overlaps what's already here, is a
    normal thing to do, not an error."""
    data = await file.read()
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail=INVALID_MEMORIES_EXPORT_DETAIL) from None

    if not isinstance(payload, dict) or not isinstance(payload.get("memories"), list):
        raise HTTPException(status_code=400, detail=INVALID_MEMORIES_EXPORT_DETAIL)
    declared_format = payload.get("format")
    if declared_format is not None and declared_format != MEMORIES_EXPORT_FORMAT:
        raise HTTPException(status_code=400, detail=INVALID_MEMORIES_EXPORT_DETAIL)

    existing_contents = db.list_memory_contents()
    created = 0
    skipped_duplicates = 0
    skipped_invalid = 0

    with activity.track("Importing memories…"):
        for entry in payload["memories"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("content"), str):
                skipped_invalid += 1
                continue
            content = entry["content"].strip()
            if not content:
                skipped_invalid += 1
                continue

            if content.lower() in existing_contents:
                skipped_duplicates += 1
                continue
            existing_contents.add(content.lower())

            topic = entry.get("topic") if isinstance(entry.get("topic"), str) else None
            occurred_at = entry.get("occurred_at") if isinstance(entry.get("occurred_at"), str) else None
            created_at = _parse_iso_datetime(entry.get("created_at"))

            record = db.create_memory(content, topic, source="import", occurred_at=occurred_at, created_at=created_at)
            try:
                metadata = {"topic": topic or "", "occurred_at": occurred_at or ""}
                await memory.index_memory(record["id"], content, metadata)
            except httpx.HTTPError:
                logger.warning("failed to index imported memory %s for retrieval", record["id"])
            created += 1

    return {"created": created, "skipped_duplicates": skipped_duplicates, "skipped_invalid": skipped_invalid}


@app.get("/memories/{memory_id}")
def get_memory(memory_id: str) -> dict:
    record = db.get_memory(memory_id)
    if record is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return record


@app.put("/memories/{memory_id}")
async def update_memory(memory_id: str, body: MemoryRequest) -> dict:
    if db.get_memory(memory_id) is None:
        raise HTTPException(status_code=404, detail="memory not found")

    metadata = {"topic": body.topic or "", "occurred_at": body.occurred_at or ""}
    try:
        with activity.track("Saving memory…"):
            await memory.index_memory(memory_id, body.content, metadata)
    except httpx.HTTPError:
        raise OLLAMA_UNAVAILABLE from None

    record = db.update_memory(memory_id, body.content, body.topic, body.occurred_at)
    assert record is not None
    return record


@app.delete("/memories/{memory_id}")
def delete_memory(memory_id: str) -> dict[str, bool]:
    if db.get_memory(memory_id) is None:
        raise HTTPException(status_code=404, detail="memory not found")
    memory.deindex_memory(memory_id)
    db.delete_memory(memory_id)
    return {"deleted": True}


@app.post("/vault/documents")
async def create_vault_document(file: UploadFile) -> dict:
    data = await file.read()
    if len(data) > config.VAULT_MAX_UPLOAD_BYTES:
        limit_mb = config.VAULT_MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"file is too large -- the limit is {limit_mb} MB per document",
        )

    try:
        title, text = vault.extract_text(file.filename or "document", data)
    except vault.VaultExtractionError as err:
        raise HTTPException(status_code=400, detail=str(err)) from None

    record = db.create_vault_document(title, file.filename or "document", text)
    try:
        with activity.track("Indexing document…"):
            chunk_count = await memory.index_vault_document(record["id"], title, text)
    except httpx.HTTPError:
        db.delete_vault_document(record["id"])
        raise OLLAMA_UNAVAILABLE from None

    db.set_vault_document_chunk_count(record["id"], chunk_count)
    record["chunk_count"] = chunk_count
    return record


@app.get("/vault/documents")
def list_vault_documents() -> dict[str, list[dict]]:
    return {"documents": db.list_vault_documents()}


@app.delete("/vault/documents/{doc_id}")
def delete_vault_document(doc_id: str) -> dict[str, bool]:
    if db.get_vault_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="document not found")
    memory.deindex_vault_document(doc_id)
    db.delete_vault_document(doc_id)
    return {"deleted": True}


@app.post("/voice-samples")
async def create_voice_sample(file: UploadFile, prompt: str = Form(...)) -> dict:
    temp_name = uuid.uuid4().hex
    raw_suffix = Path(file.filename or "sample.webm").suffix or ".webm"
    raw_path = config.VOICE_SAMPLES_DIR / f"{temp_name}-raw{raw_suffix}"
    wav_path = config.VOICE_SAMPLES_DIR / f"{temp_name}.wav"

    raw_path.write_bytes(await file.read())
    try:
        audio.convert_to_wav(str(raw_path), str(wav_path))
    except audio.AudioConversionError as err:
        raise HTTPException(
            status_code=500, detail=f"couldn't process the recording: {err}"
        ) from None
    finally:
        raw_path.unlink(missing_ok=True)

    record = db.create_voice_sample(prompt, str(wav_path))
    return {"id": record["id"], "prompt": record["prompt"], "created_at": record["created_at"]}


@app.get("/voice-samples")
def list_voice_samples() -> dict[str, list[dict]]:
    samples = db.list_voice_samples()
    return {
        "samples": [
            {"id": s["id"], "prompt": s["prompt"], "created_at": s["created_at"]}
            for s in samples
        ]
    }


@app.get("/voice-samples/{sample_id}/audio")
def get_voice_sample_audio(sample_id: str) -> Response:
    record = db.get_voice_sample(sample_id)
    if record is None:
        raise HTTPException(status_code=404, detail="voice sample not found")
    return Response(content=Path(record["file_path"]).read_bytes(), media_type="audio/wav")


@app.delete("/voice-samples/{sample_id}")
def delete_voice_sample(sample_id: str) -> dict[str, bool]:
    record = db.get_voice_sample(sample_id)
    if record is None:
        raise HTTPException(status_code=404, detail="voice sample not found")
    Path(record["file_path"]).unlink(missing_ok=True)
    db.delete_voice_sample(sample_id)
    return {"deleted": True}


class SpeakRequest(BaseModel):
    text: str


@app.post("/speak")
async def speak(body: SpeakRequest) -> Response:
    samples = db.list_voice_samples()
    if not samples:
        raise HTTPException(
            status_code=400,
            detail="No voice samples recorded yet. Record some in Train mode first.",
        )
    speaker_wavs = [sample["file_path"] for sample in samples]

    label = "Speaking…" if tts.is_model_downloaded() else "Downloading voice model… (first use, ~1.8 GB)"
    logger.info("Speak request: %d chars, %d reference samples, label=%r", len(body.text), len(speaker_wavs), label)
    try:
        with activity.track(label):
            wav_bytes = await asyncio.wait_for(
                run_in_threadpool(tts.synthesize, body.text, speaker_wavs, _current_language()),
                timeout=TTS_TIMEOUT_SECONDS,
            )
    except tts.TextToSpeechUnavailable:
        logger.warning("TTS unavailable on this machine")
        raise TTS_UNAVAILABLE from None
    except asyncio.TimeoutError:
        logger.error("Speak request timed out after %ds", TTS_TIMEOUT_SECONDS)
        raise HTTPException(
            status_code=504,
            detail=(
                "Speech synthesis timed out. If this is your first time hearing a reply, the voice "
                "model may still be downloading in the background on a slow connection -- check your "
                "network and try again shortly."
            ),
        ) from None
    return Response(content=wav_bytes, media_type="audio/wav")


class StyleEntryRequest(BaseModel):
    kind: str
    content: str
    prompt: str | None = None
    source: str = "manual"


def _validate_style_kind(kind: str) -> None:
    if kind not in ("text", "qa"):
        raise HTTPException(status_code=400, detail="kind must be 'text' or 'qa'")


# Sources a client is allowed to write via POST /style-entries. Constrained
# (rather than accepting any string) because DELETE /style-entries?source=
# groups entries by this same value -- an arbitrary client-supplied source
# would break that grouping.
#
# INVARIANT: every row in style_entries must be human-authored text -- typed
# by Alex directly ("manual"), or parsed from his own WhatsApp/Signal chat
# exports (source "whatsapp"/"signal", written by those importers, not this
# route). style_entries feeds few-shot retrieval and every style-guide
# distillation, so anything model-generated that lands here starts a
# self-training loop: the twin's own quirks get fed back as "Alex's voice"
# and amplified at the next distill. "persona-check" used to be allowed here
# (a promotion flow that saved the winning DUEL ANSWER, i.e. model output,
# after an editable-but-anchored review step) -- that was a mitigation, not
# a fix, and is now removed. The Persona check tab's composer instead
# captures what Alex writes *before* he sees either generated answer, saved
# under "own-answer" -- genuinely his own words, not edited model output.
# Any future source added here must be justified against this invariant.
_ALLOWED_STYLE_ENTRY_SOURCES = {"manual", "own-answer"}


@app.post("/style-entries")
async def create_style_entry(body: StyleEntryRequest) -> dict:
    _validate_style_kind(body.kind)
    if body.source not in _ALLOWED_STYLE_ENTRY_SOURCES:
        raise HTTPException(
            status_code=400, detail=f"source must be one of {sorted(_ALLOWED_STYLE_ENTRY_SOURCES)}"
        )
    record = db.create_style_entry(body.kind, body.content, body.prompt, source=body.source)
    try:
        await memory.index_style_entry(record["id"], record["content"], record["kind"], record["prompt"])
    except httpx.HTTPError:
        db.delete_style_entry(record["id"])
        raise OLLAMA_UNAVAILABLE from None
    return record


@app.get("/style-entries")
def list_style_entries() -> dict[str, list[dict]]:
    return {"entries": db.list_style_entries()}


@app.put("/style-entries/{entry_id}")
async def update_style_entry(entry_id: str, body: StyleEntryRequest) -> dict:
    _validate_style_kind(body.kind)
    if db.get_style_entry(entry_id) is None:
        raise HTTPException(status_code=404, detail="style entry not found")
    try:
        await memory.index_style_entry(entry_id, body.content, body.kind, body.prompt)
    except httpx.HTTPError:
        raise OLLAMA_UNAVAILABLE from None
    record = db.update_style_entry(entry_id, body.kind, body.content, body.prompt)
    assert record is not None
    return record


@app.delete("/style-entries")
def delete_style_entries(source: str) -> dict[str, int]:
    ids = [entry["id"] for entry in db.list_style_entries() if entry["source"] == source]
    deleted = db.delete_style_entries_by_source(source)
    db.delete_import_batches_by_source(source)
    for entry_id in ids:
        memory.deindex_style_entry(entry_id)
    return {"deleted": deleted}


@app.delete("/style-entries/{entry_id}")
def delete_style_entry(entry_id: str) -> dict[str, bool]:
    if db.get_style_entry(entry_id) is None:
        raise HTTPException(status_code=404, detail="style entry not found")
    memory.deindex_style_entry(entry_id)
    db.delete_style_entry(entry_id)
    return {"deleted": True}


@app.get("/style-guide")
def get_style_guide() -> dict:
    record = db.get_style_guide()
    if record is None:
        raise HTTPException(status_code=404, detail="no style guide yet")
    return record


@app.post("/style-guide/distill")
async def distill_style_guide() -> dict:
    entries = db.list_style_entries()
    if not entries:
        raise HTTPException(
            status_code=400, detail="Add some style examples first."
        )
    try:
        with (
            activity.track("Distilling style…"),
            network_activity.track("OpenRouter: distilling style guide"),
        ):
            guide_text = await persona.distill_style_guide()
    except openrouter.OpenRouterUnavailable:
        raise OPENROUTER_UNAVAILABLE from None
    except httpx.HTTPError as err:
        raise HTTPException(
            status_code=502, detail=f"OpenRouter request failed: {err}"
        ) from None
    return db.set_style_guide(guide_text, entry_count=len(entries))


@app.post("/style-guide/distill/local")
async def distill_style_guide_locally() -> dict:
    entries = db.list_style_entries()
    if not entries:
        raise HTTPException(
            status_code=400, detail="Add some style examples first."
        )
    try:
        with activity.track("Distilling style…"):
            guide_text = await persona.distill_style_guide_locally()
    except httpx.HTTPError:
        raise OLLAMA_UNAVAILABLE from None
    return db.set_style_guide(guide_text, entry_count=len(entries))


@app.get("/openrouter/config")
def get_openrouter_config() -> dict:
    return openrouter.get_status()


class OpenRouterConfigRequest(BaseModel):
    api_key: str


@app.put("/openrouter/config")
def update_openrouter_config(body: OpenRouterConfigRequest) -> dict:
    if not body.api_key.strip():
        raise HTTPException(status_code=400, detail="API key can't be empty.")
    return openrouter.set_api_key(body.api_key)


@app.delete("/style-guide")
def delete_style_guide() -> dict[str, bool]:
    db.delete_style_guide()
    return {"deleted": True}


@app.get("/import-history")
def get_import_history() -> dict[str, list[dict]]:
    return {"batches": db.list_import_batches()}


class CompareRequest(BaseModel):
    prompt: str


PERSONA_DUEL_UNAVAILABLE = HTTPException(
    status_code=400,
    detail=(
        "Add some style examples first -- a comparison needs two different "
        "setups to pit against each other."
    ),
)


@app.post("/persona/duels")
async def create_persona_duel(body: CompareRequest) -> dict:
    try:
        with activity.track("Comparing answers…"):
            result = await persona.compare_arms(body.prompt, _current_language(), _current_owner_name())
    except persona.DuelUnavailable:
        raise PERSONA_DUEL_UNAVAILABLE from None
    except httpx.HTTPError:
        raise OLLAMA_UNAVAILABLE from None

    record = db.create_persona_duel(
        prompt=body.prompt,
        language=_current_language(),
        arm_a=result["arm_a"],
        arm_b=result["arm_b"],
        model_a=config.DEFAULT_MODEL,
        model_b=config.DEFAULT_MODEL,
        answer_a=result["answer_a"],
        answer_b=result["answer_b"],
    )
    # Arm identities are deliberately withheld until a choice is recorded --
    # returning them now would stop this from being a blind comparison.
    return {"id": record["id"], "a": result["answer_a"], "b": result["answer_b"]}


class DuelChoiceRequest(BaseModel):
    choice: str


@app.post("/persona/duels/{duel_id}/choice")
def choose_persona_duel(duel_id: str, body: DuelChoiceRequest) -> dict:
    record = db.get_persona_duel(duel_id)
    if record is None:
        raise HTTPException(status_code=404, detail="comparison not found")
    if record["choice"] is not None:
        raise HTTPException(status_code=409, detail="this comparison was already decided")
    if body.choice not in ("a", "b", "tie"):
        raise HTTPException(status_code=400, detail="choice must be 'a', 'b', or 'tie'")

    db.record_persona_duel_choice(duel_id, body.choice)
    summary = db.persona_duel_summary(persona.DEFAULT_ARM, persona.ARM_LABELS)
    return {**summary, "arm_a": record["arm_a"], "arm_b": record["arm_b"]}


@app.get("/persona/duel-summary")
def persona_duel_summary() -> dict:
    return db.persona_duel_summary(persona.DEFAULT_ARM, persona.ARM_LABELS)


@app.get("/backup/config")
def get_backup_config() -> dict:
    return backup.get_backup_config()


class BackupConfigRequest(BaseModel):
    url: str
    username: str
    password: str | None = None


@app.put("/backup/config")
def update_backup_config(body: BackupConfigRequest) -> dict:
    try:
        return backup.set_backup_config(body.url, body.username, body.password)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from None


@app.post("/backup/run")
async def run_backup() -> dict:
    cfg = backup.get_backup_config()
    stored_password = db.get_setting("backup_nextcloud_password", "")
    if not cfg["url"] or not cfg["username"] or not stored_password:
        raise HTTPException(
            status_code=400,
            detail="Backup isn't configured yet. Enter your Nextcloud details first.",
        )

    archive_name = f"mimoid-backup-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.zip"
    tmp_dir = Path(tempfile.mkdtemp())
    archive_path = tmp_dir / archive_name

    try:
        with activity.track("Backing up…"):
            size_bytes = await run_in_threadpool(backup.create_backup_archive, archive_path)
            try:
                with network_activity.track("Nextcloud backup: uploading archive"):
                    uploaded_to = await backup.upload_backup(
                        archive_path, cfg["url"], cfg["username"], stored_password
                    )
            except backup.NextcloudError as err:
                raise HTTPException(status_code=502, detail=str(err)) from None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    last_backup_at = datetime.now(UTC).isoformat()
    db.set_setting("backup_last_at", last_backup_at)
    return {"uploaded_to": uploaded_to, "size_bytes": size_bytes, "last_backup_at": last_backup_at}


@app.post("/backup/download")
async def download_backup() -> Response:
    """Builds the same archive create_backup_archive()/run_backup() use, but
    hands it straight back to the caller as a file download instead of
    uploading it -- a fallback for anyone without (or not wanting to use)
    Nextcloud. Works standalone: no backup config required. Deliberately
    does NOT touch backup_last_at -- that setting means "durably backed up
    to Nextcloud", a guarantee a local download doesn't carry."""
    archive_name = f"mimoid-backup-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.zip"
    tmp_dir = Path(tempfile.mkdtemp())
    archive_path = tmp_dir / archive_name

    try:
        with activity.track("Backing up…"):
            await run_in_threadpool(backup.create_backup_archive, archive_path)
            archive_bytes = archive_path.read_bytes()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return Response(
        content=archive_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{archive_name}"'},
    )


# Parsed WhatsApp uploads, held between the parse and import steps. This is a
# single local process, so an in-memory dict is enough; trimmed to the few
# most recent so it can't grow unbounded, and lost on restart (the user just
# re-uploads -- parsing is cheap). The filename is captured at parse time
# (when the server has it from the upload) rather than round-tripped through
# the client on import, which would mean trusting the client for a value the
# server already owns.
_whatsapp_uploads: dict[str, dict] = {}
_WHATSAPP_UPLOAD_LIMIT = 3


@app.post("/whatsapp/parse")
async def parse_whatsapp(file: UploadFile) -> dict:
    data = await file.read()
    try:
        text = whatsapp.extract_chat_text(file.filename or "chat.txt", data)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from None

    messages = whatsapp.parse_export(text)
    if not messages:
        raise HTTPException(
            status_code=400,
            detail="That doesn't look like a WhatsApp chat export (.txt or .zip from 'Export chat').",
        )

    upload_id = uuid.uuid4().hex
    _whatsapp_uploads[upload_id] = {
        "filename": file.filename or "chat.txt",
        "messages": messages,
    }
    # Keep only the most recent uploads.
    for stale in list(_whatsapp_uploads)[:-_WHATSAPP_UPLOAD_LIMIT]:
        _whatsapp_uploads.pop(stale, None)

    return {
        "upload_id": upload_id,
        "participants": whatsapp.participants(messages),
        "total_messages": len(messages),
    }


class WhatsappImportRequest(BaseModel):
    upload_id: str
    me: str


async def _index_style_entry_best_effort(record: dict) -> None:
    """Used by the bulk chat importers: indexing failure (e.g. Ollama not
    reachable) shouldn't fail or roll back the whole import -- the entry
    stays in SQLite either way and is still usable for distillation and the
    raw-corpus fallback, just not for few-shot retrieval until re-indexed."""
    try:
        await memory.index_style_entry(record["id"], record["content"], record["kind"], record["prompt"])
    except httpx.HTTPError:
        logger.warning("failed to index imported style entry %s for few-shot retrieval", record["id"])


@app.post("/whatsapp/import")
async def import_whatsapp(body: WhatsappImportRequest) -> dict:
    upload = _whatsapp_uploads.get(body.upload_id)
    if upload is None:
        raise HTTPException(status_code=404, detail="Upload expired -- please choose the file again.")
    messages = upload["messages"]

    names = {p["name"] for p in whatsapp.participants(messages)}
    if body.me not in names:
        raise HTTPException(status_code=400, detail=f"'{body.me}' isn't a participant in this chat.")

    existing = db.list_style_entry_contents_by_source("whatsapp")
    with activity.track("Importing chat…"):
        entries = whatsapp.select_entries(messages, body.me, existing_contents=existing)
        for entry in entries:
            record = db.create_style_entry(entry["kind"], entry["content"], entry["prompt"], source="whatsapp")
            await _index_style_entry_best_effort(record)

    _whatsapp_uploads.pop(body.upload_id, None)

    if entries:
        db.create_import_batch("whatsapp", upload["filename"], len(entries))

    qa_count = sum(1 for e in entries if e["kind"] == "qa")
    return {
        "created": len(entries),
        "qa_count": qa_count,
        "text_count": len(entries) - qa_count,
    }


@app.post("/signal/import")
async def import_signal(files: list[UploadFile]) -> dict:
    """Signal Desktop exports one .txt per conversation, so this accepts
    multiple files at once. Unlike WhatsApp's export, Signal already labels
    which side is the owner ("From: You"), so there's no separate
    parse-then-choose-participant step -- this parses and imports in one
    request."""
    per_file_messages: list[list[dict]] = []
    conversations: list[dict] = []

    for file in files:
        data = await file.read()
        text = data.decode("utf-8-sig", errors="replace")
        messages = signal.parse_export(text)
        if not messages:
            continue
        per_file_messages.append(messages)
        name = signal.conversation_name(text) or (file.filename or "Unknown conversation")
        conversations.append({"name": name, "message_count": len(messages)})

    if not per_file_messages:
        raise HTTPException(
            status_code=400,
            detail="That doesn't look like a Signal chat export (.txt from Signal Desktop's chat export).",
        )

    existing = db.list_style_entry_contents_by_source("signal")
    with activity.track("Importing chat…"):
        entries = signal.select_entries_for_files(per_file_messages, existing_contents=existing)
        for entry in entries:
            record = db.create_style_entry(entry["kind"], entry["content"], entry["prompt"], source="signal")
            await _index_style_entry_best_effort(record)

    if entries:
        # Per-file entry counts aren't computable -- dedup runs across all
        # files together -- so this records one batch row for the whole
        # request, named after the first file (+N more when there are others).
        filenames = [file.filename or "chat.txt" for file in files]
        batch_filename = filenames[0]
        if len(filenames) > 1:
            batch_filename += f" +{len(filenames) - 1} more"
        db.create_import_batch("signal", batch_filename, len(entries))

    qa_count = sum(1 for e in entries if e["kind"] == "qa")
    return {
        "created": len(entries),
        "qa_count": qa_count,
        "text_count": len(entries) - qa_count,
        "conversations": conversations,
    }
