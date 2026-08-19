import logging
import os
import shutil
from pathlib import Path

# No handler/level is configured by default (uvicorn only configures its own
# loggers), so without this, logger.info() calls anywhere in the app are
# silently dropped -- makes the sidecar's console output useless for
# diagnosing a stuck request (e.g. a stalled first-time model download).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("MIMOID_MODEL", "qwen2.5")
EMBEDDING_MODEL = os.environ.get("MIMOID_EMBEDDING_MODEL", "nomic-embed-text")
# Speech-to-text has two backends (see stt.py) and they name models
# differently, so each gets its own setting. WHISPER_MODEL is the Hugging
# Face repo id used by mlx-whisper on Apple Silicon; WHISPER_TORCH_MODEL is
# the short name from `whisper.available_models()` used everywhere else.
# Both default to the same "turbo" checkpoint, so quality matches across
# platforms -- ask stt.active_model() for whichever one is actually live.
WHISPER_MODEL = os.environ.get("MIMOID_WHISPER_MODEL", "mlx-community/whisper-turbo")
WHISPER_TORCH_MODEL = os.environ.get("MIMOID_WHISPER_TORCH_MODEL", "turbo")
TTS_MODEL = os.environ.get("MIMOID_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
OPENROUTER_API_KEY = os.environ.get("MIMOID_OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("MIMOID_OPENROUTER_MODEL", "anthropic/claude-3.5-haiku")
# Fallback defaults for advanced/dev setups; the Tools tab's saved
# credentials take precedence, same precedence order as the OpenRouter key.
GOOGLE_CLIENT_ID = os.environ.get("MIMOID_GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("MIMOID_GOOGLE_CLIENT_SECRET")

# Historically -- and still today in dev, on purpose -- this defaulted to a
# `data/` folder next to the sidecar source (sidecar/data/ locally, handy
# to poke at). In a *packaged* app that same relative path resolves inside
# the app bundle itself, which src-tauri/src/sidecar.rs's source_dir()
# calls "read-only" in its own comment: every local memory a user has saved
# was living somewhere a plain reinstall or update -- replacing the bundle
# with a fresh build -- would silently wipe. sidecar.rs now points
# MIMOID_DATA_DIR at the app's real writable data directory instead in
# packaged builds; _LEGACY_DATA_DIR is what it used to resolve to, kept
# here only so migrate_legacy_data_dir() below can find and move across
# whatever an earlier version already wrote there.
_LEGACY_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def migrate_legacy_data_dir(legacy_dir: Path, data_dir: Path) -> bool:
    """Moves `legacy_dir` to `data_dir` if the app now points somewhere else
    and a previous run already wrote real data into the old location.
    One-time and no-op in every other case (dev, a fresh install with
    nothing at the old path yet, or a previous run of this same migration)
    -- returns whether it actually moved anything, so callers can log it."""
    if legacy_dir == data_dir or data_dir.exists() or not legacy_dir.exists():
        return False
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(legacy_dir), str(data_dir))
    return True


DATA_DIR = Path(os.environ.get("MIMOID_DATA_DIR", _LEGACY_DATA_DIR))
if migrate_legacy_data_dir(_LEGACY_DATA_DIR, DATA_DIR):
    logging.getLogger(__name__).info("Migrated sidecar data from %s to %s", _LEGACY_DATA_DIR, DATA_DIR)
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "mimoid.db"
CHROMA_DIR = DATA_DIR / "chroma"
VOICE_SAMPLES_DIR = DATA_DIR / "voice_samples"
VOICE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Ollama chat generation options. Previously unset entirely, which left
# chat/persona-compare running at Ollama's server defaults (a small default
# num_ctx in particular is a real bug, not just a tuning gap -- it silently
# truncates the *oldest* messages first when a conversation overflows it,
# which is the system prompt, i.e. the persona/style/memories). 8192 covers a
# lengthy conversation on the target hardware (32GB M2) with room to spare.
# keep_alive keeps the model warm between turns instead of the default
# unload-after-5-minutes, avoiding a cold-reload stall on the next message.
OLLAMA_NUM_CTX = int(os.environ.get("MIMOID_OLLAMA_NUM_CTX", "8192"))
OLLAMA_TEMPERATURE = float(os.environ.get("MIMOID_OLLAMA_TEMPERATURE", "0.7"))
OLLAMA_REPEAT_PENALTY = float(os.environ.get("MIMOID_OLLAMA_REPEAT_PENALTY", "1.1"))
OLLAMA_KEEP_ALIVE = os.environ.get("MIMOID_OLLAMA_KEEP_ALIVE", "30m")
# Embedding requests can be slow when Ollama has to swap the embedding model
# in -- especially now that the chat model is pinned warm (OLLAMA_KEEP_ALIVE)
# at a large num_ctx, so a chat turn's query-embed may force a cold reload of
# nomic-embed-text. The old 30s cap could time out on that swap, and the
# timeout was silently swallowed (memories dropped with no trace). 120s gives
# the swap room; the embedding model is also kept warm between turns so the
# swap only happens once, not every message.
OLLAMA_EMBED_TIMEOUT = float(os.environ.get("MIMOID_OLLAMA_EMBED_TIMEOUT", "120"))

# Cap on how many raw style entries persona.py's no-guide fallback injects
# into the system prompt. Without a distilled guide, _style_section() used to
# inject every style entry ever imported/added -- with no cap on
# list_style_entries() and the WhatsApp/Signal importers' 100-entry cap being
# per-import (not overall), a handful of imports can accumulate hundreds of
# entries, which alone can exceed the context window before any conversation
# history or memories are even added. 40 is a reasonable few-shot-sized
# sample; once a distilled guide exists this cap doesn't apply at all (the
# guide replaces the raw examples entirely).
STYLE_EXAMPLES_FALLBACK_CAP = 40

# When a distilled style guide exists, persona.py additionally retrieves this
# many style entries most relevant to the current question and injects them
# as few-shot demonstrations alongside the guide. A guide alone is rules with
# zero examples; a few relevant real examples of the owner's own writing,
# chosen per-question, is the best-documented way to push a model's output
# closer to a specific voice.
STYLE_FEWSHOT_TOP_K = 3

MEMORY_TOP_K = 4

# Tool calling. Ollama silently ignores a `tools` payload for a model that
# wasn't trained for it -- no error, the model just never calls anything --
# so sending tools to e.g. plain llama3 would present as "the twin randomly
# doesn't use its tools" rather than as a misconfiguration. Matching is by
# prefix because Ollama tags carry a size/quant suffix ("qwen2.5:7b",
# "llama3.1:8b-instruct-q4_0"). This list is necessarily a snapshot; adding
# a name here is all that's needed to light tools up for a new model.
TOOL_CAPABLE_MODEL_PREFIXES = (
    "qwen2.5",
    "qwen3",
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "mistral-nemo",
    "mistral-small",
    "firefunction",
    "command-r",
    "hermes3",
)


def model_supports_tools(model: str) -> bool:
    return model.lower().startswith(TOOL_CAPABLE_MODEL_PREFIXES)


# How many tool rounds a single chat turn may run before we stop feeding
# results back. Guards against a model that keeps calling the same tool
# forever; 3 is enough for "look up the weather in two cities, then check
# the calendar" while bounding the worst case at a few extra generations.
TOOL_MAX_ITERATIONS = int(os.environ.get("MIMOID_TOOL_MAX_ITERATIONS", "3"))

# Per-tool network timeouts. Tools run *inside* the streaming generator, so
# a hung request doesn't just slow a background job -- it stalls a reply
# mid-sentence with the user watching. Deliberately much tighter than the
# unbounded timeouts used for local Ollama calls.
WEATHER_HTTP_TIMEOUT = float(os.environ.get("MIMOID_WEATHER_TIMEOUT", "5"))
GOOGLE_HTTP_TIMEOUT = float(os.environ.get("MIMOID_GOOGLE_TIMEOUT", "10"))

# Weather is looked up fresh at most this often per location; a cache hit
# needs no network call at all, which is also why it's not logged in
# network_activity -- nothing actually left the machine. 10 minutes is
# short enough that "has it started raining" stays accurate, long enough
# that asking about the same city twice in one conversation doesn't refetch.
WEATHER_CACHE_TTL = float(os.environ.get("MIMOID_WEATHER_CACHE_TTL", "600"))

# Empty until the owner adds their own places in the Tools tab -- there's no
# generic default city that isn't either an arbitrary opinion or a leak of
# wherever the person setting this up actually lives. An unconfigured
# location just means every lookup takes the live geocoding fallback in
# weather.py instead of the fast path, which is a fine default for a tool
# that's opt-in to begin with.
DEFAULT_WEATHER_LOCATIONS: list[dict] = []

# How long the sidecar's one-shot loopback listener waits for Google's
# redirect after "Connect" is clicked, before giving up and freeing the
# port. Generous -- the user has to actually go pick a Google account and
# click through consent in a real browser window, not just click a button.
GOOGLE_OAUTH_CALLBACK_TIMEOUT = float(os.environ.get("MIMOID_GOOGLE_OAUTH_TIMEOUT", "300"))

# Historical vault (world-knowledge RAG). The vault collection uses cosine
# distance, so VAULT_MAX_DISTANCE is on a fixed [0, 2] scale: results farther
# than this are considered irrelevant to the question and dropped -- that
# filtering is what keeps history snippets out of purely personal chats
# (and vice versa) without a separate routing model. 0.55 is an empirical
# starting point for nomic-embed-text; tune via the env var if the twin
# either misses history questions or drags history into personal ones.
VAULT_TOP_K = 4
VAULT_MAX_DISTANCE = float(os.environ.get("MIMOID_VAULT_MAX_DISTANCE", "0.55"))
# Reference prose benefits from bigger chunks than diary-length memories.
VAULT_CHUNK_CHARS = 1600
VAULT_OVERLAP_CHARS = 200
VAULT_MAX_UPLOAD_BYTES = 20 * 1024 * 1024

# Quality eval harness (Workstream C). Judging with the same model that
# generated the reply is a known-weak practice, but pulling a dedicated
# second model automatically isn't something to do silently -- default judge
# is whatever model is already being used day to day (zero extra setup),
# overridable to a stronger local model. Set EVAL_JUDGE_VIA_OPENROUTER to
# route judging through OpenRouter (config.OPENROUTER_MODEL) instead, for
# calibration against a model independent of the one being judged.
EVAL_JUDGE_MODEL = os.environ.get("MIMOID_EVAL_JUDGE_MODEL", DEFAULT_MODEL)
EVAL_JUDGE_VIA_OPENROUTER = os.environ.get("MIMOID_EVAL_JUDGE_VIA_OPENROUTER", "") == "1"
EVAL_PROMPT_BANK_PATH = Path(__file__).resolve().parent / "eval_prompts.json"

# Phase 7 (German), Phase 8 (Dutch). Codes match both mlx-whisper's and
# XTTS-v2's language arguments, so no per-engine mapping is needed.
SUPPORTED_LANGUAGES = {"en": "English", "de": "German", "nl": "Dutch"}
DEFAULT_LANGUAGE = "en"

# The owner's first name, set via the first-run onboarding screen and
# persisted in the settings table (see main.py's /settings). Empty here is
# only the pre-onboarding/defensive fallback -- the frontend blocks Talk/Train
# until a real name is saved, so build_persona_prompt() below should never
# actually see "" in practice.
DEFAULT_OWNER_NAME = os.environ.get("MIMOID_OWNER_NAME", "")


def build_persona_prompt(owner_name: str) -> str:
    """Base persona intro. persona.py layers the style corpus (or distilled
    style guide) and retrieved memories on top of this to build the full
    system prompt. Deliberately repeats owner_name instead of using a
    pronoun -- that also avoids hardcoding a gender for whoever's twin this
    is."""
    if not owner_name:
        return (
            "You are the person these memories, style, and history belong to. "
            "You are not an assistant describing them or representing them -- "
            "you ARE them, replying as yourself in the first person. Never "
            "refer to yourself as 'a digital twin', 'an AI', or describe "
            "yourself in the third person -- just answer naturally, using 'I', "
            "'me', and 'my'. When you share facts or history, it comes out in "
            "your own voice, not textbook prose. You don't need to reintroduce "
            "or reconfirm who you are -- only address your identity directly "
            "if you're explicitly asked about it. Match your reply's length to "
            "the question: a short, simple question gets a short, direct "
            "answer, not an unprompted essay. Never invent specific personal "
            "facts, names, dates, or events about your own life -- only state "
            "something as a real memory if it's actually grounded in what "
            "you're given below. If you're asked about a personal or "
            "biographical detail that isn't covered by your memories or known "
            "history, say plainly that you don't know or don't recall it "
            "instead of making something up."
        )
    return (
        f"You are {owner_name}. You are not an assistant describing "
        f"{owner_name} or representing {owner_name} -- you ARE {owner_name}, "
        "replying as yourself in the first person. Never refer to yourself as "
        f"'a digital twin', 'an AI', or describe {owner_name} in the third "
        f"person -- just answer naturally the way {owner_name} would, using "
        "'I', 'me', and 'my'. When you share facts or history, it comes out in "
        "your own voice, not textbook prose. You don't need to reintroduce or "
        "reconfirm who you are -- only address your identity directly if you're "
        "explicitly asked about it. Match your reply's length to the question: a "
        "short, simple question gets a short, direct answer, not an unprompted "
        "essay. Never invent specific personal facts, names, dates, or events "
        "about your own life -- only state something as a real memory if it's "
        "actually grounded in what you're given below. If you're asked about a "
        "personal or biographical detail that isn't covered by your memories or "
        "known history, say plainly that you don't know or don't recall it "
        "instead of making something up."
    )
