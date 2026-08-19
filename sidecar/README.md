# Mimoid sidecar

Python FastAPI process that owns all local ML work (chat/LLM orchestration,
STT, RAG, TTS). The Tauri app spawns this as a child process and the
frontend talks to it directly over localhost HTTP (CORS is open since this
is a machine-local, offline service).

## Run standalone

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8756 --reload
```

`GET /health` returns `{"status": "ok"}` once the sidecar is up.

`GET /activity` returns `{"activity": "<label>"|null}` — the backend task the
sidecar is running right now (Transcribing / Thinking / Speaking / Saving
memory / Distilling style / Downloading model / Importing chat / Backing up /
Comparing answers), or `null` when idle. Tracked via `app/activity.py`'s
`track()` context manager wrapped around the relevant handlers; the desktop
app polls this to drive the "current task" part of its status bar.

## Chat (Phase 1)

The sidecar proxies [Ollama](https://ollama.com) and persists conversations
to a local SQLite database (`sidecar/data/mimoid.db`, gitignored):

- `GET /config` — every configured model name (chat, embedding, Whisper, TTS,
  OpenRouter) plus whether an OpenRouter key is set; backs the frontend's
  System tab, which explains each component in plain language
- `GET /ollama/models` / `POST /ollama/pull` — list / download Ollama models
- `POST /conversations`, `GET /conversations` — create / list conversations
- `GET /conversations/{id}/messages` — full history
- `POST /conversations/{id}/messages` — send a message, streams the
  assistant's reply back as it's generated and persists both sides

## Memories / RAG (Phase 2)

Memories (Train mode entries) are stored twice: SQLite is the source of
truth for browsing/editing, and a [ChromaDB](https://www.trychroma.com)
collection (`sidecar/data/chroma`, gitignored) holds chunked, embedded
copies for semantic retrieval. Embeddings come from Ollama's
`nomic-embed-text`.

- `POST /memories` — create (content, optional topic/occurred_at); chunks,
  embeds, and indexes it
- `GET /memories` / `GET /memories?q=...` — list / substring-search memories
- `GET /memories/{id}`, `PUT /memories/{id}`, `DELETE /memories/{id}` — read,
  edit (re-indexes), delete (de-indexes)

`POST /conversations/{id}/messages` embeds the incoming message, retrieves
the top-k most relevant memories, and injects them into the system prompt
before calling the chat model. Retrieval failures degrade gracefully (chat
still proceeds without memories) rather than blocking the request.

## Ears / STT (Phase 3)

`POST /transcribe` accepts an uploaded audio file (`multipart/form-data`,
field name `file`) and returns `{"text": "..."}`, transcribed with
[mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
(English only, `language="en"`). Requires **Apple Silicon** — `mlx-whisper`
is imported lazily inside `stt.py` so the rest of the sidecar still runs
fine without it; if the import fails (any other machine), the route returns
a clean `503` instead of crashing.

`ffmpeg` must be installed and on `PATH` (mlx-whisper shells out to it to
decode whatever audio format the browser's `MediaRecorder` produced).

On first use, mlx-whisper downloads its model (~1-2 GB) before it can
transcribe anything. `stt.is_model_downloaded()` checks the local Hugging
Face cache so `/transcribe` can report a distinct "Downloading speech
model…" activity label instead of a misleading "Transcribing…" for what
might be several minutes; the whole request is bounded by
`STT_TIMEOUT_SECONDS` (8 min) and returns `504` with an actionable message
if it's exceeded, rather than hanging forever on a stalled download.

## Voice (Phase 4)

Voice cloning uses **XTTS-v2**, picked over the faster F5-TTS because it
supports German and Dutch natively -- both real, planned languages for this
app, where F5-TTS would need a third-party German checkpoint and a
from-scratch Dutch fine-tune -- via the `coqui-tts` package (the maintained
idiap fork — install `coqui-tts`, not `TTS`, which is unmaintained). The
model itself downloads from Hugging Face on first use (~1.8GB) and needs
`COQUI_TOS_AGREED=1` set before load to accept the CPML license
(non-commercial use only — see the root README's License section)
non-interactively — `tts.py` sets this itself.

- `POST /voice-samples` — upload a guided-prompt recording (`multipart/form-data`:
  `file`, `prompt`); converted to wav via `ffmpeg`, stored, and used as
  cloning reference audio
- `GET /voice-samples`, `DELETE /voice-samples/{id}` — browse / remove samples
- `GET /voice-samples/{id}/audio` — play back a stored sample
- `POST /speak` — body `{text}`, synthesizes with the cloned voice (using
  *all* stored samples as reference) and returns `audio/wav` bytes. Returns
  `400` if no samples are recorded yet, `503` if the TTS engine can't load,
  `504` if it takes longer than `TTS_TIMEOUT_SECONDS` (10 min).

Like `mlx-whisper`, `tts.py` imports `TTS.api` lazily so the rest of the
sidecar works even if TTS can't load (e.g. missing deps, no model
downloaded yet).

On first use, loading the model downloads it (~1.8 GB) before any synthesis
can happen. `tts.is_model_downloaded()` checks coqui-tts's local cache
(`get_user_data_dir("tts")/<model_full_name>`) so `/speak` can report a
distinct "Downloading voice model…" activity label instead of a misleading
"Speaking…" for what might be several minutes; `logger.info()`/`logger.exception()`
calls throughout `tts.py` (visible in the sidecar's console — `config.py`
calls `logging.basicConfig()` so these aren't silently dropped) trace model
loading, device selection, and synthesis start/success/failure for
debugging a stuck or failed request.

## Persona & style (Phase 5)

`persona.py` assembles the chat system prompt from three layers: the base
persona intro (`config.SYSTEM_PROMPT`), a style section, and retrieved
memories (Phase 2). The style section prefers a **distilled style guide**
if one exists, falling back to the raw style corpus (few-shot examples)
otherwise.

- `POST/GET/PUT/DELETE /style-entries` — CRUD for style corpus entries.
  Each is either a `text` sample (something Alex wrote) or a `qa` pair
  (question + "how Alex would answer it"), and carries a `source`
  (`manual` by default; `whatsapp`/`signal` for imported entries, see below;
  `own-answer` for entries saved from the Persona check tab's composer, see
  below). **Invariant:** every row is human-authored text — the
  `_ALLOWED_STYLE_ENTRY_SOURCES` allowlist in `main.py` is the single write
  chokepoint that enforces it, since this table feeds few-shot retrieval and
  every style-guide distillation, and model-generated text landing here
  would start a self-training loop. `persona-check` (see below) is a
  retired, write-blocked legacy source
- `DELETE /style-entries?source=<source>` — bulk-remove all entries with a
  given source (used to clear WhatsApp/Signal imports, or legacy
  `persona-check` entries); returns `{deleted: n}`
- `GET /style-guide` — the current distilled guide (`404` if none)
- `POST /style-guide/distill` — sends the style corpus to
  [OpenRouter](https://openrouter.ai) (a one-time paid cloud call, meant
  for occasional use on a small budget rather than routine local traffic)
  to distill it into a compact guide; `400` if no API key is configured
  (env var or in-app) or there's no corpus yet
- `POST /style-guide/distill/local` — distills the corpus with the configured
  local Ollama chat model; keeps every sample on the Mac, and returns `503` if
  Ollama or the configured model is unavailable
- `DELETE /style-guide` — clears the distilled guide, reverting to raw examples
- `GET /openrouter/config` — `{has_api_key}`, never the key itself
- `PUT /openrouter/config` — body `{api_key}`; saves the key in-app (via the
  Style tab's credentials modal), stored the same way as the Backup
  subtab's Nextcloud credentials (plaintext in the `settings` table). A
  saved key takes precedence over `MIMOID_OPENROUTER_API_KEY`, which
  remains a fallback default for advanced/dev setups

### Persona check: twin-vs-twin duels

Comparing the trained persona against a generic assistant always favored the
persona — a generic system prompt gives itself away immediately, so the
comparison taught nothing. Persona check instead pits two **arms** (system-
prompt assembly strategies) against each other, both fully in-voice:

| Arm | Style section |
|---|---|
| `guide_fewshot` (the default — what live chat actually uses) | distilled guide + retrieved examples |
| `guide_only` | distilled guide only |
| `fewshot_only` | retrieved examples only |
| `plain` | no style section at all |

Which two arms are eligible for a given question depends on what was
actually retrieved for it (e.g. `guide_fewshot` degenerates into
`guide_only` if no examples matched this particular prompt); the matchup is
the default arm against whichever eligible challenger has appeared least so
far (round-robin). If fewer than two arms are distinct for this prompt, the
compare route returns `400`.

- `POST /persona/duels` — body `{prompt}` → `{id, a, b}`. Which arm produced
  which answer is withheld until a choice is recorded, to keep the
  comparison blind. `400` if fewer than two arms are eligible for this
  prompt; `503` if Ollama is unreachable
- `POST /persona/duels/{id}/choice` — body `{choice: "a"|"b"|"tie"}`;
  records the pick and returns the summary below plus `{arm_a, arm_b}` (the
  reveal)
- `GET /persona/duel-summary` → per-arm win/loss/tie + appearance counts,
  the default arm's head-to-head win rate (ties excluded from the
  denominator), and the last 8 outcomes for the scoreboard's bar chart

The old persona-vs-generic tables/routes (`persona_eval`,
`GET /persona/eval-summary`, etc.) are gone from the live surface; the
`persona_eval` table itself is left in place as historical data, unread by
anything new.

**Growing the corpus without training on itself.** The Persona check tab's
composer captures what Alex would actually say *before* either generated
answer exists on screen — a blank textarea shown the instant Compare is
clicked, unlocked only after he clicks "Show the two answers" (never
auto-revealed), then locked read-only once they render. Saved via
`POST /style-entries` with `source: "own-answer"`. This replaces an earlier
flow (shipped, then retired) that prefilled a textarea with the *winning
generated answer* for Alex to edit — an edit step is a mitigation against
self-training, not a fix: anchoring means edits are light, and selection
bias means the answer picked (because it already sounds most like Alex) is
exactly the one least likely to get heavily edited. Entries saved by that
old flow are tagged `persona-check`, now a legacy, write-blocked source —
the Style tab surfaces a count and lets Alex remove them (never auto-deleted).

## WhatsApp import

`whatsapp.py` parses a WhatsApp chat export (`.txt`, or a `.zip` from
"Export chat") and turns the chosen participant's messages into style
corpus entries. Parsing is **structural** — it keys on each line's shape
(a timestamp, then `Sender: message`), not on parsing the date — so it
handles both Android (`12/05/2023, 14:23 - Alex: …`) and iOS
(`[12.05.2023, 14:23:45] Alex: …`) exports and their locale variants
without configuration. System messages, media/deleted placeholders, and
bare-URL messages are skipped.

- `POST /whatsapp/parse` — multipart upload (`file`); returns
  `{upload_id, participants: [{name, message_count}], total_messages}`.
  `400` if it doesn't look like an export. Parsed messages are cached
  in-memory keyed by `upload_id` (a single local process; the cache is
  lost on restart, in which case the user just re-uploads)
- `POST /whatsapp/import` — body `{upload_id, me}`; maps the chosen
  participant's messages to style entries (a reply to someone else →
  `qa`, a longer standalone message → `text`), capped at 100 (longest
  kept) and deduped, and inserts them with `source='whatsapp'`. `404` if
  the `upload_id` expired, `400` if `me` isn't a participant. Returns
  `{created, qa_count, text_count}`

The cap matters because the raw style corpus is injected into every chat
prompt until a style guide is distilled — the frontend nudges toward
`POST /style-guide/distill` right after an import.

## Signal import

`signal.py` parses Signal Desktop chat exports — **one `.txt` per
conversation** — into style corpus entries. The format is paragraph-based
(blank-line-separated): a `Conversation: Name (phone)` header, then message
paragraphs (`From:`, `Type: incoming|outgoing`, `Sent:`, optional
`Received:`) each followed by a body paragraph, plus bare system-event
paragraphs like `Type: keychange` (no `From:`, no body — skipped). No date
*parsing* is needed, same structural-only principle as the WhatsApp parser;
paragraph order preserves message order.

Unlike WhatsApp's export, Signal already labels which side is the export's
owner (`From: You` vs `From: <contact>`), so there's no "which participant
is me" step — messages map straight to style entries. The
message-to-entry mapping itself (burst-merging, qa-vs-text classification,
quality filters, dedupe, cap-to-100-keep-longest) is shared with the
WhatsApp importer via `whatsapp._collect_candidates` / `_cap_and_reorder`,
since none of that logic is WhatsApp-specific.

- `POST /signal/import` — multipart upload, repeated `files` field (one
  per conversation `.txt`). For each file: parses messages and the
  conversation name, pools candidates **across all files with one shared
  dedupe set** before applying the single 100-entry cap, and inserts them
  with `source='signal'`. `400` if none of the files look like a Signal
  export. Returns `{created, qa_count, text_count, conversations:
  [{name, message_count}]}`
- Removal reuses the existing `DELETE /style-entries?source=signal` route
  — no separate endpoint needed.

## Language (Phase 7)

A single persisted setting (`GET`/`PUT /settings`, body/response
`{"language": "en"|"de"}`, default `"en"`) drives Talk mode end to end:
`/transcribe` passes it to `mlx-whisper`, `/speak` passes it to XTTS-v2, and
every chat/persona system prompt gets a "Respond in German" instruction
appended when it's set to `"de"`. Both engines are multilingual out of the
box (part of why XTTS-v2 was picked for voice cloning — see "Voice" above)
— no model swap or re-recorded voice samples needed to switch languages.
Memories and the style corpus need no special handling either; they're
freeform text regardless of language.

## Nextcloud backup

`backup.py` builds a zip archive of a consistent SQLite snapshot (via
`sqlite3`'s backup API, not a raw file copy) plus all voice sample wavs,
and uploads it to the user's Nextcloud over WebDAV. ChromaDB's memory
index isn't included — it's derived from the memories in `mimoid.db`
and can be rebuilt by re-indexing, and copying a live index file-by-file
wouldn't be crash-consistent anyway. There's no restore tooling yet;
restoring means manually replacing `sidecar/data/mimoid.db` and
`sidecar/data/voice_samples/` from a downloaded archive.

- `GET /backup/config` — `{url, username, has_password, last_backup_at}`.
  The password itself is never returned, only whether one is set
- `PUT /backup/config` — body `{url, username, password?}`; omit/blank
  `password` to keep the one already stored; `400` on an invalid URL
- `POST /backup/run` — builds the archive and uploads it to
  `{url}/remote.php/dav/files/{username}/Mimoid/mimoid-backup-<timestamp>.zip`
  (creating the folder if needed); `400` if not configured yet, `502` with
  a specific message (bad credentials / unreachable host / other WebDAV
  error) if the upload fails

Configuration (URL/username/password, last-backup timestamp) is stored in
the existing `settings` table alongside the language setting.

Configurable via environment variables:

- `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`)
- `MIMOID_MODEL` (default `llama3`)
- `MIMOID_EMBEDDING_MODEL` (default `nomic-embed-text`)
- `MIMOID_WHISPER_MODEL` (default `mlx-community/whisper-turbo`)
- `MIMOID_TTS_MODEL` (default `tts_models/multilingual/multi-dataset/xtts_v2`)
- `MIMOID_OPENROUTER_API_KEY` (unset by default — a fallback if no key
  is saved in-app via the Style tab's credentials modal; style distillation
  is disabled if neither is set)
- `MIMOID_OPENROUTER_MODEL` (default `anthropic/claude-3.5-haiku`)
- `MIMOID_DATA_DIR` (default `sidecar/data`)
