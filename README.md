# Mimoid

I really like books by the Polish writer Stanisław Lem. Inspired by one of his best, Solaris, I've made a humble attempt here to create a digital twin of yourself, if you like—completely offline and running on your local stack. Back in the day (and likely still today), people used personal diaries to collect memories and preserve them for later. This project isn't far off from that concept, except for its digital environment and its dependence on your input: talk to a version of yourself trained on your own memories, writing style, and voice.

This is a desktop app (no cloud depedencies to get up and running), reducing friction for new users to deal with all different components of LLM/RAG/VectorDB installation manually and wiring it all together.

Examples: 

1. Chat UI
   
<img width="2986" height="1854" alt="Screenshot 2026-08-15 at 21 58 16" src="https://github.com/user-attachments/assets/49fc8441-0560-4963-8eac-01ec79b67274" />
3. Training section, memory example:

<img width="2984" height="1858" alt="Screenshot 2026-08-15 at 21 58 30" src="https://github.com/user-attachments/assets/9afd62ed-e3b4-4cd8-91db-9da672855bf8" />

3. System info:

<img width="2952" height="1876" alt="Screenshot 2026-08-15 at 21 58 54" src="https://github.com/user-attachments/assets/fc337ff9-6929-45af-8766-39eb7b14d654" />
 

## Architecture

- **Shell:** [Tauri v2](https://tauri.app) — Rust core + system WebView, TypeScript frontend.
  Two views: **Talk** (a chat/voice interface to the twin) and **Train** (the
  owner's memories, voice samples, writing style — including importing a
  WhatsApp or Signal chat export — persona A/B testing, and Nextcloud backup if needed).
- **Sidecar:** Python FastAPI process (`sidecar/`) that owns all local ML —
  chat/LLM orchestration, retrieval, speech-to-text, text-to-speech (XTTS-v2
  voice cloning), persona/style assembly, plus persistence (SQLite for
  conversations/memories/voice samples/style corpus, ChromaDB for memory
  embeddings). The Rust core spawns and supervises it, restarting it if it crashes. The frontend
  talks to it directly over HTTP for chat/memories/voice (streaming) and
  Tauri commands for status/process control.
- **LLM:** [Ollama](https://ollama.com). The Rust core can detect, start, and
  stop it (`src-tauri/src/ollama.rs`); the sidecar proxies chat, embedding,
  and model management calls to it. The default chat model is `qwen2.5`,
  chosen because it supports tool calling (see below) — Ollama silently
  ignores tools on models that don't, so switching to one (e.g. plain
  `llama3`) turns the Tools tab off, with a warning explaining why.
- **Tools:** the twin can call a small set of functions mid-conversation
  instead of guessing — the current date/time, the weather (via
  Open-Meteo, no API key), and Google Calendar (create + read events).
  Which tools are available is configured in the Tools tab; a reply that
  used one says so beneath itself, and every tool run is recorded in the
  `tool_runs` table. Anything that leaves the machine also shows up in the
  footer's internet-access indicator, same as OpenRouter and backup
  traffic. Calendar reads see events across *every* calendar in the
  connected account, but writes only ever go to one calendar, named
  "Tasks" by default (configurable in the Tools tab) — never your primary
  calendar or any other, regardless of what's asked. Creating a calendar
  event drafts it first by default (a "Confirm before creating" setting in
  the Tools tab) — the twin never writes to your real calendar without you
  clicking Add in the Talk tab, unless you turn that off.

## Project layout

```
src/          frontend (TypeScript, Vite)
src-tauri/    Tauri Rust core: window, sidecar lifecycle, commands
sidecar/      Python FastAPI sidecar: STT / RAG / TTS
skills/       reference docs for the desktop framework and STT stack
```

## Run quickly locally on MAC (arm):

```
make install
make dev
```

## Prerequisites

- [Node.js](https://nodejs.org) 22+
- [Rust](https://www.rust-lang.org/tools/install) (stable) + the [Tauri prerequisites](https://tauri.app/start/prerequisites/) for your OS
- Python 3.11+ — on macOS, check `python3 --version` before your first
  build. The `python3` that ships with Xcode's Command Line Tools can sit on
  a much older version for years; the app detects this and looks for a
  newer `python3.1x` on `PATH` automatically (installable via
  `brew install python@3.11`), but starting from a working interpreter
  avoids the extra step.
- [Ollama](https://ollama.com) installed and running locally
- `ffmpeg` on `PATH` (used to decode/normalize recorded audio for STT and
  voice cloning)
  
Speech-to-text and voice cloning both run locally, on whichever accelerator
the machine has:

| | Speech-to-text | Voice cloning (XTTS-v2) |
|---|---|---|
| Apple Silicon | `mlx-whisper` on MLX | PyTorch, MPS-accelerated |
| Linux/Windows + NVIDIA | `openai-whisper` on PyTorch, CUDA | PyTorch, CUDA |
| Anything else | `openai-whisper` on PyTorch, CPU | PyTorch, CPU |

`requirements.txt` picks the right Whisper backend by platform marker, so
there's nothing to configure — the Ollama/RAG side has no platform-specific
dependencies at all. CPU-only machines work, just slower.

On Linux, see **[docs/arch-setup.md](docs/arch-setup.md)** for a from-scratch
setup (system packages, NVIDIA driver, Ollama as a systemd service).

### Microphone (push-to-talk)

The mic button needs OS permission. Use the **built app** (`make build` →
install the DMG), not `make dev` — the signed `.app` carries the
microphone entitlement and its `Info.plist` prompt, which is what lets
macOS grant access. The first time you tap the mic, macOS asks to allow
the microphone; approve it (or later flip it on under **System Settings →
Privacy & Security → Microphone**). If the button ever seems to do
nothing, that's where to look — the app now surfaces a message pointing
you there instead of failing silently.

## Development

```sh
make install   # npm install + sidecar venv/deps
make dev       # launch the app; Tauri spawns the sidecar automatically
```

The app window shows live status dots for the sidecar and Ollama — both
should turn green once running.

To iterate on the sidecar by itself (with hot reload, hitting it directly
over HTTP):

```sh
make sidecar-dev
```

See [`sidecar/README.md`](./sidecar/README.md) for sidecar-specific details.


## Building

```sh
make build
```

Produces a DMG (and a raw `.app`) on macOS, an NSIS installer on Windows,
and a `.deb` plus an AppImage on Linux — Tauri only builds the targets valid
for the host it runs on, so the one command does the right thing everywhere.
Output lands in `src-tauri/target/release/bundle/`.

The sidecar's Python source is
bundled as a Tauri resource; on first launch the app creates its own venv
under the app's data directory and installs `sidecar/requirements.txt` into
it, showing a first-run wizard with progress (`src/bootstrap.ts`) before
handing off to the existing Ollama/model setup flow. This only happens
once — subsequent launches skip straight past it.

Your memories, conversations, and voice samples live in that same app data
directory (`~/Library/Application Support/com.mimoid.app/data` on macOS) —
not inside the `.app` bundle, so reinstalling or updating never touches
them. (An earlier version did put this data inside the bundle; `config.py`
migrates it out automatically, once, the first time you launch a build
with this fix.)

`make build` produces an **unsigned** DMG — macOS Gatekeeper will complain
when opening it, and push-to-talk mic access won't work (hardened runtime
requires a real signature; see below). For a real distributable build, use
the signed CI pipeline instead.

## Windows build (experimental, GitHub Actions, manual)

The `Windows build` workflow (`.github/workflows/release-windows.yml`)
builds an **unsigned** `.exe` installer on demand — Actions tab → "Windows
build" → "Run workflow". No secrets required. It's unsigned, so Windows
SmartScreen will warn on first run ("Windows protected your PC" → "More
info" → "Run anyway").

Mimoid was designed and built for macOS first, so the Windows build has
one real gap today:

- **Works**: text chat, memories, style corpus, the historical vault, voice
  input, and voice cloning. Speech-to-text runs `openai-whisper` on
  PyTorch — CUDA-accelerated with an NVIDIA GPU, CPU otherwise (see the
  Prerequisites table above); it's slower than the Apple Silicon `mlx-whisper`
  path, not unavailable.
- **Needs a manual prerequisite**: voice cloning (recording samples and
  hearing replies in your voice) shells out to `ffmpeg`, which isn't
  bundled — install it yourself and make sure it's on `PATH` (the same
  "you install this separately" pattern as Ollama).

### Setting up Google Calendar if needed

The calendar tool needs a Google Cloud OAuth client, since Google requires
one per app (there's no way around this for personal-Gmail calendar
access):

1. In the [Google Cloud Console](https://console.cloud.google.com/), create
   a project (or reuse one) and enable the **Google Calendar API**.
2. Under **APIs & Services → Credentials**, create an OAuth client ID of
   type **Desktop app**.
3. Under **APIs & Services → OAuth consent screen**, add your own Google
   account as a test user (the app stays in "Testing" status — that's
   fine, it's just for you).
4. In Mimoid's Tools tab, open the Calendar card's **Manage
   connection**, paste in the client ID and secret, then **Connect with
   Google**. This opens your system browser for Google's consent screen;
   the sidecar catches the redirect on a short-lived local port
   (`http://127.0.0.1:<port>/callback` — Google's "Desktop app" client type
   allows any loopback port, so no fixed redirect URI needs registering)
   and exchanges it for a token, stored only on this Mac.
5. Make sure the connected account has a calendar named **Tasks** (a
   secondary calendar, not your primary one — create it under Google
   Calendar's "Other calendars" if it doesn't exist yet). The twin can read
   events from every calendar in the account, but it only ever *writes*
   new events to the one named here (configurable in the Tools tab if you'd
   rather use a different name) — never to your primary calendar.

## Backing up to Nextcloud

Train → Backup lets you point the app at your own Nextcloud (URL,
username, and an [app password](https://docs.nextcloud.com/server/latest/user_manual/en/session_management.html#managing-devices))
and upload a zipped snapshot of your conversations, memories, voice
samples, and style corpus on demand. The Nextcloud credentials are stored
locally in the sidecar's SQLite database — never sent anywhere except
your own Nextcloud server. There's no automated restore yet; see
[`sidecar/README.md`](./sidecar/README.md) for what's included and what
isn't (the ChromaDB memory index is intentionally excluded — it's
rebuildable from the backed-up SQLite data).

## License

Mimoid's own code is [MIT licensed](./LICENSE).

Two dependencies carry their own separate terms worth knowing about:

- **Voice cloning (XTTS-v2)** ships under Coqui's Public Model License
  (CPML 1.0), which permits non-commercial use only — fine for this app's
  private, personal-use design, but a reason not to repurpose that model
  for anything commercial. `tts.py` accepts these terms non-interactively
  on first load; see the model's own license file for the exact text.
- **Bundled fonts** (Space Grotesk, IBM Plex Mono, Spectral, self-hosted
  under `public/fonts/` for offline use) are each [SIL Open Font
  License](https://openfontlicense.org) 1.1 — see
  [`public/fonts/NOTICE.md`](./public/fonts/NOTICE.md) for the exact license
  text and copyright line per family.
