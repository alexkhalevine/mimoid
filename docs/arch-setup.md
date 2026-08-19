# Running Mimoid on Arch Linux

A from-scratch setup for a bare Arch box, including an NVIDIA GPU. Written
for running from source (`make dev` / `make build`) — that's the supported
Linux path; there's no packaged Linux release.

Everything below assumes a working `sudo` and an up-to-date system:

```sh
sudo pacman -Syu
```

## 1. Build toolchain and Tauri's system dependencies

Tauri renders through WebKitGTK on Linux, which is the one genuinely
non-obvious dependency — without `webkit2gtk-4.1` the Rust build fails at
link time with a `pkg-config` error rather than anything about the GUI.

```sh
sudo pacman -S --needed \
  base-devel curl wget file git openssl \
  webkit2gtk-4.1 appmenu-gtk-module libappindicator-gtk3 librsvg \
  rust nodejs npm python python-pip
```

The first two lines of packages are Tauri's own documented Arch
prerequisites — if this list ever drifts, [Tauri's prerequisites
page](https://tauri.app/start/prerequisites/) is the authoritative source.

Notes:

- `base-devel` covers `gcc`, `make` and `pkg-config`.
- `rust` from the repos is fine; if you'd rather manage toolchains
  yourself, install [rustup](https://rustup.rs) instead and skip `rust`.
- Arch's `python` is 3.11+, which is what the sidecar needs.

## 2. Runtime dependencies

```sh
sudo pacman -S --needed ffmpeg lsof
```

- **`ffmpeg`** is required, not optional: the sidecar shells out to it to
  decode and normalize every recorded clip before speech-to-text or voice
  cloning sees it. Without it, recording appears to work and then fails on
  conversion.
- **`lsof`** is optional. It's used to clear a stale process off the
  sidecar's port (8756) at startup; without it the app falls back to `ss`
  from `iproute2`, which is already installed on any systemd Arch box.

## 3. Audio (microphone)

Push-to-talk needs a working input device that the WebKit webview can
reach through PipeWire:

```sh
sudo pacman -S --needed pipewire pipewire-pulse wireplumber pavucontrol
systemctl --user enable --now pipewire pipewire-pulse wireplumber
```

Confirm an input source exists and isn't muted (`pavucontrol` → *Input
Devices*, or `wpctl status`). If the mic button reports that the
microphone didn't respond, this is the first place to look.

## 4. NVIDIA driver (optional, but the whole point of a GPU box)

You need the **proprietary driver only** — *not* the CUDA toolkit. PyTorch's
PyPI wheels for Linux already bundle their own CUDA runtime (they install as
`torch==<version>+cuXXX` along with a set of `nvidia-*` packages), so
installing `cuda` system-wide just costs several GB and buys nothing here.

```sh
sudo pacman -S --needed nvidia nvidia-utils
sudo reboot
```

After the reboot:

```sh
nvidia-smi
```

If that prints your GPU and a driver version, you're done. Once the sidecar
venv exists (step 6) you can confirm PyTorch agrees:

```sh
sidecar/.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
# -> True 2.13.0+cu130
```

`True` means both speech-to-text and voice cloning will use the GPU. `False`
is not fatal — everything still runs on CPU, just slower.

> Using an LTS or custom kernel? Install the matching headers
> (`linux-lts-headers`, etc.) alongside the driver, or use `nvidia-dkms`.

## 5. Ollama

Mimoid does **not** start Ollama for you on Linux. On macOS the app can
spawn `ollama serve` as a child process, but here systemd owns the service,
its `ollama` user, and the models under `/usr/share/ollama` — a second
server started by the app would either lose the race to bind port 11434 or
win it and then not see any model the service had already pulled. So the app
detects Ollama and tells you what to run.

```sh
sudo pacman -S ollama          # or: ollama-cuda, for GPU-accelerated inference
sudo systemctl enable --now ollama
```

Then pull the two models the app expects:

```sh
ollama pull qwen2.5
ollama pull nomic-embed-text
```

(The setup screen can also pull these for you once Ollama is reachable.)

Verify:

```sh
curl -s http://127.0.0.1:11434/api/version
```

## 6. Build and run Mimoid

```sh
git clone git@github.com:alexkhalevine/mimoid.git
cd mimoid
npm install
make dev
```

`make dev` creates `sidecar/.venv` and installs the Python dependencies into
it on first run, then launches the Tauri app, which supervises the sidecar
itself. The first launch is slow — it downloads PyTorch and friends (a few
GB), and the speech and voice models download lazily on first use (~1–2 GB
for Whisper, ~1.8 GB for XTTS-v2).

To produce a bundle instead:

```sh
make build
```

This writes a `.deb` and an AppImage to
`src-tauri/target/release/bundle/`. Both are unsigned.

## Disk space

The preflight check refuses to bootstrap with less than 10 GB free. A
realistic full install lands around 15–20 GB:

| | approx. |
|---|---|
| PyTorch + CUDA runtime wheels | ~5–7 GB |
| Whisper `turbo` weights | ~1.6 GB |
| XTTS-v2 | ~1.8 GB |
| `qwen2.5` + `nomic-embed-text` | ~5 GB |
| Rust build artifacts (`src-tauri/target`) | ~2–3 GB |

## Troubleshooting

**`pkg-config` can't find `webkit2gtk-4.1` during `cargo build`** — the
package is missing, or you have the 4.0 series. Tauri v2 wants 4.1:
`sudo pacman -S webkit2gtk-4.1`.

**The window is blank, or rendering is broken** — WebKitGTK's DMA-BUF
renderer misbehaves on some drivers. Try
`WEBKIT_DISABLE_DMABUF_RENDERER=1 make dev`.

**`torch.cuda.is_available()` is `False` after installing the driver** — you
almost certainly haven't rebooted, or the running kernel and the driver
module are out of step after an update. Check `nvidia-smi` first: if that
fails too, it's the driver, not PyTorch.

**Speech-to-text says it isn't available** — the sidecar found no usable
Whisper backend. Reinstall the sidecar dependencies:
`rm -rf sidecar/.venv && make dev`. (`sidecar/.venv/bin/python -c "from app
import stt; print(stt._backend(), stt.active_model())"` should print
`torch turbo` on Linux.)

**Transcription is very slow** — you're on CPU. Check `nvidia-smi` and the
`torch.cuda.is_available()` probe above. As a stopgap, a smaller model
works: `MIMOID_WHISPER_TORCH_MODEL=small make dev`.
