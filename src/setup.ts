import { invoke } from "@tauri-apps/api/core";
import { ollamaInstalled, ollamaRunning, startOllama } from "./ollama";
import { getSidecarConfig, listModels, pullModel, type PullProgress } from "./sidecar";

// A fresh sidecar process can legitimately take a while to become reachable
// on its very first launch -- heavy Python imports (torch, TTS, mlx-whisper)
// can each take upwards of a minute the first time. 360 attempts * 500ms =
// 3 minutes before giving up; onSlow fires once (after ~10s) so the caller
// can reassure the user this is expected rather than a hang.
async function waitForSidecar(onSlow: () => void): Promise<boolean> {
  const maxAttempts = 360;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    if (await invoke<boolean>("sidecar_status")) return true;
    if (attempt === 20) onSlow();
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return false;
}

function modelInstalled(models: { name: string }[], target: string): boolean {
  return models.some((model) => model.name === target || model.name.startsWith(`${target}:`));
}

// Ollama is installed very differently per platform (a .app drag on macOS, a
// package + systemd unit on Linux), so the "go install it" copy has to differ
// too. Sniffing the user agent is enough for choosing wording -- nothing
// functional hangs off it, and the webview's UA is stable per platform.
function isLinux(): boolean {
  return /Linux|X11/.test(navigator.userAgent) && !/Android/.test(navigator.userAgent);
}

const OLLAMA_INSTALL_HINT = isLinux()
  ? "Ollama isn't installed. On Arch: `sudo pacman -S ollama` (or `ollama-cuda` for NVIDIA GPU acceleration), then `sudo systemctl enable --now ollama`."
  : "Ollama isn't installed. Install it from ollama.com, then come back here.";

function progressField(line: PullProgress, key: string): number | undefined {
  const value = line[key];
  return typeof value === "number" ? value : undefined;
}

interface SetupElements {
  panel: HTMLElement;
  message: HTMLParagraphElement;
  action: HTMLButtonElement;
  progress: HTMLProgressElement;
  spinner: HTMLElement;
}

function getElements(): SetupElements | null {
  const panel = document.querySelector<HTMLElement>("#setup-panel");
  const message = document.querySelector<HTMLParagraphElement>("#setup-message");
  const action = document.querySelector<HTMLButtonElement>("#setup-action");
  const progress = document.querySelector<HTMLProgressElement>("#setup-progress");
  const spinner = document.querySelector<HTMLElement>("#setup-spinner");
  if (!panel || !message || !action || !progress || !spinner) return null;
  return { panel, message, action, progress, spinner };
}

export async function runSetupFlow(onReady: () => void): Promise<void> {
  const els = getElements();
  if (!els) return;
  const { panel, message, action, progress, spinner } = els;

  panel.hidden = false;

  const setMessage = (text: string) => {
    message.textContent = text;
  };
  // Visible only while the flow is auto-progressing with nothing for the
  // user to do yet -- hidden the moment an action button (something to
  // click) or the real download progress bar takes over as the active
  // indicator instead.
  const setSpinner = (visible: boolean) => {
    spinner.hidden = !visible;
  };
  const setAction = (label: string | null, onClick?: () => void) => {
    if (!label) {
      action.hidden = true;
      return;
    }
    action.hidden = false;
    action.disabled = false;
    action.textContent = label;
    action.onclick = onClick ?? null;
  };

  setMessage("Waiting for the sidecar to start…");
  setAction(null);
  setSpinner(true);
  const sidecarReady = await waitForSidecar(() => {
    // Still polling past the point a healthy start normally takes -- a
    // fresh process can genuinely take a while on its first-ever launch
    // (heavy Python imports), so reassure rather than let this look hung.
    setMessage("Still starting… first launch can take a minute or two while models load.");
  });
  if (!sidecarReady) {
    setSpinner(false);
    setMessage("The sidecar didn't start. Check the app logs and restart Mimoid.");
    setAction("Retry", () => void runSetupFlow(onReady));
    return;
  }

  setMessage("Checking for Ollama…");
  if (!(await ollamaInstalled())) {
    setSpinner(false);
    setMessage(OLLAMA_INSTALL_HINT);
    setAction("I've installed it — check again", () => void runSetupFlow(onReady));
    return;
  }

  if (!(await ollamaRunning())) {
    setSpinner(false);
    if (isLinux()) {
      // systemd owns the service on Linux, so the app deliberately doesn't
      // spawn its own `ollama serve` (see src-tauri/src/ollama.rs). Offering
      // a "Start Ollama" button here would just surface the same explanation
      // one click later, so show it up front and re-check instead.
      setMessage(
        "Ollama is installed but not running. Start it with `systemctl --user start ollama`, or `sudo systemctl start ollama` if you installed it system-wide.",
      );
      setAction("I've started it — check again", () => void runSetupFlow(onReady));
      return;
    }
    setMessage("Ollama is installed but not running.");
    setAction("Start Ollama", () => {
      void (async () => {
        setAction("Starting…");
        action.disabled = true;
        setSpinner(true);
        try {
          await startOllama();
        } catch (err) {
          setSpinner(false);
          setMessage(`Couldn't start Ollama: ${(err as Error).message}`);
          setAction("Retry", () => void runSetupFlow(onReady));
          return;
        }
        await runSetupFlow(onReady);
      })();
    });
    return;
  }

  setMessage("Checking for required models…");
  setAction(null);
  setSpinner(true);
  const { model, embedding_model } = await getSidecarConfig();
  const models = await listModels();
  const requiredModels = [...new Set([model, embedding_model])];
  const missingModel = requiredModels.find((name) => !modelInstalled(models, name));

  if (missingModel) {
    setSpinner(false);
    setMessage(`Model "${missingModel}" isn't downloaded yet.`);
    setAction(`Download ${missingModel}`, () => {
      void (async () => {
        setAction(null);
        progress.hidden = false;
        progress.removeAttribute("value");
        try {
          await pullModel(missingModel, (line) => {
            const status = typeof line.status === "string" ? line.status : "";
            const completed = progressField(line, "completed");
            const total = progressField(line, "total");
            if (completed !== undefined && total) {
              progress.max = total;
              progress.value = completed;
            }
            setMessage(status || "Downloading…");
          });
        } catch (err) {
          setMessage(`Download failed: ${(err as Error).message}`);
          setAction("Retry", () => void runSetupFlow(onReady));
          progress.hidden = true;
          return;
        }
        progress.hidden = true;
        await runSetupFlow(onReady);
      })();
    });
    return;
  }

  panel.hidden = true;
  onReady();
}
