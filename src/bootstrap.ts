import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export type BootstrapStatus =
  | { state: "checking" }
  | { state: "insufficient_resources"; message: string }
  | { state: "installing"; line: string }
  | { state: "ready" }
  | { state: "error"; message: string };

const EVENT_NAME = "sidecar-bootstrap-status";

interface BootstrapElements {
  panel: HTMLElement;
  message: HTMLParagraphElement;
  log: HTMLPreElement;
}

function getElements(): BootstrapElements | null {
  const panel = document.querySelector<HTMLElement>("#bootstrap-panel");
  const message = document.querySelector<HTMLParagraphElement>("#bootstrap-message");
  const log = document.querySelector<HTMLPreElement>("#bootstrap-log");
  if (!panel || !message || !log) return null;
  return { panel, message, log };
}

/**
 * Shows a first-run wizard while the Rust side creates the sidecar's Python
 * venv and installs its dependencies, then resolves once the sidecar is
 * ready to be spawned. On subsequent launches (bootstrap already completed)
 * this resolves immediately without showing anything. On a fatal error the
 * promise never resolves — the user has to fix the problem (free disk
 * space, add RAM) and restart the app.
 */
export async function runBootstrapWizard(): Promise<void> {
  const els = getElements();
  if (!els) return;
  const { panel, message, log } = els;

  let resolveReady: () => void;
  const ready = new Promise<void>((resolve) => {
    resolveReady = resolve;
  });

  const render = (status: BootstrapStatus) => {
    switch (status.state) {
      case "checking":
        panel.hidden = false;
        message.textContent = "Checking your system…";
        break;
      case "insufficient_resources":
        panel.hidden = false;
        message.textContent = status.message;
        break;
      case "installing":
        panel.hidden = false;
        message.textContent = "Setting up Mimoid for the first time — this only happens once…";
        log.hidden = false;
        log.textContent += `${status.line}\n`;
        log.scrollTop = log.scrollHeight;
        break;
      case "ready":
        panel.hidden = true;
        resolveReady();
        break;
      case "error":
        panel.hidden = false;
        message.textContent = `Setup failed: ${status.message}. Fix the issue and restart Mimoid.`;
        break;
    }
  };

  // Register the listener before polling the current state, so an event
  // fired in the gap between the two can't be missed.
  const unlisten = await listen<BootstrapStatus>(EVENT_NAME, (event) => render(event.payload));
  const current = await invoke<BootstrapStatus | null>("get_bootstrap_status");
  if (current) render(current);

  await ready;
  unlisten();
}
