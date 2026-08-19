import { invoke } from "@tauri-apps/api/core";
import { startOllama, stopOllama } from "./ollama";

const POLL_INTERVAL_MS = 3000;

type ServiceKey = "sidecar" | "ollama";

const commands: Record<ServiceKey, string> = {
  sidecar: "sidecar_status",
  ollama: "ollama_status",
};

const health: Record<ServiceKey, boolean> = { sidecar: false, ollama: false };

function updateOllamaToggle(healthy: boolean): void {
  const button = document.querySelector<HTMLButtonElement>("#ollama-toggle");
  if (!button) return;
  button.hidden = false;
  button.dataset.running = String(healthy);
  button.textContent = healthy ? "Stop" : "Start";
}

function updateReadyPill(): void {
  const dot = document.querySelector<HTMLElement>("#status-ready-dot");
  const text = document.querySelector<HTMLElement>("#status-ready-text");
  if (!dot || !text) return;

  const allHealthy = health.sidecar && health.ollama;
  dot.classList.toggle("ready-pill-dot--down", !allHealthy);
  if (allHealthy) {
    text.textContent = "Ready";
  } else if (!health.sidecar && !health.ollama) {
    text.textContent = "Starting…";
  } else if (!health.sidecar) {
    text.textContent = "Sidecar down";
  } else {
    text.textContent = "Ollama down";
  }
}

async function refreshStatus(service: ServiceKey): Promise<void> {
  const dot = document.querySelector<HTMLElement>(`#${service}-dot`);
  const text = document.querySelector<HTMLElement>(`#${service}-text`);
  if (!dot || !text) return;

  try {
    const healthy = await invoke<boolean>(commands[service]);
    dot.classList.toggle("status-dot--ok", healthy);
    dot.classList.toggle("status-dot--down", !healthy);
    text.textContent = healthy ? "running" : "not reachable";
    if (service === "ollama") updateOllamaToggle(healthy);
    health[service] = healthy;
  } catch (err) {
    dot.classList.remove("status-dot--ok");
    dot.classList.add("status-dot--down");
    text.textContent = "error";
    health[service] = false;
    console.error(`[${service}] status check failed`, err);
  }
  updateReadyPill();
}

function wireOllamaToggle(): void {
  const button = document.querySelector<HTMLButtonElement>("#ollama-toggle");
  if (!button) return;

  button.addEventListener("click", async () => {
    button.disabled = true;
    const running = button.dataset.running === "true";
    try {
      await (running ? stopOllama() : startOllama());
      await refreshStatus("ollama");
    } catch (err) {
      console.error("failed to toggle ollama", err);
    } finally {
      button.disabled = false;
    }
  });
}

export function startStatusPolling(): void {
  wireOllamaToggle();
  const refreshAll = () => {
    (Object.keys(commands) as ServiceKey[]).forEach((service) => {
      void refreshStatus(service);
    });
  };
  refreshAll();
  setInterval(refreshAll, POLL_INTERVAL_MS);
}

/** Wires the "Ready" pill (header) and the footer's CPU/RAM summary button
 * to open/close the same system-details popover -- two entry points, one
 * shared panel (Sidecar/Ollama rows + CPU/RAM bars). */
export function initStatusPopover(): void {
  const popover = document.querySelector<HTMLElement>("#status-details-popover");
  const readyPill = document.querySelector<HTMLButtonElement>("#status-ready-pill");
  const summaryButton = document.querySelector<HTMLButtonElement>("#sysbar-summary");
  if (!popover || !readyPill) return;

  const setOpen = (open: boolean) => {
    popover.hidden = !open;
    readyPill.setAttribute("aria-expanded", String(open));
    summaryButton?.setAttribute("aria-expanded", String(open));
  };

  const toggle = () => setOpen(popover.hidden);

  readyPill.addEventListener("click", toggle);
  summaryButton?.addEventListener("click", toggle);

  document.addEventListener("click", (event) => {
    if (popover.hidden) return;
    const target = event.target as Node;
    if (popover.contains(target) || readyPill.contains(target) || summaryButton?.contains(target)) return;
    setOpen(false);
  });
}
