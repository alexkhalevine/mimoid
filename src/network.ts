import { SIDECAR_BASE_URL } from "./config";

const POLL_INTERVAL_MS = 2000;
// Even a fast OpenRouter round-trip should visibly register, not just
// flicker between polls -- so a finished call stays lit for at least this
// long once the frontend notices it, independent of how long it actually took.
const MIN_FLASH_MS = 2000;
const HISTORY_DISPLAY_LIMIT = 8;

interface NetworkEvent {
  id: number;
  label: string;
  at: number; // unix epoch seconds
}

interface NetworkActivityResponse {
  current: { id: number; label: string } | null;
  history: NetworkEvent[];
}

let lastSeenId = -1;
let offTimer: ReturnType<typeof setTimeout> | null = null;

function escapeHtml(text: string): string {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function setLit(label: string): void {
  const dot = document.querySelector<HTMLElement>("#network-dot");
  const text = document.querySelector<HTMLElement>("#network-current-label");
  const trigger = document.querySelector<HTMLButtonElement>("#network-indicator");
  if (!dot) return;
  dot.classList.add("network-dot--active");
  trigger?.setAttribute("title", label);
  if (text) text.textContent = label;
}

function setIdle(): void {
  const dot = document.querySelector<HTMLElement>("#network-dot");
  const text = document.querySelector<HTMLElement>("#network-current-label");
  const trigger = document.querySelector<HTMLButtonElement>("#network-indicator");
  if (!dot) return;
  dot.classList.remove("network-dot--active");
  trigger?.setAttribute("title", "No internet access right now");
  if (text) text.textContent = "Idle";
}

function renderHistory(history: NetworkEvent[]): void {
  const list = document.querySelector<HTMLElement>("#network-history-list");
  if (!list) return;

  if (history.length === 0) {
    list.innerHTML = `<li class="network-history-empty">No internet access yet this session.</li>`;
    return;
  }

  list.innerHTML = history
    .slice(0, HISTORY_DISPLAY_LIMIT)
    .map((event) => {
      const when = new Date(event.at * 1000).toLocaleTimeString();
      return (
        `<li class="network-history-row">` +
        `<span class="network-history-label">${escapeHtml(event.label)}</span>` +
        `<span class="network-history-time">${when}</span>` +
        `</li>`
      );
    })
    .join("");
}

async function refreshNetworkActivity(): Promise<void> {
  let data: NetworkActivityResponse;
  try {
    const res = await fetch(`${SIDECAR_BASE_URL}/network-activity`);
    if (!res.ok) throw new Error("network-activity request failed");
    data = (await res.json()) as NetworkActivityResponse;
  } catch {
    // Sidecar not reachable -- leave the dot in its last known state rather
    // than flashing it off, matching refreshActivity()'s dash-on-error style.
    return;
  }

  if (data.current) {
    if (offTimer) {
      clearTimeout(offTimer);
      offTimer = null;
    }
    setLit(data.current.label);
    lastSeenId = data.current.id;
  } else {
    const newest = data.history[0];
    if (newest && newest.id !== lastSeenId) {
      lastSeenId = newest.id;
      setLit(newest.label);
      if (offTimer) clearTimeout(offTimer);
      offTimer = setTimeout(() => {
        setIdle();
        offTimer = null;
      }, MIN_FLASH_MS);
    } else if (!offTimer) {
      setIdle();
    }
  }

  renderHistory(data.history);
}

function wirePopover(): void {
  const popover = document.querySelector<HTMLElement>("#network-popover");
  const trigger = document.querySelector<HTMLButtonElement>("#network-indicator");
  if (!popover || !trigger) return;

  const setOpen = (open: boolean) => {
    popover.hidden = !open;
    trigger.setAttribute("aria-expanded", String(open));
  };

  trigger.addEventListener("click", () => setOpen(popover.hidden));

  document.addEventListener("click", (event) => {
    if (popover.hidden) return;
    const target = event.target as Node;
    if (popover.contains(target) || trigger.contains(target)) return;
    setOpen(false);
  });
}

/** Starts the footer's internet-access indicator: lights up (with a
 * tooltip naming the component) whenever the sidecar reaches beyond the
 * local machine -- OpenRouter, Nextcloud backup, Ollama model downloads --
 * and never for local Ollama chat/embedding traffic. Polls
 * GET /network-activity on the same interval as the rest of the system bar. */
export function startNetworkActivityMonitor(): void {
  wirePopover();
  setIdle();
  void refreshNetworkActivity();
  setInterval(() => void refreshNetworkActivity(), POLL_INTERVAL_MS);
}
