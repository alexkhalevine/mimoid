import { invoke } from "@tauri-apps/api/core";
import { SIDECAR_BASE_URL } from "./config";
import { getStorageInfo } from "./sidecar";

const POLL_INTERVAL_MS = 2000;

interface SystemStats {
  cpu_percent: number;
  mem_used_bytes: number;
  total_mem_bytes: number;
  process_count: number;
}

// Known activity labels get a color that reads at a glance from the
// footer -- matches the voice orb's own color language (violet while
// listening, teal for everything else the twin is actively doing).
const TASK_COLOR_CLASS: Record<string, string> = {
  "Transcribing…": "system-bar-task-value--busy",
  "Downloading speech model… (first use, ~1-2 GB)": "system-bar-task-value--busy",
  "Thinking…": "system-bar-task-value--busy",
  "Speaking…": "system-bar-task-value--busy",
  "Saving memory…": "system-bar-task-value--busy",
  "Distilling style…": "system-bar-task-value--busy",
  "Downloading model…": "system-bar-task-value--busy",
  "Downloading voice model… (first use, ~1.8 GB)": "system-bar-task-value--busy",
  "Importing chat…": "system-bar-task-value--busy",
  "Backing up…": "system-bar-task-value--busy",
  "Comparing answers…": "system-bar-task-value--busy",
};

function formatGb(bytes: number): string {
  return `${(bytes / 1_073_741_824).toFixed(1)} GB`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1_048_576) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1_073_741_824) return `${(bytes / 1_048_576).toFixed(1)} MB`;
  return formatGb(bytes);
}

async function refreshStats(): Promise<void> {
  const summaryText = document.querySelector<HTMLElement>("#sysbar-summary-text");
  const popoverCpuValue = document.querySelector<HTMLElement>("#popover-cpu-value");
  const popoverCpuBar = document.querySelector<HTMLElement>("#popover-cpu-bar");
  const popoverRamValue = document.querySelector<HTMLElement>("#popover-ram-value");
  const popoverRamBar = document.querySelector<HTMLElement>("#popover-ram-bar");
  // Mirrors the same summary into the Talk tab's orb rail so it's visible
  // without leaving the tab (see the "Talk 2b" rail redesign).
  const railStats = document.querySelector<HTMLElement>("#rail-stats-value");

  try {
    const stats = await invoke<SystemStats>("system_stats");
    const cpuPct = Math.round(stats.cpu_percent);
    const used = formatGb(stats.mem_used_bytes);
    const ramText = stats.total_mem_bytes ? `${used} / ${formatGb(stats.total_mem_bytes)}` : used;
    const ramPct = stats.total_mem_bytes ? Math.min(100, (stats.mem_used_bytes / stats.total_mem_bytes) * 100) : 0;
    const summary = `CPU ${cpuPct}% · RAM ${ramText}`;

    if (summaryText) summaryText.textContent = summary;
    if (railStats) railStats.textContent = summary;
    if (popoverCpuValue) popoverCpuValue.textContent = `${cpuPct}%`;
    if (popoverCpuBar) popoverCpuBar.style.width = `${Math.min(100, cpuPct)}%`;
    if (popoverRamValue) popoverRamValue.textContent = ramText;
    if (popoverRamBar) popoverRamBar.style.width = `${ramPct}%`;
  } catch (err) {
    if (summaryText) summaryText.textContent = "CPU — · RAM —";
    if (railStats) railStats.textContent = "CPU — · RAM —";
    if (popoverCpuValue) popoverCpuValue.textContent = "—";
    if (popoverRamValue) popoverRamValue.textContent = "—";
    console.error("[monitor] system_stats failed", err);
  }
}

async function refreshActivity(): Promise<void> {
  const el = document.querySelector<HTMLElement>("#sysbar-activity");
  const railEl = document.querySelector<HTMLElement>("#rail-task-value");
  if (!el && !railEl) return;

  let label = "—";
  try {
    const res = await fetch(`${SIDECAR_BASE_URL}/activity`);
    if (!res.ok) throw new Error("activity request failed");
    const data = (await res.json()) as { activity: string | null };
    label = data.activity ?? "Idle";
  } catch {
    // Sidecar not up yet (e.g. during first-run setup) -- show a dash
    // rather than an alarming error.
    label = "—";
  }
  const busyClass = TASK_COLOR_CLASS[label] ?? "";
  if (el) {
    el.textContent = label;
    el.className = `system-bar-task-value ${busyClass}`.trim();
  }
  if (railEl) {
    railEl.textContent = label;
    railEl.className = `voice-dock-footer-value ${busyClass}`.trim();
  }
}

// ChromaDB size only changes when memories/style/vault entries are
// added or removed -- walking the directory tree on every 2s tick would
// be wasteful, so it gets its own slower interval.
const STORAGE_POLL_INTERVAL_MS = 30_000;

async function refreshStorage(): Promise<void> {
  const chromaSizeText = document.querySelector<HTMLElement>("#chroma-size-text");
  if (!chromaSizeText) return;

  try {
    const info = await getStorageInfo();
    chromaSizeText.textContent = formatBytes(info.chroma_bytes);
  } catch {
    chromaSizeText.textContent = "—";
  }
}

/** Starts the always-visible bottom status bar: Mimoid's CPU/RAM
 * footprint (from the Rust `system_stats` command) and the sidecar's
 * current task (from `GET /activity`), polled on an interval. Also feeds
 * the header/footer's shared system-details popover (see status.ts). */
export function startSystemMonitor(): void {
  const tick = () => {
    void refreshStats();
    void refreshActivity();
  };
  tick();
  setInterval(tick, POLL_INTERVAL_MS);

  void refreshStorage();
  setInterval(() => void refreshStorage(), STORAGE_POLL_INTERVAL_MS);
}
