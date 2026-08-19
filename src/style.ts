import {
  deleteStyleEntriesBySource,
  getStyleGuide,
  listImportBatches,
  listStyleEntries,
  type ImportBatch,
  type StyleEntry,
  type StyleGuide,
} from "./sidecar";
import { initStyleDetail } from "./styleDetail";
import { initStyleExamples } from "./styleExamples";
import { initStyleGuide } from "./styleGuide";
import { initStyleImport } from "./styleImport";

interface StyleTabEntry {
  tab: HTMLButtonElement;
  panel: HTMLElement;
}

/** A ~15-line local 3-way switcher for Style's own sub-tabs -- deliberately
 * not registered with main.ts's wireSubtabs(), which owns the 7-panel Train
 * subtab invariant and knows nothing about Style's internal tabs. */
function wireStyleTabs(): void {
  const tabs: (StyleTabEntry | null)[] = [
    { tabId: "style-tab-examples", panelId: "style-tabpanel-examples" },
    { tabId: "style-tab-guide", panelId: "style-tabpanel-guide" },
    { tabId: "style-tab-import", panelId: "style-tabpanel-import" },
  ].map(({ tabId, panelId }) => {
    const tab = document.querySelector<HTMLButtonElement>(`#${tabId}`);
    const panel = document.querySelector<HTMLElement>(`#${panelId}`);
    return tab && panel ? { tab, panel } : null;
  });
  if (tabs.some((entry) => entry === null)) return;
  const entries = tabs as StyleTabEntry[];

  const show = (target: StyleTabEntry) => {
    for (const entry of entries) {
      const active = entry === target;
      entry.panel.hidden = !active;
      entry.tab.classList.toggle("style-tab--active", active);
    }
  };

  for (const entry of entries) {
    entry.tab.addEventListener("click", () => show(entry));
  }
}

/** "persona-check" is a retired, write-blocked source (see main.py's
 * _ALLOWED_STYLE_ENTRY_SOURCES invariant comment): entries PR #76 saved from
 * a duel's winning ANSWER, i.e. model output, before the composer fix made
 * that impossible. Any such rows already on disk are surfaced here with a
 * count and an explicit removal action -- never auto-deleted, since it's
 * Alex's data and only he knows whether to keep or drop it. */
function renderLegacyNotice(entries: StyleEntry[]): void {
  const notice = document.querySelector<HTMLElement>("#style-legacy-notice");
  const countEl = document.querySelector<HTMLElement>("#style-legacy-notice-count");
  if (!notice || !countEl) return;
  const count = entries.filter((e) => e.source === "persona-check").length;
  countEl.textContent = String(count);
  notice.hidden = count === 0;
}

/** Wired once (not on every refresh) to avoid stacking duplicate listeners. */
function wireLegacyNotice(onRemoved: () => void): void {
  const button = document.querySelector<HTMLButtonElement>("#style-legacy-notice-remove");
  if (!button) return;
  button.addEventListener("click", () => {
    void (async () => {
      button.disabled = true;
      try {
        await deleteStyleEntriesBySource("persona-check");
        onRemoved();
      } catch (err) {
        console.error("failed to remove legacy persona-check entries", err);
      } finally {
        button.disabled = false;
      }
    })();
  });
}

/** Orchestrator for the Style page. Owns the entries/guide/batches state;
 * each sub-view (Examples, Guide, Import, the detail overlay) is wired
 * independently via its own init*(), which returns null on its own rather
 * than taking down the whole page if one element is missing. Must never
 * reject -- main.ts's Promise.all([...init*()]) has no .catch, so a
 * rejection here would leave every already-unhidden Train panel visible at
 * once instead of the intended default subtab. */
export async function initStyle(): Promise<void> {
  const panel = document.querySelector<HTMLElement>("#style-panel");
  if (!panel) return;
  panel.hidden = false;

  wireStyleTabs();

  let entries: StyleEntry[] = [];
  let guide: StyleGuide | null = null;
  let batches: ImportBatch[] = [];

  const getEntries = () => entries;
  const getGuide = () => guide;
  const getBatches = () => batches;

  wireLegacyNotice(() => void refreshEntries());

  const detail = initStyleDetail({ onChanged: () => void refreshEntries() });
  const examples = initStyleExamples({
    getEntries,
    onOpenNew: () => detail?.openNew(),
    onOpenEntry: (id) => {
      const entry = entries.find((e) => e.id === id);
      if (entry) detail?.openEdit(entry);
    },
  });
  const guideView = initStyleGuide({ getEntries, getGuide, refreshGuide: () => refreshGuideOnly() });
  const imports = await initStyleImport({ getBatches, onImported: () => void refreshAll() });

  async function refreshEntries(): Promise<void> {
    const listError = document.querySelector<HTMLElement>("#style-list-error");
    try {
      entries = await listStyleEntries();
      if (listError) listError.hidden = true;
    } catch (err) {
      console.error("failed to load style entries", err);
      if (listError) {
        listError.textContent = "Couldn't load style examples. Is the sidecar running?";
        listError.hidden = false;
      }
    }
    examples?.render();
    guideView?.render();
    renderLegacyNotice(entries);
  }

  async function refreshGuideOnly(): Promise<void> {
    try {
      guide = await getStyleGuide();
    } catch (err) {
      console.error("failed to load style guide", err);
    }
    guideView?.render();
  }

  async function refreshBatchesOnly(): Promise<void> {
    try {
      batches = await listImportBatches();
    } catch (err) {
      console.error("failed to load import history", err);
    }
    imports?.render();
  }

  async function refreshAll(): Promise<void> {
    await Promise.allSettled([refreshEntries(), refreshGuideOnly(), refreshBatchesOnly()]);
  }

  await refreshAll();
}
