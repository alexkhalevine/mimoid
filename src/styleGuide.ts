import {
  deleteStyleGuide,
  distillStyleGuide,
  distillStyleGuideLocally,
  type StyleEntry,
  type StyleGuide,
} from "./sidecar";
import { hasOpenRouterApiKey, openOpenRouterModal } from "./openrouterModal";
import { formatRelativeDay } from "./styleFormat";

interface StyleGuideElements {
  pipelineRaw: HTMLElement;
  pipelineGuide: HTMLElement;
  guideContent: HTMLElement;
  distillButton: HTMLButtonElement;
  localDistillButton: HTMLButtonElement;
  openrouterConfigureButton: HTMLButtonElement;
  clearGuideButton: HTMLButtonElement;
  distillError: HTMLElement;
  freshness: HTMLElement;
  statusDot: HTMLElement;
  statusLabel: HTMLElement;
}

function getElements(): StyleGuideElements | null {
  const pipelineRaw = document.querySelector<HTMLElement>("#style-pipeline-raw");
  const pipelineGuide = document.querySelector<HTMLElement>("#style-pipeline-guide");
  const guideContent = document.querySelector<HTMLElement>("#style-guide-content");
  const distillButton = document.querySelector<HTMLButtonElement>("#style-distill");
  const localDistillButton = document.querySelector<HTMLButtonElement>("#style-distill-local");
  const openrouterConfigureButton = document.querySelector<HTMLButtonElement>("#style-openrouter-configure");
  const clearGuideButton = document.querySelector<HTMLButtonElement>("#style-guide-clear");
  const distillError = document.querySelector<HTMLElement>("#style-distill-error");
  const freshness = document.querySelector<HTMLElement>("#style-guide-freshness");
  const statusDot = document.querySelector<HTMLElement>("#style-guide-status-dot");
  const statusLabel = document.querySelector<HTMLElement>("#style-guide-status-label");
  if (
    !pipelineRaw ||
    !pipelineGuide ||
    !guideContent ||
    !distillButton ||
    !localDistillButton ||
    !openrouterConfigureButton ||
    !clearGuideButton ||
    !distillError ||
    !freshness ||
    !statusDot ||
    !statusLabel
  ) {
    return null;
  }
  return {
    pipelineRaw,
    pipelineGuide,
    guideContent,
    distillButton,
    localDistillButton,
    openrouterConfigureButton,
    clearGuideButton,
    distillError,
    freshness,
    statusDot,
    statusLabel,
  };
}

export interface StyleGuideCallbacks {
  getEntries: () => StyleEntry[];
  getGuide: () => StyleGuide | null;
  /** Refetches the guide from the sidecar, updates the orchestrator's
   * cached copy, and calls this view's render() again. */
  refreshGuide: () => Promise<void>;
}

export interface StyleGuideView {
  render: () => void;
}

function renderPipeline(els: StyleGuideElements, entries: StyleEntry[], guide: StyleGuide | null): void {
  els.pipelineRaw.textContent = `${entries.length} raw example${entries.length === 1 ? "" : "s"}`;
  if (guide) {
    els.pipelineGuide.textContent = "Distilled guide ready";
    els.pipelineGuide.classList.remove("style-pipeline-step--pending");
    els.pipelineGuide.classList.add("style-pipeline-step--active");
  } else {
    els.pipelineGuide.textContent = "No distilled guide yet";
    els.pipelineGuide.classList.remove("style-pipeline-step--active");
    els.pipelineGuide.classList.add("style-pipeline-step--pending");
  }
}

function renderGuideContent(els: StyleGuideElements, guide: StyleGuide | null): void {
  if (guide) {
    els.guideContent.textContent = guide.content;
    els.clearGuideButton.hidden = false;
  } else {
    els.guideContent.textContent = "None yet — using raw examples below.";
    els.clearGuideButton.hidden = true;
  }
}

// A guide is stale if any entry was touched after the guide was generated,
// or if the corpus size no longer matches the size at distill time -- the
// timestamp check alone misses deletions, since removing an entry doesn't
// bump any other entry's updated_at.
function renderFreshness(els: StyleGuideElements, entries: StyleEntry[], guide: StyleGuide | null): void {
  if (!guide) {
    els.freshness.hidden = true;
    return;
  }
  els.freshness.hidden = false;
  const guideUpdatedAt = Date.parse(guide.updated_at);
  const stale =
    entries.some((entry) => Date.parse(entry.updated_at) > guideUpdatedAt) ||
    (guide.entry_count > 0 && entries.length !== guide.entry_count);

  els.statusDot.classList.toggle("style-guide-status-dot--stale", stale);
  els.statusDot.classList.toggle("style-guide-status-dot--fresh", !stale);

  const countClause =
    guide.entry_count > 0 ? ` from ${guide.entry_count} example${guide.entry_count === 1 ? "" : "s"}` : "";
  els.statusLabel.textContent = `${stale ? "Stale" : "Fresh"} — generated ${formatRelativeDay(guide.updated_at)}${countClause}`;
}

export function initStyleGuide(callbacks: StyleGuideCallbacks): StyleGuideView | null {
  const els = getElements();
  if (!els) return null;

  const render = () => {
    const entries = callbacks.getEntries();
    const guide = callbacks.getGuide();
    renderPipeline(els, entries, guide);
    renderGuideContent(els, guide);
    renderFreshness(els, entries, guide);
  };

  const runDistill = async (locally: boolean): Promise<void> => {
    // Both buttons write the same style_guide row, so both are disabled
    // while either runs (running them concurrently would race) -- but only
    // the one actually clicked shows the .btn-loading spinner, otherwise the
    // idle button falsely looked like it was working too.
    const activeButton = locally ? els.localDistillButton : els.distillButton;
    els.distillButton.disabled = true;
    els.localDistillButton.disabled = true;
    activeButton.classList.add("btn-loading");
    els.distillError.hidden = true;
    try {
      await (locally ? distillStyleGuideLocally() : distillStyleGuide());
      await callbacks.refreshGuide();
    } catch (err) {
      els.distillError.textContent = (err as Error).message;
      els.distillError.hidden = false;
    } finally {
      els.distillButton.disabled = false;
      els.localDistillButton.disabled = false;
      activeButton.classList.remove("btn-loading");
    }
  };

  els.distillButton.addEventListener("click", () => {
    if (!hasOpenRouterApiKey()) {
      openOpenRouterModal(() => void runDistill(false));
      return;
    }
    void runDistill(false);
  });

  els.localDistillButton.addEventListener("click", () => {
    void runDistill(true);
  });

  els.openrouterConfigureButton.addEventListener("click", () => {
    openOpenRouterModal();
  });

  els.clearGuideButton.addEventListener("click", () => {
    void (async () => {
      els.clearGuideButton.disabled = true;
      try {
        await deleteStyleGuide();
        await callbacks.refreshGuide();
      } catch (err) {
        console.error("failed to clear style guide", err);
      } finally {
        els.clearGuideButton.disabled = false;
      }
    })();
  });

  return { render };
}
