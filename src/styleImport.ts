import type { ImportBatch } from "./sidecar";
import { initSignalImport } from "./signalImport";
import { initWhatsappImport } from "./whatsappImport";
import { formatRelativeDay } from "./styleFormat";

interface StyleImportElements {
  whatsappCard: HTMLButtonElement;
  signalCard: HTMLButtonElement;
  whatsappModal: HTMLDialogElement;
  signalModal: HTMLDialogElement;
  whatsappModalClose: HTMLButtonElement;
  signalModalClose: HTMLButtonElement;
  historyList: HTMLElement;
}

function getElements(): StyleImportElements | null {
  const whatsappCard = document.querySelector<HTMLButtonElement>("#style-import-whatsapp-card");
  const signalCard = document.querySelector<HTMLButtonElement>("#style-import-signal-card");
  const whatsappModal = document.querySelector<HTMLDialogElement>("#style-whatsapp-modal");
  const signalModal = document.querySelector<HTMLDialogElement>("#style-signal-modal");
  const whatsappModalClose = document.querySelector<HTMLButtonElement>("#style-whatsapp-modal-close");
  const signalModalClose = document.querySelector<HTMLButtonElement>("#style-signal-modal-close");
  const historyList = document.querySelector<HTMLElement>("#style-import-history");
  if (
    !whatsappCard ||
    !signalCard ||
    !whatsappModal ||
    !signalModal ||
    !whatsappModalClose ||
    !signalModalClose ||
    !historyList
  ) {
    return null;
  }
  return { whatsappCard, signalCard, whatsappModal, signalModal, whatsappModalClose, signalModalClose, historyList };
}

export interface StyleImportCallbacks {
  getBatches: () => ImportBatch[];
  /** Importing changes both the entry corpus and the batch history, so this
   * tells the orchestrator to refetch everything and re-render every view,
   * not just this one. */
  onImported: () => void;
}

export interface StyleImportView {
  render: () => void;
}

function renderHistory(els: StyleImportElements, batches: ImportBatch[]): void {
  els.historyList.innerHTML = "";
  if (batches.length === 0) {
    const empty = document.createElement("li");
    empty.className = "style-import-history-empty";
    empty.textContent = "No imports yet.";
    els.historyList.appendChild(empty);
    return;
  }
  for (const batch of batches) {
    const item = document.createElement("li");
    item.className = "style-import-history-item";
    const sourceLabel = batch.source === "whatsapp" ? "WhatsApp" : "Signal";
    const countLabel = `${batch.entry_count} example${batch.entry_count === 1 ? "" : "s"}`;
    item.textContent = `${sourceLabel} · ${batch.filename} — ${countLabel} · ${formatRelativeDay(batch.created_at)}`;
    els.historyList.appendChild(item);
  }
}

/** Delegates the actual parse/import flows to the untouched whatsappImport.ts
 * / signalImport.ts modules -- this module only owns the surrounding cards,
 * modals, and the recent-imports history list. */
export async function initStyleImport(callbacks: StyleImportCallbacks): Promise<StyleImportView | null> {
  const els = getElements();
  if (!els) return null;

  els.whatsappCard.addEventListener("click", () => els.whatsappModal.showModal());
  els.signalCard.addEventListener("click", () => els.signalModal.showModal());
  els.whatsappModalClose.addEventListener("click", () => els.whatsappModal.close());
  els.signalModalClose.addEventListener("click", () => els.signalModal.close());

  const render = () => {
    renderHistory(els, callbacks.getBatches());
  };

  await Promise.all([
    initWhatsappImport(() => callbacks.onImported()),
    initSignalImport(() => callbacks.onImported()),
  ]);

  return { render };
}
