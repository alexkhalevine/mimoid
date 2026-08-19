import { deleteStyleEntriesBySource, importSignalChats, listStyleEntries } from "./sidecar";

interface SignalElements {
  fileInput: HTMLInputElement;
  importButton: HTMLButtonElement;
  status: HTMLElement;
  error: HTMLElement;
  removeButton: HTMLButtonElement;
}

function getElements(): SignalElements | null {
  const fileInput = document.querySelector<HTMLInputElement>("#signal-file");
  const importButton = document.querySelector<HTMLButtonElement>("#signal-import-run");
  const status = document.querySelector<HTMLElement>("#signal-status");
  const error = document.querySelector<HTMLElement>("#signal-error");
  const removeButton = document.querySelector<HTMLButtonElement>("#signal-remove-imported");
  if (!fileInput || !importButton || !status || !error || !removeButton) return null;
  return { fileInput, importButton, status, error, removeButton };
}

async function refreshRemoveButton(els: SignalElements): Promise<void> {
  try {
    const entries = await listStyleEntries();
    els.removeButton.hidden = !entries.some((entry) => entry.source === "signal");
  } catch {
    els.removeButton.hidden = true;
  }
}

/** Signal already labels which side of the conversation is the export's
 * owner ("From: You" vs the contact's name), so unlike WhatsApp import
 * there's no participant to pick -- selecting file(s) is enough to import. */
export async function initSignalImport(onImported: () => void): Promise<void> {
  const els = getElements();
  if (!els) return;

  const resetTransient = () => {
    els.error.hidden = true;
    els.status.hidden = true;
  };

  els.fileInput.addEventListener("change", () => {
    resetTransient();
    els.importButton.hidden = !els.fileInput.files?.length;
  });

  els.importButton.addEventListener("click", () => {
    void (async () => {
      resetTransient();
      const files = Array.from(els.fileInput.files ?? []);
      if (files.length === 0) return;

      els.importButton.disabled = true;
      try {
        const result = await importSignalChats(files);
        const conversationList = result.conversations.map((c) => `${c.name} (${c.message_count})`).join(", ");
        els.status.textContent =
          result.created === 0
            ? "No new messages to import (they may already be imported)."
            : `Imported ${result.created} style ${result.created === 1 ? "example" : "examples"} ` +
              `(${result.qa_count} Q&A, ${result.text_count} writing) from ${conversationList}. ` +
              `Tip: run either distillation option above so chat prompts stay compact.`;
        els.status.hidden = false;
        onImported();
        await refreshRemoveButton(els);
      } catch (err) {
        els.error.textContent = (err as Error).message;
        els.error.hidden = false;
      } finally {
        els.importButton.disabled = false;
      }
    })();
  });

  els.removeButton.addEventListener("click", () => {
    void (async () => {
      resetTransient();
      els.removeButton.disabled = true;
      try {
        const deleted = await deleteStyleEntriesBySource("signal");
        els.status.textContent = `Removed ${deleted} imported ${deleted === 1 ? "entry" : "entries"}.`;
        els.status.hidden = false;
        onImported();
        await refreshRemoveButton(els);
      } catch (err) {
        els.error.textContent = (err as Error).message;
        els.error.hidden = false;
      } finally {
        els.removeButton.disabled = false;
      }
    })();
  });

  await refreshRemoveButton(els);
}
