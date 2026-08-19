import {
  deleteStyleEntriesBySource,
  importWhatsappMessages,
  listStyleEntries,
  parseWhatsappExport,
  type WhatsappParticipant,
} from "./sidecar";

interface WhatsappElements {
  fileInput: HTMLInputElement;
  participants: HTMLElement;
  importButton: HTMLButtonElement;
  status: HTMLElement;
  error: HTMLElement;
  removeButton: HTMLButtonElement;
}

function getElements(): WhatsappElements | null {
  const fileInput = document.querySelector<HTMLInputElement>("#whatsapp-file");
  const participants = document.querySelector<HTMLElement>("#whatsapp-participants");
  const importButton = document.querySelector<HTMLButtonElement>("#whatsapp-import-run");
  const status = document.querySelector<HTMLElement>("#whatsapp-status");
  const error = document.querySelector<HTMLElement>("#whatsapp-error");
  const removeButton = document.querySelector<HTMLButtonElement>("#whatsapp-remove-imported");
  if (!fileInput || !participants || !importButton || !status || !error || !removeButton) {
    return null;
  }
  return { fileInput, participants, importButton, status, error, removeButton };
}

function renderParticipants(els: WhatsappElements, participants: WhatsappParticipant[]): void {
  els.participants.innerHTML = "";
  participants.forEach((participant, index) => {
    const label = document.createElement("label");
    label.className = "whatsapp-participant";

    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "whatsapp-me";
    radio.value = participant.name;
    radio.checked = index === 0; // default to the most active participant

    const text = document.createElement("span");
    text.textContent = `${participant.name} — ${participant.message_count} messages`;

    label.append(radio, text);
    els.participants.appendChild(label);
  });
  els.participants.hidden = participants.length === 0;
  els.importButton.hidden = participants.length === 0;
}

function selectedParticipant(els: WhatsappElements): string | null {
  const checked = els.participants.querySelector<HTMLInputElement>('input[name="whatsapp-me"]:checked');
  return checked?.value ?? null;
}

async function refreshRemoveButton(els: WhatsappElements): Promise<void> {
  try {
    const entries = await listStyleEntries();
    els.removeButton.hidden = !entries.some((entry) => entry.source === "whatsapp");
  } catch {
    els.removeButton.hidden = true;
  }
}

export async function initWhatsappImport(onImported: () => void): Promise<void> {
  const els = getElements();
  if (!els) return;

  const resetTransient = () => {
    els.error.hidden = true;
    els.status.hidden = true;
  };

  els.fileInput.addEventListener("change", () => {
    void (async () => {
      resetTransient();
      els.participants.hidden = true;
      els.importButton.hidden = true;
      const file = els.fileInput.files?.[0];
      if (!file) return;
      try {
        const result = await parseWhatsappExport(file);
        renderParticipants(els, result.participants);
      } catch (err) {
        els.error.textContent = (err as Error).message;
        els.error.hidden = false;
      }
    })();
  });

  els.importButton.addEventListener("click", () => {
    void (async () => {
      resetTransient();
      const file = els.fileInput.files?.[0];
      const me = selectedParticipant(els);
      if (!file || !me) return;

      els.importButton.disabled = true;
      try {
        // Re-parse on import so a server restart between steps can't leave a
        // stale/expired upload id; the parse is cheap.
        const parsed = await parseWhatsappExport(file);
        const result = await importWhatsappMessages(parsed.upload_id, me);
        els.status.textContent =
          result.created === 0
            ? "No new messages to import (they may already be imported)."
            : `Imported ${result.created} style ${result.created === 1 ? "example" : "examples"} ` +
              `(${result.qa_count} Q&A, ${result.text_count} writing). ` +
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
        const deleted = await deleteStyleEntriesBySource("whatsapp");
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
