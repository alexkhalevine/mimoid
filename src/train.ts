import { showError } from "./errorModal";
import { saveBlobAs } from "./fileSave";
import {
  createMemory,
  deleteMemory,
  exportMemories,
  importMemories,
  listMemories,
  updateMemory,
  type Memory,
  type MemoryInput,
} from "./sidecar";

interface TrainElements {
  panel: HTMLElement;
  form: HTMLFormElement;
  content: HTMLTextAreaElement;
  topic: HTMLInputElement;
  occurredAt: HTMLInputElement;
  submit: HTMLButtonElement;
  cancel: HTMLButtonElement;
  formError: HTMLElement;
  search: HTMLInputElement;
  list: HTMLElement;
  exportButton: HTMLButtonElement;
  importButton: HTMLButtonElement;
  importFile: HTMLInputElement;
  ioStatus: HTMLElement;
  ioError: HTMLElement;
}

function getElements(): TrainElements | null {
  const panel = document.querySelector<HTMLElement>("#train-panel");
  const form = document.querySelector<HTMLFormElement>("#memory-form");
  const content = document.querySelector<HTMLTextAreaElement>("#memory-content");
  const topic = document.querySelector<HTMLInputElement>("#memory-topic");
  const occurredAt = document.querySelector<HTMLInputElement>("#memory-occurred-at");
  const submit = document.querySelector<HTMLButtonElement>("#memory-submit");
  const cancel = document.querySelector<HTMLButtonElement>("#memory-cancel");
  const formError = document.querySelector<HTMLElement>("#memory-form-error");
  const search = document.querySelector<HTMLInputElement>("#memory-search");
  const list = document.querySelector<HTMLElement>("#memory-list");
  const exportButton = document.querySelector<HTMLButtonElement>("#memory-export");
  const importButton = document.querySelector<HTMLButtonElement>("#memory-import");
  const importFile = document.querySelector<HTMLInputElement>("#memory-import-file");
  const ioStatus = document.querySelector<HTMLElement>("#memory-io-status");
  const ioError = document.querySelector<HTMLElement>("#memory-io-error");
  if (
    !panel ||
    !form ||
    !content ||
    !topic ||
    !occurredAt ||
    !submit ||
    !cancel ||
    !formError ||
    !search ||
    !list ||
    !exportButton ||
    !importButton ||
    !importFile ||
    !ioStatus ||
    !ioError
  ) {
    return null;
  }
  return {
    panel,
    form,
    content,
    topic,
    occurredAt,
    submit,
    cancel,
    formError,
    search,
    list,
    exportButton,
    importButton,
    importFile,
    ioStatus,
    ioError,
  };
}

function renderMemory(memory: Memory, els: TrainElements, onChange: () => void): HTMLElement {
  const item = document.createElement("article");
  item.className = "memory-item";

  const meta = document.createElement("div");
  meta.className = "memory-item-meta";
  const parts = [memory.topic, memory.occurred_at].filter(Boolean);
  meta.textContent = parts.length ? parts.join(" · ") : "—";

  const content = document.createElement("p");
  content.className = "memory-item-content";
  content.textContent = memory.content;

  const actions = document.createElement("div");
  actions.className = "memory-item-actions";

  const editButton = document.createElement("button");
  editButton.type = "button";
  editButton.textContent = "Edit";
  editButton.addEventListener("click", () => {
    els.content.value = memory.content;
    els.topic.value = memory.topic ?? "";
    els.occurredAt.value = memory.occurred_at ?? "";
    els.form.dataset.editingId = memory.id;
    els.submit.textContent = "Save changes";
    els.cancel.hidden = false;
    els.content.focus();
  });

  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.textContent = "Delete";
  const attemptDelete = () => {
    void (async () => {
      deleteButton.disabled = true;
      try {
        await deleteMemory(memory.id);
        onChange();
      } catch (err) {
        deleteButton.disabled = false;
        showError("SYM-MEMORY-DELETE-FAILED", {
          detail: (err as Error).message,
          onPrimary: attemptDelete,
        });
      }
    })();
  };
  deleteButton.addEventListener("click", attemptDelete);

  actions.append(editButton, deleteButton);
  item.append(meta, content, actions);
  return item;
}

async function refreshList(els: TrainElements): Promise<void> {
  const memories = await listMemories(els.search.value.trim() || undefined);
  els.list.innerHTML = "";
  if (memories.length === 0) {
    const empty = document.createElement("p");
    empty.className = "memory-list-empty";
    empty.textContent = els.search.value.trim() ? "No matching memories." : "No memories yet.";
    els.list.appendChild(empty);
    return;
  }
  for (const memory of memories) {
    els.list.appendChild(renderMemory(memory, els, () => void refreshList(els)));
  }
}

function resetForm(els: TrainElements): void {
  els.form.reset();
  delete els.form.dataset.editingId;
  els.submit.textContent = "Add memory";
  els.cancel.hidden = true;
  els.formError.hidden = true;
}

function showIoStatus(els: TrainElements, message: string): void {
  els.ioError.hidden = true;
  els.ioStatus.textContent = message;
  els.ioStatus.hidden = false;
}

function showIoError(els: TrainElements, message: string): void {
  els.ioStatus.hidden = true;
  els.ioError.textContent = message;
  els.ioError.hidden = false;
}

function handleExportClick(els: TrainElements): void {
  void (async () => {
    els.exportButton.disabled = true;
    els.exportButton.classList.add("btn-loading");
    try {
      const blob = await exportMemories();
      const stamp = new Date()
        .toISOString()
        .replace(/[-:]/g, "")
        .replace(/\..+/, "")
        .replace("T", "-");
      const saved = await saveBlobAs(blob, `mimoid-memories-${stamp}.json`, ["json"]);
      if (saved) showIoStatus(els, "Memories exported.");
    } catch (err) {
      showIoError(els, (err as Error).message);
    } finally {
      els.exportButton.disabled = false;
      els.exportButton.classList.remove("btn-loading");
    }
  })();
}

function handleImportFileChosen(els: TrainElements): void {
  const file = els.importFile.files?.[0];
  // Cleared eagerly (not just on success) so choosing the exact same file
  // again after a failed import still fires a "change" event -- the input
  // otherwise treats it as no change at all.
  els.importFile.value = "";
  if (!file) return;

  void (async () => {
    els.importButton.disabled = true;
    els.importButton.classList.add("btn-loading");
    try {
      const result = await importMemories(file);
      const parts = [`Imported ${result.created}.`];
      if (result.skipped_duplicates) parts.push(`${result.skipped_duplicates} already saved.`);
      if (result.skipped_invalid) parts.push(`${result.skipped_invalid} couldn't be read.`);
      showIoStatus(els, parts.join(" "));
      await refreshList(els);
    } catch (err) {
      showIoError(els, (err as Error).message);
    } finally {
      els.importButton.disabled = false;
      els.importButton.classList.remove("btn-loading");
    }
  })();
}

export async function initTrain(): Promise<void> {
  const els = getElements();
  if (!els) return;

  els.panel.hidden = false;
  els.cancel.addEventListener("click", () => resetForm(els));
  els.exportButton.addEventListener("click", () => handleExportClick(els));
  els.importButton.addEventListener("click", () => els.importFile.click());
  els.importFile.addEventListener("change", () => handleImportFileChosen(els));

  let searchTimer: ReturnType<typeof setTimeout> | undefined;
  els.search.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => void refreshList(els), 200);
  });

  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    const input: MemoryInput = {
      content: els.content.value.trim(),
      topic: els.topic.value.trim() || undefined,
      occurred_at: els.occurredAt.value.trim() || undefined,
    };
    if (!input.content) return;

    const editingId = els.form.dataset.editingId;
    els.submit.disabled = true;
    els.formError.hidden = true;

    void (async () => {
      try {
        if (editingId) {
          await updateMemory(editingId, input);
        } else {
          await createMemory(input);
        }
        resetForm(els);
        await refreshList(els);
      } catch (err) {
        els.formError.textContent = (err as Error).message;
        els.formError.hidden = false;
      } finally {
        els.submit.disabled = false;
      }
    })();
  });

  await refreshList(els);
}
