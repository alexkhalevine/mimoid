import { showError } from "./errorModal";
import {
  createStyleEntry,
  deleteStyleEntry,
  updateStyleEntry,
  type StyleEntry,
  type StyleEntryInput,
} from "./sidecar";
import { formatRelativeDay } from "./styleFormat";

interface StyleDetailElements {
  overlay: HTMLElement;
  backdrop: HTMLElement;
  panel: HTMLElement;
  typeTag: HTMLElement;
  sourceTag: HTMLElement;
  date: HTMLElement;
  close: HTMLButtonElement;
  kindInputs: NodeListOf<HTMLInputElement>;
  questionField: HTMLElement;
  question: HTMLInputElement;
  bodyLabel: HTMLElement;
  body: HTMLTextAreaElement;
  error: HTMLElement;
  save: HTMLButtonElement;
  cancel: HTMLButtonElement;
  deleteButton: HTMLButtonElement;
}

function getElements(): StyleDetailElements | null {
  const overlay = document.querySelector<HTMLElement>("#style-detail");
  const backdrop = document.querySelector<HTMLElement>("#style-detail-backdrop");
  const panel = document.querySelector<HTMLElement>("#style-detail-panel");
  const typeTag = document.querySelector<HTMLElement>("#style-detail-type-tag");
  const sourceTag = document.querySelector<HTMLElement>("#style-detail-source-tag");
  const date = document.querySelector<HTMLElement>("#style-detail-date");
  const close = document.querySelector<HTMLButtonElement>("#style-detail-close");
  const kindInputs = document.querySelectorAll<HTMLInputElement>('input[name="style-detail-kind"]');
  const questionField = document.querySelector<HTMLElement>("#style-detail-question-field");
  const question = document.querySelector<HTMLInputElement>("#style-detail-question");
  const bodyLabel = document.querySelector<HTMLElement>("#style-detail-body-label");
  const body = document.querySelector<HTMLTextAreaElement>("#style-detail-body");
  const error = document.querySelector<HTMLElement>("#style-detail-error");
  const save = document.querySelector<HTMLButtonElement>("#style-detail-save");
  const cancel = document.querySelector<HTMLButtonElement>("#style-detail-cancel");
  const deleteButton = document.querySelector<HTMLButtonElement>("#style-detail-delete");
  if (
    !overlay ||
    !backdrop ||
    !panel ||
    !typeTag ||
    !sourceTag ||
    !date ||
    !close ||
    kindInputs.length === 0 ||
    !questionField ||
    !question ||
    !bodyLabel ||
    !body ||
    !error ||
    !save ||
    !cancel ||
    !deleteButton
  ) {
    return null;
  }
  return {
    overlay,
    backdrop,
    panel,
    typeTag,
    sourceTag,
    date,
    close,
    kindInputs,
    questionField,
    question,
    bodyLabel,
    body,
    error,
    save,
    cancel,
    deleteButton,
  };
}

const SOURCE_LABEL: Record<string, string> = {
  whatsapp: "WHATSAPP IMPORT",
  signal: "SIGNAL IMPORT",
};

export interface StyleDetailCallbacks {
  /** The entry corpus changed (created, saved, or deleted) -- the
   * orchestrator refetches and re-renders every other view. */
  onChanged: () => void;
}

export interface StyleDetailView {
  openNew: () => void;
  openEdit: (entry: StyleEntry) => void;
}

export function initStyleDetail(callbacks: StyleDetailCallbacks): StyleDetailView | null {
  const els = getElements();
  if (!els) return null;

  type Mode = "closed" | "new" | "edit";
  let mode: Mode = "closed";
  let editingId: string | null = null;
  let returnFocus: HTMLElement | null = null;

  const currentKind = (): "text" | "qa" => {
    for (const input of els.kindInputs) if (input.checked) return input.value as "text" | "qa";
    return "text";
  };

  const setKind = (kind: "text" | "qa") => {
    for (const input of els.kindInputs) input.checked = input.value === kind;
    els.questionField.hidden = kind !== "qa";
    els.typeTag.textContent = kind === "qa" ? "Q&A example" : "Writing sample";
    els.bodyLabel.textContent = kind === "qa" ? "Answer" : "Writing sample";
  };

  const focusInitial = () => {
    (currentKind() === "qa" ? els.question : els.body).focus();
  };

  const discard = () => {
    mode = "closed";
    editingId = null;
    els.overlay.hidden = true;
    const toFocus = returnFocus;
    returnFocus = null;
    toFocus?.focus();
  };

  const openNew = () => {
    mode = "new";
    editingId = null;
    returnFocus = document.activeElement as HTMLElement | null;
    els.error.hidden = true;
    setKind("text");
    els.question.value = "";
    els.body.value = "";
    els.sourceTag.hidden = true;
    els.date.textContent = "New example";
    els.deleteButton.hidden = true;
    els.overlay.hidden = false;
    focusInitial();
  };

  const openEdit = (entry: StyleEntry) => {
    mode = "edit";
    editingId = entry.id;
    returnFocus = document.activeElement as HTMLElement | null;
    els.error.hidden = true;
    setKind(entry.kind);
    els.question.value = entry.prompt ?? "";
    els.body.value = entry.content;
    els.sourceTag.hidden = entry.source === "manual";
    els.sourceTag.textContent = SOURCE_LABEL[entry.source] ?? "";
    els.date.textContent = formatRelativeDay(entry.updated_at);
    els.deleteButton.hidden = false;
    els.overlay.hidden = false;
    focusInitial();
  };

  for (const input of els.kindInputs) {
    input.addEventListener("change", () => setKind(currentKind()));
  }

  els.save.addEventListener("click", () => {
    const kind = currentKind();
    const content = els.body.value.trim();
    if (!content) {
      els.error.textContent = "Write something before saving.";
      els.error.hidden = false;
      return;
    }
    els.error.hidden = true;
    const input: StyleEntryInput = {
      kind,
      content,
      prompt: kind === "qa" ? els.question.value.trim() || undefined : undefined,
    };

    els.save.disabled = true;
    els.save.classList.add("btn-loading");
    const savingId = editingId;
    const savingMode = mode;
    void (async () => {
      try {
        if (savingMode === "edit" && savingId) {
          await updateStyleEntry(savingId, input);
        } else {
          await createStyleEntry(input);
        }
        discard();
        callbacks.onChanged();
      } catch (err) {
        els.error.textContent = (err as Error).message;
        els.error.hidden = false;
      } finally {
        els.save.disabled = false;
        els.save.classList.remove("btn-loading");
      }
    })();
  });

  els.deleteButton.addEventListener("click", () => {
    const id = editingId;
    if (!id) return;
    const attemptDelete = () => {
      void (async () => {
        els.deleteButton.disabled = true;
        try {
          await deleteStyleEntry(id);
          discard();
          callbacks.onChanged();
        } catch (err) {
          showError("SYM-STYLE-DELETE-FAILED", {
            detail: (err as Error).message,
            onPrimary: attemptDelete,
          });
        } finally {
          els.deleteButton.disabled = false;
        }
      })();
    };
    attemptDelete();
  });

  els.cancel.addEventListener("click", () => discard());
  els.close.addEventListener("click", () => discard());
  els.backdrop.addEventListener("click", () => discard());

  const focusableElements = (): HTMLElement[] =>
    Array.from(els.panel.querySelectorAll<HTMLElement>("button, input, textarea")).filter(
      (el) => !(el as HTMLButtonElement).disabled && el.offsetParent !== null
    );

  // One document-level listener registered once, not per-open/close (which
  // would leak). No-ops while closed, and defers to an open import <dialog>
  // (native Esc already closes those) so Esc there doesn't also dismiss the
  // overlay underneath it.
  document.addEventListener("keydown", (event) => {
    if (mode === "closed") return;
    if (document.querySelector("dialog[open]")) return;

    if (event.key === "Escape") {
      event.preventDefault();
      discard();
      return;
    }
    if (event.key !== "Tab") return;
    const focusable = focusableElements();
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });

  return { openNew, openEdit };
}
