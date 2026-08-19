import { getOpenRouterConfig, saveOpenRouterApiKey } from "./sidecar";

interface OpenRouterModalElements {
  dialog: HTMLDialogElement;
  form: HTMLFormElement;
  apiKey: HTMLInputElement;
  error: HTMLElement;
  cancel: HTMLButtonElement;
  save: HTMLButtonElement;
}

function getElements(): OpenRouterModalElements | null {
  const dialog = document.querySelector<HTMLDialogElement>("#openrouter-modal");
  const form = document.querySelector<HTMLFormElement>("#openrouter-modal-form");
  const apiKey = document.querySelector<HTMLInputElement>("#openrouter-api-key");
  const error = document.querySelector<HTMLElement>("#openrouter-modal-error");
  const cancel = document.querySelector<HTMLButtonElement>("#openrouter-modal-cancel");
  const save = document.querySelector<HTMLButtonElement>("#openrouter-modal-save");
  if (!dialog || !form || !apiKey || !error || !cancel || !save) return null;
  return { dialog, form, apiKey, error, cancel, save };
}

let hasApiKey = false;
let pendingOnSaved: (() => void) | null = null;

/** Whether an OpenRouter API key is already configured (env var or saved via
 * this modal) -- lets callers open the modal proactively instead of letting
 * a doomed network call fail first. Reflects the state as of the last
 * `initOpenRouterModal()` fetch or successful save this session. */
export function hasOpenRouterApiKey(): boolean {
  return hasApiKey;
}

/** Opens the credentials modal. `onSaved`, if given, fires once right after
 * a successful save (and the dialog closing) -- used to auto-retry whatever
 * action the user was originally trying to do. */
export function openOpenRouterModal(onSaved?: () => void): void {
  const els = getElements();
  if (!els) return;
  pendingOnSaved = onSaved ?? null;
  els.error.hidden = true;
  els.apiKey.value = "";
  els.dialog.showModal();
  els.apiKey.focus();
}

export async function initOpenRouterModal(): Promise<void> {
  const els = getElements();
  if (!els) return;

  els.cancel.addEventListener("click", () => {
    pendingOnSaved = null;
    els.dialog.close();
  });

  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    const apiKey = els.apiKey.value.trim();
    if (!apiKey) return;

    els.error.hidden = true;
    els.save.disabled = true;
    els.save.classList.add("btn-loading");

    void (async () => {
      try {
        const config = await saveOpenRouterApiKey(apiKey);
        hasApiKey = config.has_api_key;
        els.dialog.close();
        const onSaved = pendingOnSaved;
        pendingOnSaved = null;
        onSaved?.();
      } catch (err) {
        els.error.textContent = (err as Error).message;
        els.error.hidden = false;
      } finally {
        els.save.disabled = false;
        els.save.classList.remove("btn-loading");
      }
    })();
  });

  try {
    const config = await getOpenRouterConfig();
    hasApiKey = config.has_api_key;
  } catch (err) {
    console.error("failed to load OpenRouter config", err);
  }
}
