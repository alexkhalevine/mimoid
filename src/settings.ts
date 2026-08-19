import { SIDECAR_BASE_URL } from "./config";

export const SUPPORTED_LANGUAGES: Record<string, string> = {
  en: "English",
  de: "German",
  nl: "Dutch",
};

interface Settings {
  language: string;
  pause_ms: number;
  owner_name: string;
}

async function fetchSettings(): Promise<Settings> {
  const res = await fetch(`${SIDECAR_BASE_URL}/settings`);
  if (!res.ok) throw new Error("failed to load settings");
  return (await res.json()) as Settings;
}

// The endpoint takes either field on its own, so the language selector and the
// voice pause slider can each save what they own without clobbering the other.
async function putSettings(patch: Partial<Settings>): Promise<Settings> {
  const res = await fetch(`${SIDECAR_BASE_URL}/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) throw new Error("failed to update settings");
  return (await res.json()) as Settings;
}

export async function getLanguage(): Promise<string> {
  return (await fetchSettings()).language;
}

export async function setLanguage(language: string): Promise<string> {
  return (await putSettings({ language })).language;
}

// --- Voice pause length -----------------------------------------------------
// How long the voice loop waits after you stop talking before it ends the turn.
// Cached at init so voice.ts can read it synchronously when it arms the silence
// detector, rather than awaiting the sidecar on every listening turn.
// Matches SILENCE_MS in vad.ts. The slider's range is deliberately narrower
// than what the sidecar accepts (500-5000) -- these are the values worth
// offering, not the limits of what's valid.
const DEFAULT_PAUSE_MS = 2000;
const MIN_PAUSE_MS = 1000;
const MAX_PAUSE_MS = 4000;
const PAUSE_STEP_MS = 250;

let pauseMs = DEFAULT_PAUSE_MS;

export function getPauseMs(): number {
  return pauseMs;
}

function formatPause(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`;
}

// --- Owner name --------------------------------------------------------------
// Set once at first-run onboarding (see onboarding.ts) and threaded into every
// place that used to hardcode "Alex" -- the persona's system prompt and a
// handful of UI strings ("hear in <name>'s voice", the voice-sample script).
// Cached here for the same reason pauseMs is: those UI strings are built once,
// synchronously, when their owning module initializes (after onboarding has
// already resolved), not re-fetched on every render.
export const MAX_OWNER_NAME_LENGTH = 60;

let ownerName = "";

export function getOwnerName(): string {
  return ownerName;
}

/** Fetches the currently saved owner name (empty string if never set) and
 * caches it. Used both by onboarding.ts (to decide whether to show the
 * first-run screen at all) and by initOwnerNameField() below. */
export async function loadOwnerName(): Promise<string> {
  const settings = await fetchSettings();
  ownerName = settings.owner_name;
  return ownerName;
}

/** Persists a new owner name and updates the cache. Throws (with the sidecar's
 * validation message surfaced via a non-ok response) on an empty/overlong name
 * or a network failure -- callers decide how to show that. */
export async function saveOwnerName(name: string): Promise<string> {
  const settings = await putSettings({ owner_name: name });
  ownerName = settings.owner_name;
  return ownerName;
}

/** Populates and wires the System tab's "Your name" field -- the "you can
 * change this later in System" the first-run screen promises. Same
 * disable-while-saving / revert-on-failure contract as initLanguageSelector. */
export async function initOwnerNameField(): Promise<void> {
  const form = document.querySelector<HTMLFormElement>("#system-owner-name-form");
  const input = document.querySelector<HTMLInputElement>("#system-owner-name-input");
  const saveButton = document.querySelector<HTMLButtonElement>("#system-owner-name-save");
  const errorEl = document.querySelector<HTMLElement>("#system-owner-name-error");
  if (!form || !input || !saveButton || !errorEl) return;

  input.maxLength = MAX_OWNER_NAME_LENGTH;
  input.value = ownerName;

  const showError = (message: string) => {
    errorEl.textContent = message;
    errorEl.hidden = false;
  };

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    void (async () => {
      const requested = input.value.trim();
      if (!requested) {
        showError("Enter a name before saving.");
        return;
      }
      errorEl.hidden = true;
      input.disabled = true;
      saveButton.disabled = true;
      saveButton.classList.add("btn-loading");
      try {
        const saved = await saveOwnerName(requested);
        input.value = saved;
      } catch (err) {
        console.error("failed to update owner name", err);
        showError("Couldn't save that name -- try again.");
        input.value = ownerName;
      } finally {
        input.disabled = false;
        saveButton.disabled = false;
        saveButton.classList.remove("btn-loading");
      }
    })();
  });
}

/** Populates and wires the header language selector. Safe to call once the
 * sidecar is confirmed reachable (it fetches the current setting on init). */
export async function initLanguageSelector(): Promise<void> {
  const select = document.querySelector<HTMLSelectElement>("#language-select");
  if (!select) return;

  select.innerHTML = "";
  for (const [code, label] of Object.entries(SUPPORTED_LANGUAGES)) {
    const option = document.createElement("option");
    option.value = code;
    option.textContent = label;
    select.appendChild(option);
  }

  let currentLanguage = "en";
  try {
    currentLanguage = await getLanguage();
  } catch (err) {
    console.error("failed to load current language", err);
  }
  select.value = currentLanguage;
  select.hidden = false;

  select.addEventListener("change", () => {
    void (async () => {
      const requested = select.value;
      select.disabled = true;
      try {
        currentLanguage = await setLanguage(requested);
      } catch (err) {
        console.error("failed to update language", err);
        select.value = currentLanguage;
      } finally {
        select.disabled = false;
      }
    })();
  });
}

/** Populates and wires the Talk tab's "pause before replying" slider. Same
 * contract as initLanguageSelector: safe to call once the sidecar is reachable. */
export async function initVoicePauseControl(): Promise<void> {
  const slider = document.querySelector<HTMLInputElement>("#voice-pause-slider");
  const readout = document.querySelector<HTMLElement>("#voice-pause-value");
  if (!slider || !readout) return;

  // The markup carries the same numbers so the control looks right before this
  // runs; these make the module the single source of truth for the range.
  slider.min = String(MIN_PAUSE_MS);
  slider.max = String(MAX_PAUSE_MS);
  slider.step = String(PAUSE_STEP_MS);

  try {
    const settings = await fetchSettings();
    if (Number.isFinite(settings.pause_ms)) pauseMs = settings.pause_ms;
  } catch (err) {
    console.error("failed to load voice pause setting", err);
  }
  slider.value = String(pauseMs);
  readout.textContent = formatPause(pauseMs);

  // Live readout while dragging; only persist once the drag settles, so a
  // single slide doesn't fire a PUT per pixel.
  slider.addEventListener("input", () => {
    readout.textContent = formatPause(Number(slider.value));
  });

  slider.addEventListener("change", () => {
    void (async () => {
      const requested = Number(slider.value);
      slider.disabled = true;
      try {
        const settings = await putSettings({ pause_ms: requested });
        pauseMs = settings.pause_ms;
      } catch (err) {
        console.error("failed to update voice pause setting", err);
      } finally {
        // Snap back to whatever actually stuck -- the saved value on success,
        // the previous one if the save failed.
        slider.value = String(pauseMs);
        readout.textContent = formatPause(pauseMs);
        slider.disabled = false;
      }
    })();
  });
}
