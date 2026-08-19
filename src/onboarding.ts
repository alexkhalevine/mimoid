// First-run "what should it call you?" gate -- see the attached design
// (Mimoid_First_Run_Onboarding). A saved owner name doubles as the
// "onboarding already happened" flag: there's no separate first-run boolean,
// the presence of a non-empty name IS the signal (same shape as
// bootstrap.ts's runBootstrapWizard: resolves immediately if already done,
// otherwise blocks until settled).
import { loadOwnerName, saveOwnerName } from "./settings";

interface Elements {
  backdrop: HTMLElement;
  form: HTMLFormElement;
  input: HTMLInputElement;
  errorEl: HTMLElement;
  submit: HTMLButtonElement;
}

function getElements(): Elements | null {
  const backdrop = document.querySelector<HTMLElement>("#name-onboarding-backdrop");
  const form = document.querySelector<HTMLFormElement>("#name-onboarding-form");
  const input = document.querySelector<HTMLInputElement>("#name-onboarding-input");
  const errorEl = document.querySelector<HTMLElement>("#name-onboarding-error");
  const submit = document.querySelector<HTMLButtonElement>("#name-onboarding-submit");
  if (!backdrop || !form || !input || !errorEl || !submit) return null;
  return { backdrop, form, input, errorEl, submit };
}

/**
 * Resolves immediately (without showing anything) if an owner name is
 * already saved from a previous launch. Otherwise shows the full-screen
 * first-contact screen and blocks until a valid name is entered AND saved --
 * this must complete before initChat()/the persona pipeline runs, since the
 * system prompt needs a real name, not a placeholder.
 */
export async function runNameOnboarding(): Promise<void> {
  let existing = "";
  try {
    existing = await loadOwnerName();
  } catch (err) {
    // The sidecar is confirmed up by the time this runs (called from inside
    // runSetupFlow's onReady), so a failure here is unexpected -- fail open
    // rather than stranding the user on a screen that can't even load its
    // own state. The save path below still validates before persisting.
    console.error("[onboarding] failed to load owner name", err);
  }
  if (existing.trim()) return;

  const els = getElements();
  if (!els) return;

  const { backdrop, form, input, errorEl, submit } = els;

  const showError = (message: string) => {
    errorEl.textContent = message;
    errorEl.hidden = false;
  };
  const clearError = () => {
    errorEl.hidden = true;
  };

  backdrop.hidden = false;
  input.focus();

  await new Promise<void>((resolve) => {
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const name = input.value.trim();
      if (!name) {
        showError("Enter a name to continue.");
        return;
      }
      clearError();
      input.disabled = true;
      submit.disabled = true;
      submit.classList.add("btn-loading");
      void (async () => {
        try {
          await saveOwnerName(name);
          backdrop.hidden = true;
          resolve();
        } catch (err) {
          console.error("[onboarding] failed to save owner name", err);
          showError("Couldn't save that -- check your connection and try again.");
          input.disabled = false;
          submit.disabled = false;
          submit.classList.remove("btn-loading");
        }
      })();
    });
  });
}
