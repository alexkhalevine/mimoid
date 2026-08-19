// The global error modal ("A Disturbance") -- a single shared component
// mounted once at the app root (index.html) so Talk, Train, and System can
// all trigger it identically via showError().
import { getErrorDefinition, type ErrorCode } from "./errors";

export interface ShowErrorOptions {
  /** Raw technical detail (an exception message, an HTTP status, a stack
   * trace) for a coder or an AI agent working on a fix later -- shown only
   * inside the collapsed "Details" disclosure, never in the poetic copy
   * itself, and always logged to the console regardless of whether the
   * user ever expands it. */
  detail?: string;
  /** Invoked when the user clicks the primary action button; the modal
   * closes immediately afterward -- the action's own result surfaces its
   * own feedback (a new turn, an updated list, ...), not a second modal. */
  onPrimary?: () => void;
}

interface Elements {
  backdrop: HTMLElement;
  title: HTMLElement;
  body: HTMLElement;
  dismiss: HTMLButtonElement;
  primary: HTMLButtonElement;
  code: HTMLElement;
  detailsToggle: HTMLButtonElement;
  technical: HTMLElement;
}

let els: Elements | null = null;
let isOpen = false;
let lastFocused: HTMLElement | null = null;

function getElements(): Elements | null {
  if (els) return els;
  const backdrop = document.querySelector<HTMLElement>("#error-modal-backdrop");
  const title = document.querySelector<HTMLElement>("#error-modal-title");
  const body = document.querySelector<HTMLElement>("#error-modal-body");
  const dismiss = document.querySelector<HTMLButtonElement>("#error-modal-dismiss");
  const primary = document.querySelector<HTMLButtonElement>("#error-modal-primary");
  const code = document.querySelector<HTMLElement>("#error-modal-code");
  const detailsToggle = document.querySelector<HTMLButtonElement>("#error-modal-details-toggle");
  const technical = document.querySelector<HTMLElement>("#error-modal-technical");
  if (!backdrop || !title || !body || !dismiss || !primary || !code || !detailsToggle || !technical) {
    return null;
  }
  els = { backdrop, title, body, dismiss, primary, code, detailsToggle, technical };
  return els;
}

function close(): void {
  const e = getElements();
  if (!e || !isOpen) return;
  isOpen = false;
  e.backdrop.hidden = true;
  document.removeEventListener("keydown", onKeydown);
  const toFocus = lastFocused;
  lastFocused = null;
  toFocus?.focus();
}

function focusableActions(e: Elements): HTMLButtonElement[] {
  return [e.dismiss, e.primary, e.detailsToggle].filter((el) => !el.hidden);
}

function onKeydown(event: KeyboardEvent): void {
  const e = getElements();
  if (!e) return;
  if (event.key === "Escape") {
    event.preventDefault();
    close();
    return;
  }
  if (event.key !== "Tab") return;
  // A minimal focus trap across just the card's own buttons -- there's
  // nothing else focusable inside the modal.
  const focusable = focusableActions(e);
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
}

/** Shows (or, if one is already open, replaces in place) the global error
 * modal for `code`. Safe to call from anywhere -- Talk, Train, System all
 * share this one mount point. Always logs the code + detail to the console,
 * regardless of whether the user ever expands the in-modal Details. */
export function showError(code: ErrorCode, options: ShowErrorOptions = {}): void {
  const e = getElements();
  if (!e) return;
  const def = getErrorDefinition(code);

  console.error(`[${def.code}] ${def.title}${options.detail ? ` -- ${options.detail}` : ""}`);

  e.title.textContent = def.title;
  e.body.textContent = def.body;
  e.code.textContent = def.code;

  if (options.detail) {
    e.detailsToggle.hidden = false;
    e.detailsToggle.setAttribute("aria-expanded", "false");
    e.technical.hidden = true;
    e.technical.textContent = options.detail;
  } else {
    e.detailsToggle.hidden = true;
    e.technical.hidden = true;
    e.technical.textContent = "";
  }

  if (def.primaryLabel && options.onPrimary) {
    const onPrimary = options.onPrimary;
    e.primary.hidden = false;
    e.primary.textContent = def.primaryLabel;
    e.primary.onclick = () => {
      close();
      onPrimary();
    };
  } else {
    e.primary.hidden = true;
    e.primary.onclick = null;
  }

  if (!isOpen) {
    isOpen = true;
    lastFocused = document.activeElement as HTMLElement | null;
    e.backdrop.hidden = false;
    document.addEventListener("keydown", onKeydown);
  }
  e.dismiss.focus();
}

/** Wires the modal's own interactions (backdrop click, Dismiss, Details
 * toggle). Call once at startup; showError() works from anywhere after
 * that. */
export function initErrorModal(): void {
  const e = getElements();
  if (!e) return;

  // No extra reduced-motion handling needed here -- styles.css's blanket
  // prefers-reduced-motion rule already zeroes every animation-duration,
  // which covers the bloom/ring/popin/fadein loops the same way.

  e.backdrop.addEventListener("click", (event) => {
    if (event.target === e.backdrop) close();
  });
  e.dismiss.addEventListener("click", close);
  e.detailsToggle.addEventListener("click", () => {
    const expanded = e.detailsToggle.getAttribute("aria-expanded") === "true";
    e.detailsToggle.setAttribute("aria-expanded", String(!expanded));
    e.technical.hidden = expanded;
  });
}
