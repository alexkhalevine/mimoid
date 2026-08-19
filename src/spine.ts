// Mirror Spine: the chat-history navigator overlaying #message-list -- a
// central seam, per-message ticks (user=right/violet, twin=left/teal), and a
// draggable "pearl" that scrubs the transcript. Reads message elements
// directly off the live DOM (role via .message--user/.message--assistant,
// timestamp via a data-created-at attribute chat.ts sets) rather than
// tracking a parallel data model, so it can't drift out of sync with what's
// actually rendered.
//
// v1 scope (deferred: tick density clustering past ~200 turns, full ARIA
// slider semantics/live aria-valuenow/Home-End): the seam always renders;
// ticks + pearl only appear once there's enough history to be worth
// scrubbing.

// Below this many messages (~6 turns), the seam alone is enough -- ticks and
// the pearl would be visual noise for a handful of exchanges.
const SPINE_MIN_MESSAGES = 12;

let listEl: HTMLElement | null = null;
let ticksEl: HTMLElement | null = null;
let pearlEl: HTMLElement | null = null;
let hitColumnEl: HTMLButtonElement | null = null;
let topbarCountEl: HTMLElement | null = null;
let topbarLatestButton: HTMLButtonElement | null = null;

let dragging = false;
let syncScheduled = false;

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** "Jul 21 · 3:42 PM" (or just "3:42 PM" when showDate is false) -- used both
 * for the per-message meta line (chat.ts) and the topbar's current-turn
 * label. chat.ts suppresses the date on every message except the first of a
 * new calendar day, chat-app style; the topbar always passes showDate=true
 * since it should keep labelling the date regardless of what that message's
 * own bubble is showing. Locale-default, no new dependency. */
export function formatMessageMeta(iso: string, showDate = true): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const timePart = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(date);
  if (!showDate) return timePart;
  const datePart = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(date);
  return `${datePart} · ${timePart}`;
}

function messages(): HTMLElement[] {
  return listEl ? Array.from(listEl.querySelectorAll<HTMLElement>(".message")) : [];
}

function jumpTo(message: HTMLElement): void {
  message.scrollIntoView({ block: "center", behavior: prefersReducedMotion() ? "auto" : "smooth" });
}

function renderTicks(): void {
  if (!ticksEl) return;
  const msgs = messages();
  const sparse = msgs.length < SPINE_MIN_MESSAGES;

  ticksEl.hidden = sparse;
  if (pearlEl) pearlEl.hidden = sparse;
  if (hitColumnEl) hitColumnEl.hidden = sparse;
  if (sparse) {
    ticksEl.innerHTML = "";
    return;
  }

  ticksEl.innerHTML = "";
  const n = msgs.length;
  msgs.forEach((message, index) => {
    const isUser = message.classList.contains("message--user");
    const tick = document.createElement("button");
    tick.type = "button";
    tick.className = `spine-tick spine-tick--${isUser ? "user" : "assistant"}`;
    if (message.classList.contains("message--voiced")) tick.classList.add("spine-tick--voiced");
    tick.style.top = `${(index / Math.max(1, n - 1)) * 100}%`;
    tick.setAttribute("aria-label", `Jump to ${isUser ? "your" : "the twin's"} message`);
    tick.addEventListener("click", () => jumpTo(message));
    ticksEl!.appendChild(tick);
  });
}

function updatePearl(): void {
  if (!listEl || !pearlEl || pearlEl.hidden) return;
  const max = listEl.scrollHeight - listEl.clientHeight;
  const fraction = max > 0 ? listEl.scrollTop / max : 0;
  pearlEl.style.top = `${Math.min(1, Math.max(0, fraction)) * 100}%`;
}

/** Finds the message whose top edge is closest to the viewport's top edge --
 * used both to label "Turn N / total" and as the anchor for arrow-key
 * stepping, so both agree on "what's currently in view." */
function nearestMessageIndex(msgs: HTMLElement[]): number {
  if (!listEl || msgs.length === 0) return -1;
  const listTop = listEl.getBoundingClientRect().top;
  let nearestIndex = 0;
  let nearestDistance = Infinity;
  msgs.forEach((message, index) => {
    const distance = Math.abs(message.getBoundingClientRect().top - listTop);
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestIndex = index;
    }
  });
  return nearestIndex;
}

function updateTopbar(): void {
  if (!topbarCountEl) return;
  const msgs = messages();
  if (msgs.length === 0) {
    topbarCountEl.textContent = "No messages yet";
    return;
  }
  const index = nearestMessageIndex(msgs);
  const current = msgs[index];
  const createdAt = current?.dataset.createdAt;
  // Always includes the date here regardless of whether this message's own
  // bubble is suppressing it (same-day messages hide their date prefix) --
  // the topbar's job is to always orient you to a date.
  const meta = createdAt ? formatMessageMeta(createdAt, true) : "";
  const dateOnly = meta.split(" · ")[0];
  topbarCountEl.textContent = `Turn ${index + 1} / ${msgs.length}${dateOnly ? ` · ${dateOnly}` : ""}`;
}

function scheduleSync(): void {
  if (syncScheduled) return;
  syncScheduled = true;
  requestAnimationFrame(() => {
    syncScheduled = false;
    renderTicks();
    updatePearl();
    updateTopbar();
  });
}

/** Call after any change to #message-list's children (a new turn prepended,
 * a streamed token, history loaded, a new chat clearing it, a message marked
 * voiced) -- rAF-batched internally, so it's cheap to call on every token. */
export function syncSpine(): void {
  scheduleSync();
}

function railTop(): number | null {
  if (!hitColumnEl) return null;
  return hitColumnEl.getBoundingClientRect().top;
}

function railHeight(): number | null {
  if (!hitColumnEl) return null;
  return hitColumnEl.getBoundingClientRect().height;
}

function scrollFromClientY(clientY: number): void {
  if (!listEl) return;
  const top = railTop();
  const height = railHeight();
  if (top === null || height === null || height <= 0) return;
  const fraction = Math.min(1, Math.max(0, (clientY - top) / height));
  const max = listEl.scrollHeight - listEl.clientHeight;
  listEl.scrollTop = fraction * max;
}

function onPointerDown(event: PointerEvent): void {
  dragging = true;
  hitColumnEl?.setPointerCapture(event.pointerId);
  pearlEl?.classList.add("spine-pearl--dragging");
  scrollFromClientY(event.clientY);
}

function onPointerMove(event: PointerEvent): void {
  if (!dragging) return;
  scrollFromClientY(event.clientY);
}

function onPointerUp(event: PointerEvent): void {
  dragging = false;
  hitColumnEl?.releasePointerCapture(event.pointerId);
  pearlEl?.classList.remove("spine-pearl--dragging");
}

function stepTick(direction: 1 | -1): void {
  const msgs = messages();
  if (msgs.length === 0) return;
  const current = nearestMessageIndex(msgs);
  const target = msgs[Math.min(msgs.length - 1, Math.max(0, current + direction))];
  if (target) jumpTo(target);
}

function onKeydown(event: KeyboardEvent): void {
  if (event.key === "ArrowDown") {
    event.preventDefault();
    stepTick(1);
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    stepTick(-1);
  }
}

/** Wires the seam/ticks/pearl overlay to #message-list. Safe to call once;
 * chat.ts calls syncSpine() afterward whenever the message list changes. */
export function initSpineNavigator(): void {
  listEl = document.querySelector<HTMLElement>("#message-list");
  ticksEl = document.querySelector<HTMLElement>("#spine-ticks");
  pearlEl = document.querySelector<HTMLElement>("#spine-pearl");
  hitColumnEl = document.querySelector<HTMLButtonElement>("#spine-hit-column");
  topbarCountEl = document.querySelector<HTMLElement>("#spine-topbar-count");
  topbarLatestButton = document.querySelector<HTMLButtonElement>("#spine-topbar-latest");
  if (!listEl) return;

  listEl.addEventListener("scroll", () => scheduleSync());

  if (hitColumnEl) {
    hitColumnEl.addEventListener("pointerdown", onPointerDown);
    hitColumnEl.addEventListener("pointermove", onPointerMove);
    hitColumnEl.addEventListener("pointerup", onPointerUp);
    hitColumnEl.addEventListener("pointercancel", onPointerUp);
    hitColumnEl.addEventListener("keydown", onKeydown);
  }

  topbarLatestButton?.addEventListener("click", () => {
    if (listEl) listEl.scrollTop = listEl.scrollHeight;
  });

  scheduleSync();
}
