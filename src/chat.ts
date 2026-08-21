import {
  confirmPendingCalendarEvent,
  createConversation,
  discardPendingCalendarEvent,
  exportConversation,
  getMessages,
  listConversations,
  listPendingCalendarEvents,
  listToolRuns,
  resetConversation,
  sendMessage,
  type ChatMessage,
  type PendingCalendarEvent,
  type ToolRun,
} from "./sidecar";
import { speak, stopCurrentPlayback } from "./speech";
import { showError } from "./errorModal";
import { saveBlobAs } from "./fileSave";
import { setOrbAudioState } from "./orbStatus";
import { emitTurnSettled, isConversationModeOn, requestConversationStop } from "./conversationMode";
import { formatMessageMeta, initSpineNavigator, syncSpine } from "./spine";
import { getOwnerName } from "./settings";

let autoplayNextReply = false;

// The very first synthesis of a session also downloads the ~1.8GB voice
// model, so it's much slower than later ones -- warn the user the first
// time so a long wait doesn't look like a hang.
let firstPlayback = true;

/** Called by voice.ts right before it auto-submits a voice-originated
 * message, so the reply to that turn gets spoken automatically once ready. */
export function markNextReplyForAutoplay(): void {
  autoplayNextReply = true;
}

// Only one send can be in flight at a time (the input is disabled while
// waiting), so a single module-level controller is enough for voice.ts's
// Esc-to-cancel path to reach the current turn.
let currentTurnAbort: AbortController | null = null;

/** Aborts the in-flight send, if any. Returns whether there was one to
 * cancel. Used by the voice loop's Esc-to-cancel path while the orb is in
 * its "thinking" state. */
export function cancelCurrentTurn(): boolean {
  if (!currentTurnAbort) return false;
  currentTurnAbort.abort();
  return true;
}

function setTtsStatus(text: string | null): void {
  const status = document.querySelector<HTMLElement>("#tts-status");
  if (!status) return;
  if (text) {
    status.textContent = text;
    status.hidden = false;
  } else {
    status.hidden = true;
  }
}

// Calendar-day key ("Wed Jul 22 2026") for the day-marker logic below --
// chat-app style: the date prefix only shows on the first message of a new
// day, not on every single message.
function dayKey(iso: string): string {
  return new Date(iso).toDateString();
}

function renderMessage(
  role: ChatMessage["role"],
  content: string,
  createdAt: string,
  showDate = true,
): HTMLDivElement {
  const el = document.createElement("div");
  el.className = `message message--${role}`;
  // Read directly by spine.ts to build ticks/the topbar's turn label --
  // keeps the navigator in sync with whatever's actually in the DOM instead
  // of tracking a parallel data model.
  el.dataset.createdAt = createdAt;

  // The row (.message) is a full-width flex line that pushes the turn to its
  // side of the seam (user right, twin left); the inner column is capped at
  // one half so no bubble crosses the center seam. Meta sits ABOVE the bubble
  // (outside it) so it never lands as muted mono text on the violet fill.
  const inner = document.createElement("div");
  inner.className = "message-inner";

  const meta = document.createElement("span");
  meta.className = "message-meta";
  meta.textContent = formatMessageMeta(createdAt, showDate);
  inner.appendChild(meta);

  const text = document.createElement("span");
  text.className = "message-text";
  text.textContent = content;
  inner.appendChild(text);

  el.appendChild(inner);
  return el;
}

// Shown in the twin's reply slot from submit until the first token streams
// in -- otherwise that slot just sits empty for the 10-15s a local model
// can take to start generating, with no signal anything is happening.
function buildThinkingIndicator(): HTMLDivElement {
  const el = document.createElement("div");
  el.className = "message-thinking";

  const dot = document.createElement("span");
  dot.className = "message-thinking-dot";
  dot.setAttribute("aria-hidden", "true");
  for (let i = 0; i < 3; i++) {
    const ring = document.createElement("span");
    ring.className = "message-thinking-ring";
    dot.appendChild(ring);
  }

  const label = document.createElement("span");
  label.className = "message-thinking-label";
  label.textContent = "surfacing…";

  el.append(dot, label);
  return el;
}

const TOOL_LABELS: Record<string, string> = {
  get_current_datetime: "checked the time",
  get_weather: "checked the weather",
};

// When the twin uses a tool it's answering from live data rather than from
// what it remembers, and that difference should be visible -- same reasoning
// as the footer's internet-access list: the twin reaching outside itself is
// never silent. Appended under the reply once the turn settles.
function addToolChips(el: HTMLDivElement, runs: ToolRun[]): void {
  if (runs.length === 0) return;
  const inner = el.querySelector<HTMLDivElement>(".message-inner");
  if (!inner) return;

  const row = document.createElement("div");
  row.className = "message-tools";
  // Deduplicated: two weather lookups in one turn is one "checked the
  // weather" chip, not two identical ones.
  for (const name of [...new Set(runs.map((run) => run.tool_name))]) {
    const chip = document.createElement("span");
    chip.className = "message-tool-chip";
    chip.textContent = TOOL_LABELS[name] ?? name.replace(/_/g, " ");
    chip.title = runs
      .filter((run) => run.tool_name === name)
      .map((run) => run.result)
      .join("\n\n");
    row.appendChild(chip);
  }
  inner.appendChild(row);
}

// create_calendar_event drafts rather than writes when tools_calendar_confirm
// is on (the default) -- nothing reaches Google until Alex acts on one of
// these cards. Confirm and Discard both remove the row on success; the
// draft itself only disappears from the sidecar once one of them lands, so
// a failed request leaves the card exactly as it was, safe to retry.
function addPendingEventCards(el: HTMLDivElement, events: PendingCalendarEvent[]): void {
  if (events.length === 0) return;
  const inner = el.querySelector<HTMLDivElement>(".message-inner");
  if (!inner) return;

  for (const event of events) {
    const card = document.createElement("div");
    card.className = "pending-event-card";

    const title = document.createElement("div");
    title.className = "pending-event-title";
    title.textContent = event.title;

    const when = document.createElement("div");
    when.className = "pending-event-when";
    const start = new Date(event.start_at);
    const formattedStart = Number.isNaN(start.getTime())
      ? event.start_at
      : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(start);
    when.textContent = `${formattedStart} · ${event.duration_minutes} min`;

    const actions = document.createElement("div");
    actions.className = "pending-event-actions";

    const confirmButton = document.createElement("button");
    confirmButton.type = "button";
    confirmButton.textContent = "Add to calendar";

    const discardButton = document.createElement("button");
    discardButton.type = "button";
    discardButton.className = "btn-secondary";
    discardButton.textContent = "Discard";

    const runAction = (action: () => Promise<unknown>) => {
      void (async () => {
        confirmButton.disabled = true;
        discardButton.disabled = true;
        try {
          await action();
          card.remove();
        } catch (err) {
          showError("SYM-CALENDAR-EVENT-FAILED", { detail: (err as Error).message });
          confirmButton.disabled = false;
          discardButton.disabled = false;
        }
      })();
    };

    confirmButton.addEventListener("click", () => runAction(() => confirmPendingCalendarEvent(event.id)));
    discardButton.addEventListener("click", () => runAction(() => discardPendingCalendarEvent(event.id)));

    actions.append(confirmButton, discardButton);
    card.append(title, when, actions);
    if (event.description) {
      const description = document.createElement("div");
      description.className = "pending-event-description";
      description.textContent = event.description;
      card.insertBefore(description, actions);
    }
    inner.appendChild(card);
  }
}

function addSpeakButton(el: HTMLDivElement, getText: () => string): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "speak-button";
  button.title = "Play this reply";

  const dot = document.createElement("span");
  dot.className = "speak-dot";
  const label = document.createElement("span");
  label.className = "speak-label";
  label.textContent = `hear in ${getOwnerName()}'s voice`;
  button.append(dot, label);

  button.addEventListener("click", () => {
    void playReply(button, getText(), el);
  });
  // Sits under the bubble, inside the same capped column, so it stays on the
  // twin's side of the seam.
  (el.querySelector<HTMLElement>(".message-inner") ?? el).appendChild(button);
  return button;
}

async function playReply(button: HTMLButtonElement, text: string, messageEl?: HTMLElement): Promise<void> {
  if (!text.trim()) return;
  button.disabled = true;
  button.classList.add("speak-button--busy");

  const baseMessage = firstPlayback
    ? "Preparing your voice… the first time downloads the voice model (~1.8 GB), so this can take a minute."
    : "Generating audio in your voice…";

  // A ticking elapsed-time counter is the only reliable "is this actually
  // still working" signal we can give here -- there's no real download
  // progress available from the sidecar, but a number that keeps climbing
  // tells the user the UI hasn't frozen, and how long they've been waiting.
  // The orb reuses its "thinking" (indeterminate) visual for this same wait,
  // since there's no real byte-progress to back a determinate fill with --
  // once audio actually starts, `onStart` below switches it to "speaking".
  const startedAt = Date.now();
  let playbackStarted = false;
  const tick = () => {
    if (playbackStarted) return;
    const elapsed = Math.round((Date.now() - startedAt) / 1000);
    const label = elapsed >= 5 ? `${baseMessage} (${elapsed}s elapsed)` : baseMessage;
    setTtsStatus(label);
    setOrbAudioState("thinking", label);
  };
  tick();
  const ticker = setInterval(tick, 1000);

  try {
    await speak(text, {
      onStart: () => {
        playbackStarted = true;
        setTtsStatus(null);
        setOrbAudioState("speaking");
      },
    });
    firstPlayback = false;
  } catch (err) {
    console.error("playback failed", err);
    button.title = (err as Error).message;
    setTtsStatus(`Couldn't play that reply: ${(err as Error).message}`);
    setOrbAudioState(null);
    // Leave the error visible briefly, then clear it.
    setTimeout(() => setTtsStatus(null), 6000);
    return;
  } finally {
    clearInterval(ticker);
    button.disabled = false;
    button.classList.remove("speak-button--busy");
  }
  setOrbAudioState(null);
  // Bright tick on the spine for a turn that's actually been heard --
  // session-only (not persisted), so this resets on reload.
  if (messageEl) {
    messageEl.classList.add("message--voiced");
    syncSpine();
  }
}

// Chronological ordering (Mirror Spine): oldest at the top, newest at the
// bottom, chat-app style -- "caught up with the live conversation" means
// scrolled near the bottom edge.
const NEAR_BOTTOM_PX = 120;

function isNearBottom(el: HTMLElement): boolean {
  return el.scrollTop + el.clientHeight >= el.scrollHeight - NEAR_BOTTOM_PX;
}

export async function initChat(): Promise<void> {
  const panel = document.querySelector<HTMLElement>("#chat-panel");
  const list = document.querySelector<HTMLDivElement>("#message-list");
  const form = document.querySelector<HTMLFormElement>("#chat-form");
  const input = document.querySelector<HTMLInputElement>("#chat-input");
  const newMessageBadge = document.querySelector<HTMLButtonElement>("#new-message-badge");
  const newChatButton = document.querySelector<HTMLButtonElement>("#new-chat-button");
  const exportChatButton = document.querySelector<HTMLButtonElement>("#export-chat-button");
  if (!panel || !list || !form || !input) return;

  initSpineNavigator();

  if (newMessageBadge) {
    newMessageBadge.addEventListener("click", () => {
      list.scrollTop = list.scrollHeight;
      newMessageBadge.hidden = true;
    });
    list.addEventListener("scroll", () => {
      if (isNearBottom(list)) newMessageBadge.hidden = true;
    });
  }

  const conversations = await listConversations();
  const conversation = conversations[0] ?? (await createConversation());
  let conversationId = conversation.id;

  // Chronological order, chat-app style: the backend already returns oldest
  // -> newest (see db.get_messages's ORDER BY id ASC), so a plain forward
  // append is all that's needed. Track the running day so only the first
  // message of a new calendar day shows its date prefix (see dayKey/
  // renderMessage's showDate).
  const history = await getMessages(conversationId);
  list.innerHTML = "";
  let lastDayKey: string | null = null;
  for (const entry of history) {
    if (entry.role === "system") continue;
    const day = dayKey(entry.created_at);
    const showDate = day !== lastDayKey;
    lastDayKey = day;
    const el = renderMessage(entry.role, entry.content, entry.created_at, showDate);
    if (entry.role === "assistant") addSpeakButton(el, () => entry.content);
    list.appendChild(el);
  }
  list.scrollTop = list.scrollHeight;
  syncSpine();

  panel.hidden = false;
  input.disabled = false;
  input.focus();

  if (newChatButton) {
    newChatButton.addEventListener("click", () => {
      void (async () => {
        // Abort any in-flight send/reply against the conversation being left
        // behind -- it still writes to its own (now-archived) row, it just
        // won't affect the fresh one we're about to switch to.
        cancelCurrentTurn();
        stopCurrentPlayback();
        setOrbAudioState(null);
        // Don't keep the mic hot on a fresh chat -- tear the hands-free loop
        // down too (routed through conversationMode.ts to avoid importing
        // voice.ts here, which would be circular).
        requestConversationStop();

        const fresh = await resetConversation();
        conversationId = fresh.id;
        list.innerHTML = "";
        lastDayKey = null;
        syncSpine();
        input.value = "";
        if (newMessageBadge) newMessageBadge.hidden = true;
        input.focus();
      })();
    });
  }

  if (exportChatButton) {
    // conversationId is this closure's own `let`, reassigned by "New chat"
    // above -- wiring the listener here (rather than exporting a getter just
    // for this) always reads whichever conversation is actually current.
    exportChatButton.addEventListener("click", () => {
      void (async () => {
        exportChatButton.disabled = true;
        exportChatButton.classList.add("btn-loading");
        try {
          const blob = await exportConversation(conversationId);
          const stamp = new Date()
            .toISOString()
            .replace(/[-:]/g, "")
            .replace(/\..+/, "")
            .replace("T", "-");
          await saveBlobAs(blob, `mimoid-chat-${stamp}.json`, ["json"]);
        } catch (err) {
          showError("SYM-CHAT-EXPORT-FAILED", { detail: (err as Error).message });
        } finally {
          exportChatButton.disabled = false;
          exportChatButton.classList.remove("btn-loading");
        }
      })();
    });
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const content = input.value.trim();
    if (!content) return;

    // Autoplay a voice-originated turn (the flag markNextReplyForAutoplay set)
    // OR any turn while hands-free conversation mode is on -- there, even a
    // typed message is spoken back and keeps the loop going.
    const shouldAutoplay = autoplayNextReply || isConversationModeOn();
    autoplayNextReply = false;
    // "thinking" starts for any spoken turn -- shouldAutoplay doubles as "this
    // reply will be read aloud" regardless of whether the turn came from a
    // spoken utterance or a typed message in conversation mode.
    if (shouldAutoplay) setOrbAudioState("thinking");

    input.value = "";
    input.disabled = true;

    // Chronological order: the new turn is appended at the END of the list,
    // below everything older -- question then its answer, reading top to
    // bottom same as the rest of the history.
    const now = new Date().toISOString();
    const today = dayKey(now);
    const userShowDate = today !== lastDayKey;
    lastDayKey = today;
    const userEl = renderMessage("user", content, now, userShowDate);
    list.append(userEl);
    const assistantEl = renderMessage("assistant", "", now, false); // same turn, same day as the question just above it
    const assistantText = assistantEl.querySelector<HTMLSpanElement>(".message-text");
    const thinkingIndicator = buildThinkingIndicator();
    assistantText?.before(thinkingIndicator);
    list.append(assistantEl);
    list.scrollTop = list.scrollHeight;
    syncSpine();
    if (newMessageBadge) newMessageBadge.hidden = true;

    const controller = new AbortController();
    currentTurnAbort = controller;
    let thinkingCleared = false;
    const clearThinking = () => {
      if (!thinkingCleared) {
        thinkingCleared = true;
        setOrbAudioState(null);
        thinkingIndicator.remove();
      }
    };

    // Emitted exactly once per turn, after the reply AND any spoken audio have
    // fully settled -- this is the cue voice.ts's hands-free loop resumes on
    // (see conversationMode.ts for why the orbStatus null transition can't be
    // used for this).
    let settled = false;
    const settle = (outcome: Parameters<typeof emitTurnSettled>[0]) => {
      if (settled) return;
      settled = true;
      emitTurnSettled(outcome);
    };

    void (async () => {
      try {
        const { turnStartedAt } = await sendMessage(
          conversationId,
          content,
          (token) => {
            clearThinking(); // the reply is streaming in now, "thinking" is over
            if (assistantText) assistantText.textContent += token;
            syncSpine();
            // Only snap to the bottom if the user is already reading near it
            // -- if they've scrolled up into history, hold position and let
            // them opt back in via the "new message" affordance instead.
            if (isNearBottom(list)) {
              list.scrollTop = list.scrollHeight;
            } else if (newMessageBadge) {
              newMessageBadge.hidden = false;
            }
          },
          controller.signal,
          (memoryCount, historyCount) => {
            // Names what's actually happening during the wait, if there's
            // anything to name -- otherwise the orb's generic "Thinking it
            // through…" default stands.
            if (thinkingCleared) return;
            const parts: string[] = [];
            if (memoryCount > 0) parts.push(`${memoryCount} ${memoryCount === 1 ? "memory" : "memories"}`);
            if (historyCount > 0) parts.push(`${historyCount} history ${historyCount === 1 ? "note" : "notes"}`);
            if (parts.length > 0) {
              setOrbAudioState("thinking", `Pulling ${parts.join(" + ")}…`);
            }
          },
        );
        const fullText = assistantText?.textContent ?? "";
        // Capture this before the speak button grows the assistant bubble --
        // otherwise adding it always fails isNearBottom by definition (it
        // just pushed the true bottom further down), even though the user
        // was following along a moment ago.
        const wasNearBottom = isNearBottom(list);
        const button = addSpeakButton(assistantEl, () => fullText);
        // Best-effort: the reply is already on screen and correct, so a
        // failure to fetch the "which tools ran" detail costs a chip, not
        // the turn. `now` scopes this to the turn just finished.
        try {
          addToolChips(assistantEl, await listToolRuns(conversationId, turnStartedAt ?? undefined));
        } catch {
          /* chips are decoration; never fail a delivered reply over them */
        }
        try {
          addPendingEventCards(
            assistantEl,
            await listPendingCalendarEvents(conversationId, turnStartedAt ?? undefined),
          );
        } catch {
          /* same best-effort reasoning as the tool chips above -- the reply
             text is already correct and on screen either way */
        }
        if (wasNearBottom) list.scrollTop = list.scrollHeight;
        syncSpine();
        if (shouldAutoplay) {
          // Settle only once the audio actually finishes -- playReply swallows
          // its own errors and its promise still resolves, so a TTS failure
          // still counts as "spoken" and the loop keeps going (silently).
          void playReply(button, fullText, assistantEl).finally(() => settle("spoken"));
        } else {
          settle("reply-only");
        }
      } catch (err) {
        if (controller.signal.aborted) {
          // Cancelled via the orb's Esc path -- discard the turn, no error.
          assistantEl.remove();
          syncSpine();
          settle("cancelled");
        } else {
          // Remove the failed turn entirely rather than leaving an empty
          // reply bubble behind -- the error modal's copy reassures the
          // user their message wasn't lost, and Retry resends the exact
          // same content as a fresh turn.
          userEl.remove();
          assistantEl.remove();
          syncSpine();
          showError("SYM-CHAT-SEND-FAILED", {
            detail: (err as Error).message,
            onPrimary: () => {
              input.value = content;
              form.requestSubmit();
            },
          });
          settle("error");
        }
      } finally {
        clearThinking();
        if (currentTurnAbort === controller) currentTurnAbort = null;
        input.disabled = false;
        input.focus();
      }
    })();
  });
}
