// Shared state + signals for the Talk tab's hands-free "conversation mode"
// (the continuous listen -> reply-aloud -> resume-listening loop). Kept as a
// leaf module (no imports of voice.ts/chat.ts) so both can use it without a
// circular import -- voice.ts already imports FROM chat.ts, so anything
// chat.ts needs from the voice loop has to go through a seam like this.

// --- On/off flag -----------------------------------------------------------
// voice.ts (the toggle owner) writes this; chat.ts reads it so a *typed* send
// while the mode is on is also spoken back and keeps the loop going.
let conversationOn = false;

export function setConversationMode(on: boolean): void {
  conversationOn = on;
}

export function isConversationModeOn(): boolean {
  return conversationOn;
}

// --- "Turn settled" signal --------------------------------------------------
// chat.ts emits exactly once per turn, after the reply AND any spoken audio
// have fully finished. This is the clean resume cue voice.ts needs:
// setOrbAudioState(null) can't be used because chat.ts flips the orb to null
// mid-turn (when the first token streams in) before playReply re-raises it.
//
// - "spoken":     reply generated and its audio finished playing
// - "reply-only": reply generated, no audio was played (autoplay off)
// - "error":      generation or a hard failure -- pause the loop
// - "cancelled":  the turn was aborted (Esc / new chat) -- don't resume
export type TurnOutcome = "spoken" | "reply-only" | "error" | "cancelled";

let turnListener: ((outcome: TurnOutcome) => void) | null = null;

export function onTurnSettled(cb: (outcome: TurnOutcome) => void): void {
  turnListener = cb;
}

export function emitTurnSettled(outcome: TurnOutcome): void {
  turnListener?.(outcome);
}

// --- "Stop the loop" request ------------------------------------------------
// Lets chat.ts's new-chat handler tear the loop down without importing
// voice.ts. voice.ts subscribes and runs its teardown.
let stopListener: (() => void) | null = null;

export function onConversationStop(cb: () => void): void {
  stopListener = cb;
}

export function requestConversationStop(): void {
  stopListener?.();
}
