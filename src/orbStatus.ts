// A small cross-module signal so chat.ts and speech.ts can report "the twin
// is thinking/speaking" without importing voice.ts directly (voice.ts already
// imports from chat.ts, so the reverse import would be circular). voice.ts is
// the only subscriber -- it translates these into the orb's data-state.

export type OrbAudioState = "thinking" | "speaking";

type Listener = (state: OrbAudioState | null, subCaption?: string) => void;

let listener: Listener | null = null;

export function onOrbAudioState(cb: Listener): void {
  listener = cb;
}

/** Reports a new orb audio state, or `null` to clear back to idle. Only has
 * an effect if voice.ts is listening (e.g. mic unsupported => no-op). */
export function setOrbAudioState(state: OrbAudioState | null, subCaption?: string): void {
  listener?.(state, subCaption);
}
