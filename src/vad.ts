// Voice-activity / silence detection for the hands-free conversation loop.
// Deliberately DOM-free and driven entirely by (level, now) samples the
// caller feeds it each animation frame, so it's trivially unit-testable and
// reuses the RMS `level` voice.ts already computes for its level meter -- no
// second AudioContext/AnalyserNode.
//
// All timing uses caller-supplied timestamps (performance.now()) rather than
// frame counts, because requestAnimationFrame throttles hard when the window
// is backgrounded -- counting frames would stretch "2 seconds" to whatever
// the throttled frame rate happens to be.

// Above this RMS level = voice present; below SILENCE_THRESHOLD = silence.
// The gap between them is hysteresis so a level hovering near one threshold
// doesn't flap the detector on and off.
export const ONSET_THRESHOLD = 0.05;
export const SILENCE_THRESHOLD = 0.03;

// Onset must be sustained this long before it counts, so a single click/pop
// doesn't register as the user starting to speak.
export const ONSET_DEBOUNCE_MS = 150;

// How long the user must stay quiet, AFTER having spoken, before we treat the
// utterance as finished. This is the "~2 seconds of silence" from the spec, and
// the default for the user-facing "pause before replying" setting -- callers can
// override it per detector via SilenceDetectorOptions.silenceMs.
export const SILENCE_MS = 2000;

// If the user never speaks at all, don't hold the mic open forever.
export const INITIAL_SILENCE_TIMEOUT = 20000;

// Even if the user never pauses, cap a single utterance so a stuck-loud mic
// can't record indefinitely.
export const MAX_UTTERANCE_MS = 30000;

export interface SilenceDetectorCallbacks {
  /** Fires once, when confirmed (debounced) speech onset is detected. */
  onSpeechStart?: () => void;
  /** Fires once: either SILENCE_MS of quiet after speech, or the utterance
   * cap was hit. The caller should stop the recording. */
  onSilence?: () => void;
  /** Fires once if no speech is ever detected within INITIAL_SILENCE_TIMEOUT.
   * The caller should treat this as an empty turn. */
  onInitialTimeout?: () => void;
}

export interface SilenceDetectorOptions {
  /** Trailing silence that ends the utterance. Defaults to SILENCE_MS. */
  silenceMs?: number;
}

export interface SilenceDetector {
  /** Feed one sample. `level` is RMS ~0..1; `now` is a performance.now() ms. */
  feed: (level: number, now: number) => void;
  /** Reset all internal state so the detector can be reused for a new turn. */
  reset: () => void;
}

export function createSilenceDetector(
  callbacks: SilenceDetectorCallbacks,
  options: SilenceDetectorOptions = {},
): SilenceDetector {
  const silenceMs = options.silenceMs ?? SILENCE_MS;
  let startedAt: number | null = null; // first sample of the current turn
  let hasSpoken = false;
  let speechConfirmedAt: number | null = null; // when speech first crossed onset
  let onsetSince: number | null = null; // start of the current above-onset run
  let lastLoudAt = 0; // last time level was >= SILENCE_THRESHOLD
  let done = false; // onSilence/onInitialTimeout already fired this turn

  const reset = () => {
    startedAt = null;
    hasSpoken = false;
    speechConfirmedAt = null;
    onsetSince = null;
    lastLoudAt = 0;
    done = false;
  };

  const feed = (level: number, now: number) => {
    if (done) return;
    if (startedAt === null) startedAt = now;

    if (!hasSpoken) {
      // Still waiting for the user to start talking.
      if (level > ONSET_THRESHOLD) {
        if (onsetSince === null) onsetSince = now;
        if (now - onsetSince >= ONSET_DEBOUNCE_MS) {
          hasSpoken = true;
          speechConfirmedAt = now;
          lastLoudAt = now;
          callbacks.onSpeechStart?.();
        }
      } else {
        onsetSince = null; // the run was too short -- reset it
        if (now - startedAt >= INITIAL_SILENCE_TIMEOUT) {
          done = true;
          callbacks.onInitialTimeout?.();
        }
      }
      return;
    }

    // The user has spoken: watch for a trailing silence, or the hard cap.
    if (level >= SILENCE_THRESHOLD) lastLoudAt = now;

    const silentLongEnough = now - lastLoudAt >= silenceMs;
    const utteranceTooLong = speechConfirmedAt !== null && now - speechConfirmedAt >= MAX_UTTERANCE_MS;
    if (silentLongEnough || utteranceTooLong) {
      done = true;
      callbacks.onSilence?.();
    }
  };

  return { feed, reset };
}
