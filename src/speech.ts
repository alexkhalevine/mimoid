import { SIDECAR_BASE_URL } from "./config";

// Generous safety net above the sidecar's own ~10-minute timeout (see
// TTS_TIMEOUT_SECONDS in main.py) for a first-time voice-model download on a
// slow connection -- the sidecar's clearer, more specific error should
// normally fire first; this just guarantees the request never hangs forever.
const SPEAK_TIMEOUT_MS = 11 * 60 * 1000;

// Lets stopCurrentPlayback() reach whatever audio is currently playing --
// only one speak() call is ever in flight at a time in practice.
let cancelPlayback: (() => void) | null = null;

/** Stops whatever reply is currently playing (used by the voice loop's
 * Esc-to-cancel path when the orb is in its "speaking" state). No-op if
 * nothing is playing. */
export function stopCurrentPlayback(): void {
  cancelPlayback?.();
}

/** Synthesizes `text` in the cloned voice and plays it back. Resolves once
 * playback finishes, is cancelled via `stopCurrentPlayback()`, or rejects on
 * a synthesis/playback error. `onStart` fires once audio actually starts
 * playing (not merely once `.play()` is called). */
export async function speak(text: string, opts?: { onStart?: () => void }): Promise<void> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), SPEAK_TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch(`${SIDECAR_BASE_URL}/speak`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: controller.signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      throw new Error("Speech synthesis is taking too long -- check your network connection and try again.");
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    let detail = "failed to synthesize speech";
    try {
      detail = ((await res.json()) as { detail?: string }).detail ?? detail;
    } catch {
      // ignore, fall back to the generic message
    }
    throw new Error(detail);
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  try {
    const audio = new Audio(url);
    await new Promise<void>((resolve, reject) => {
      cancelPlayback = () => {
        audio.pause();
        resolve(); // a user-initiated cancel is a clean stop, not an error
      };
      audio.addEventListener("playing", () => opts?.onStart?.(), { once: true });
      audio.addEventListener("ended", () => resolve());
      audio.addEventListener("error", () => reject(new Error("audio playback failed")));
      audio.play().catch(reject);
    });
  } finally {
    cancelPlayback = null;
    URL.revokeObjectURL(url);
  }
}
