export function mediaRecorderSupported(): boolean {
  return Boolean(navigator.mediaDevices?.getUserMedia) && typeof MediaRecorder !== "undefined";
}

export interface RecordingSession {
  stop: () => void;
  /** Stops capture and releases the mic WITHOUT invoking `onStopped` -- used
   * by the voice loop's tap-again/Esc "cancel" path, which discards the
   * in-flight recording instead of transcribing it. */
  cancel: () => void;
  /** The live input stream, exposed so callers can drive a level meter
   * (e.g. via an AnalyserNode) without this module knowing about rendering. */
  stream: MediaStream;
}

// If the OS never answers the microphone-permission request, getUserMedia
// hangs forever and the button looks dead. On macOS the usual cause is a
// missing entitlement / the app not being allowed in Privacy settings.
// This watchdog turns that silent hang into an actionable message.
const MIC_REQUEST_TIMEOUT_MS = 20000;

function requestMicWithTimeout(): Promise<MediaStream> {
  const request = navigator.mediaDevices.getUserMedia({ audio: true });
  let timer: ReturnType<typeof setTimeout>;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      const err = new Error("microphone request timed out");
      err.name = "MicTimeout";
      reject(err);
      // If the request resolves later (e.g. the user finally answers a
      // stuck prompt), release the device so it isn't left open.
      void request.then((stream) => stream.getTracks().forEach((track) => track.stop())).catch(() => {});
    }, MIC_REQUEST_TIMEOUT_MS);
  });
  return Promise.race([request, timeout]).finally(() => clearTimeout(timer)) as Promise<MediaStream>;
}

function micErrorMessage(err: Error): string {
  switch (err.name) {
    case "MicTimeout":
      return (
        "The microphone didn't respond. On macOS, open System Settings → Privacy & Security → " +
        "Microphone and make sure Mimoid is turned on. On Linux, check that PipeWire (or " +
        "PulseAudio) is running and that an input device is selected. Then try again."
      );
    case "NotAllowedError":
    case "SecurityError":
      return (
        "Microphone access is blocked. On macOS, allow it in System Settings → Privacy & Security → " +
        "Microphone. On Linux, check your desktop's privacy/permission settings for the app. " +
        "Then try again."
      );
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "No microphone was found. Connect or enable one, then try again.";
    default:
      return `Couldn't access the microphone: ${err.message || err.name || "unknown error"}`;
  }
}

/**
 * Starts capturing from the microphone. Resolves once recording has begun;
 * `onStopped` fires with the assembled clip once `session.stop()` is called.
 * Resolves to null (and reports via `onError`) if mic access fails.
 */
export async function startRecording(
  onStopped: (blob: Blob) => void,
  onError: (message: string) => void,
): Promise<RecordingSession | null> {
  if (!navigator.mediaDevices?.getUserMedia) {
    onError("Microphone capture isn't available in this window.");
    return null;
  }

  let stream: MediaStream;
  try {
    stream = await requestMicWithTimeout();
  } catch (err) {
    console.error("[voice] microphone request failed", err);
    onError(micErrorMessage(err as Error));
    return null;
  }

  const chunks: Blob[] = [];
  let mediaRecorder: MediaRecorder;
  try {
    mediaRecorder = new MediaRecorder(stream);
  } catch (err) {
    stream.getTracks().forEach((track) => track.stop());
    console.error("[voice] MediaRecorder init failed", err);
    onError(`Couldn't start recording: ${(err as Error).message || "unsupported audio format"}`);
    return null;
  }

  let cancelled = false;
  mediaRecorder.addEventListener("dataavailable", (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  });
  mediaRecorder.addEventListener("stop", () => {
    stream.getTracks().forEach((track) => track.stop());
    if (!cancelled) onStopped(new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" }));
  });

  mediaRecorder.start();
  return {
    stop: () => mediaRecorder.stop(),
    cancel: () => {
      cancelled = true;
      mediaRecorder.stop();
    },
    stream,
  };
}

export function extensionFor(blobType: string): string {
  if (blobType.includes("mp4")) return "mp4";
  if (blobType.includes("ogg")) return "ogg";
  return "webm";
}
