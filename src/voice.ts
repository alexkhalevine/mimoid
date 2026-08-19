import { SIDECAR_BASE_URL } from "./config";
import { extensionFor, mediaRecorderSupported, startRecording, type RecordingSession } from "./recorder";
import { markNextReplyForAutoplay, cancelCurrentTurn } from "./chat";
import { stopCurrentPlayback } from "./speech";
import { onOrbAudioState } from "./orbStatus";
import {
  isConversationModeOn,
  onConversationStop,
  onTurnSettled,
  setConversationMode,
} from "./conversationMode";
import { createSilenceDetector, type SilenceDetector } from "./vad";
import { getCapabilities } from "./sidecar";
import { getOwnerName, getPauseMs } from "./settings";

type RecorderState = "idle" | "recording" | "transcribing" | "thinking" | "speaking";

// Generous safety net above the sidecar's own ~8-minute timeout (see
// STT_TIMEOUT_SECONDS in main.py) for a first-time speech-model download on
// a slow connection -- the sidecar's clearer, more specific error should
// normally fire first; this just guarantees the button never stays stuck.
const TRANSCRIBE_TIMEOUT_MS = 9 * 60 * 1000;

// In conversation mode, after this many listening turns in a row catch no
// speech, pause the loop rather than holding the mic open forever in an
// empty room.
const MAX_CONSECUTIVE_EMPTIES = 3;

// Lets cancelActiveTurn() abort an in-flight transcription from the "Esc to
// cancel" path; distinguishes a user-cancel (silent) from a real timeout
// (shown as an error) since both surface as the same AbortError.
let transcribeAbortController: AbortController | null = null;
let userCancelledTranscribe = false;

async function transcribe(blob: Blob): Promise<string> {
  const formData = new FormData();
  formData.append("file", blob, `recording.${extensionFor(blob.type)}`);

  const controller = new AbortController();
  transcribeAbortController = controller;
  const timer = setTimeout(() => controller.abort(), TRANSCRIBE_TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch(`${SIDECAR_BASE_URL}/transcribe`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      throw new Error("Transcription is taking too long -- check your network connection and try again.");
    }
    throw err;
  } finally {
    clearTimeout(timer);
    if (transcribeAbortController === controller) transcribeAbortController = null;
  }
  if (!res.ok) {
    let detail = "failed to transcribe audio";
    try {
      detail = ((await res.json()) as { detail?: string }).detail ?? detail;
    } catch {
      // ignore, fall back to the generic message
    }
    throw new Error(detail);
  }
  const data = (await res.json()) as { text: string };
  return data.text;
}

// Labels for the orb's own states. "thinking"/"speaking" are triggered
// externally via orbStatus.ts (from chat.ts/speech.ts), not from this
// module's own click handler. A function (not a module-level constant)
// because it reads the owner's name -- module-level constants evaluate at
// import time, which is before runNameOnboarding() has ever run, so a frozen
// object here would bake in the pre-onboarding empty cache. Called once from
// initVoice(), which only runs after onboarding has resolved.
function buildOrbCopy(ownerName: string): Record<RecorderState, { label: string; sub: string }> {
  return {
    idle: { label: "Tap the orb to talk", sub: "Voice or type — your call" },
    recording: { label: "LISTENING…", sub: "Speak — I'll reply when you pause" },
    transcribing: { label: "TRANSCRIBING…", sub: "Turning speech into words" },
    thinking: { label: "TWIN IS THINKING…", sub: "Thinking it through…" },
    speaking: { label: "SPEAKING · " + ownerName.toUpperCase() + "'S VOICE", sub: `Replying in ${ownerName}'s voice` },
  };
}

const ORB_STATE: Record<RecorderState, string> = {
  idle: "idle",
  recording: "listening",
  transcribing: "transcribing",
  thinking: "thinking",
  speaking: "speaking",
};

// The filmstrip is a purely decorative, non-interactive ambient indicator --
// it groups the 5 real states into the 4 chips the design calls for
// (transcribing buckets under "think", since it's still part of turning the
// utterance into something the twin can act on).
const FILMSTRIP_BUCKET: Record<RecorderState, "idle" | "listen" | "think" | "speak"> = {
  idle: "idle",
  recording: "listen",
  transcribing: "think",
  thinking: "think",
  speaking: "speak",
};

export function initVoice(): void {
  const button = document.querySelector<HTMLButtonElement>("#mic-button");
  const orb = document.querySelector<HTMLElement>("#voice-orb");
  const orbLabel = document.querySelector<HTMLElement>("#voice-orb-label");
  const orbSub = document.querySelector<HTMLElement>("#voice-orb-sub");
  const input = document.querySelector<HTMLInputElement>("#chat-input");
  const form = document.querySelector<HTMLFormElement>("#chat-form");
  const errorEl = document.querySelector<HTMLElement>("#mic-error");
  const toggle = document.querySelector<HTMLButtonElement>("#conversation-toggle");
  const levelMeterEl = document.querySelector<HTMLElement>("#voice-level-meter");
  const levelBars = levelMeterEl
    ? Array.from(levelMeterEl.querySelectorAll<HTMLElement>(".voice-level-bar"))
    : [];
  const pauseControl = document.querySelector<HTMLElement>(".voice-pause");
  const filmstripEl = document.querySelector<HTMLElement>("#voice-filmstrip");
  const filmstripChips = filmstripEl
    ? Array.from(filmstripEl.querySelectorAll<HTMLElement>(".voice-filmstrip-chip"))
    : [];
  if (!button || !orb || !orbLabel || !orbSub || !input || !form || !errorEl) return;

  const ORB_COPY = buildOrbCopy(getOwnerName());

  if (!mediaRecorderSupported()) {
    // Not supported in this WebView -- leave the mic button hidden and hide
    // the hands-free toggle and pause control, since all three depend on mic
    // capture.
    if (toggle) toggle.hidden = true;
    if (pauseControl) pauseControl.hidden = true;
    return;
  }

  button.hidden = false;
  button.disabled = false;

  let state: RecorderState = "idle";
  let session: RecordingSession | null = null;
  let audioCtx: AudioContext | null = null;
  let analyser: AnalyserNode | null = null;
  let levelRAF: number | null = null;
  // Set while a listen is recording (both modes) -- fed the RMS level each
  // frame to detect end-of-utterance. Null only if the AudioContext failed.
  let detector: SilenceDetector | null = null;
  let consecutiveEmpties = 0;

  const stopAudioAnalysis = () => {
    if (levelRAF !== null) {
      cancelAnimationFrame(levelRAF);
      levelRAF = null;
    }
    analyser = null;
    detector = null;
    if (audioCtx) {
      void audioCtx.close().catch(() => {});
      audioCtx = null;
    }
    if (levelMeterEl) levelMeterEl.hidden = true;
    for (const bar of levelBars) bar.style.height = "";
  };

  // Drives BOTH the end-of-utterance detection and the level meter off one
  // AnalyserNode. The detector is the load-bearing half -- without it the mic
  // never closes on its own -- so an absent meter element must not skip any of
  // this; only the bar rendering is conditional on it.
  const startAudioAnalysis = (stream: MediaStream) => {
    // Auto-end the utterance once the user has been quiet for their configured
    // pause; treat a listen with no speech at all as an empty turn.
    detector = createSilenceDetector(
      {
        onSilence: () => session?.stop(),
        onInitialTimeout: () => handleInitialTimeout(),
      },
      { silenceMs: getPauseMs() },
    );
    try {
      audioCtx = new AudioContext();
      // startRecording() awaits getUserMedia, so we're constructing this
      // outside the click's user-gesture task -- WKWebView starts such a
      // context suspended, which would freeze the analyser at a flat 128
      // waveform (level 0.0) and stop the detector ever seeing speech.
      if (audioCtx.state === "suspended") void audioCtx.resume().catch(() => {});
      const source = audioCtx.createMediaStreamSource(stream);
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
    } catch (err) {
      console.error("[voice] audio analysis unavailable", err);
      // No levels means no auto-stop -- fall back to tap-to-stop and say so,
      // rather than leaving the user waiting on a pause that never lands.
      detector = null;
      setState("recording", "Tap the orb again when you're done");
      return;
    }
    if (levelMeterEl) levelMeterEl.hidden = false;
    const data = new Uint8Array(analyser.frequencyBinCount);
    const render = () => {
      if (!analyser) return;
      analyser.getByteTimeDomainData(data);
      let sumSquares = 0;
      for (let i = 0; i < data.length; i++) {
        const v = (data[i] - 128) / 128;
        sumSquares += v * v;
      }
      const level = Math.sqrt(sumSquares / data.length); // ~0..1
      detector?.feed(level, performance.now());
      // feed() can synchronously end the utterance (onSilence -> stop ->
      // handleRecordingStopped -> stopAudioAnalysis), which nulls `analyser`.
      // Bail before re-arming so a torn-down loop doesn't leave a live
      // rAF handle behind that blocks the next listen.
      if (!analyser) return;
      levelBars.forEach((bar, i) => {
        const barLevel = Math.max(0, Math.min(1, level * 5 - i));
        bar.style.height = `${15 + barLevel * 85}%`;
      });
      levelRAF = requestAnimationFrame(render);
    };
    levelRAF = requestAnimationFrame(render);
  };

  const setState = (next: RecorderState, subCaptionOverride?: string) => {
    state = next;
    orb.dataset.state = ORB_STATE[next];
    orbLabel.textContent = ORB_COPY[next].label;
    orbSub.textContent = subCaptionOverride ?? ORB_COPY[next].sub;
    // Can't start a new recording until the current turn (transcribing /
    // thinking / speaking) settles or is cancelled via Esc.
    button.disabled = next === "transcribing" || next === "thinking" || next === "speaking";
    const bucket = FILMSTRIP_BUCKET[next];
    for (const chip of filmstripChips) {
      chip.classList.toggle("voice-filmstrip-chip--active", chip.dataset.bucket === bucket);
    }
  };

  // chat.ts/speech.ts report "thinking"/"speaking" via this shared channel
  // rather than importing voice.ts directly (which already imports FROM
  // chat.ts, so the reverse import would be circular).
  onOrbAudioState((next, subCaption) => {
    if (next === null) {
      if (state === "thinking" || state === "speaking") setState("idle");
      return;
    }
    // A turn started (our own auto-submit, or a typed send). If the mic is
    // still open from a conversation-mode listen, cancel it so the turn takes
    // over cleanly instead of leaving a recording session running behind it.
    if (state === "recording") {
      session?.cancel();
      session = null;
      stopAudioAnalysis();
    }
    setState(next, subCaption);
  });

  // Seed the filmstrip's active chip -- setState() is otherwise only ever
  // called on a real transition, and the orb itself starts idle in markup.
  setState("idle");

  const showError = (message: string) => {
    errorEl.textContent = message;
    errorEl.hidden = false;
  };

  const shakeOrb = () => {
    orb.classList.add("shake");
    setTimeout(() => orb.classList.remove("shake"), 350);
  };

  const updateToggleUI = (on: boolean) => {
    if (!toggle) return;
    toggle.setAttribute("aria-checked", String(on));
    toggle.classList.toggle("conversation-toggle--on", on);
  };

  // Starts (or resumes) a listening turn. The single entry point for the
  // manual mic click, the conversation-mode toggle, the post-reply resume,
  // and the empty-turn retry -- guarded so only ever one recording session
  // and one rAF loop exist at a time.
  const startListening = () => {
    if (state !== "idle" || session !== null || levelRAF !== null) return;
    errorEl.hidden = true;
    void startRecording(
      (blob) => void handleRecordingStopped(blob),
      (message) => {
        showError(message);
        // A mic failure on a hands-free resume can't recover on its own --
        // stop the loop rather than spin retrying a dead mic.
        if (isConversationModeOn()) stopConversation();
      },
    ).then((started) => {
      if (started) {
        session = started;
        const sub = isConversationModeOn() ? "Conversation mode — I'm listening" : undefined;
        setState("recording", sub);
        startAudioAnalysis(started.stream);
      }
    });
  };

  // Conversation mode only: a listening turn that caught no words. Silently
  // relisten, but give up after a few in a row so an empty room doesn't loop
  // forever.
  const handleEmptyTurn = () => {
    if (!isConversationModeOn()) return;
    consecutiveEmpties += 1;
    if (consecutiveEmpties >= MAX_CONSECUTIVE_EMPTIES) {
      stopConversation();
      showError("Paused conversation mode — I wasn't hearing anything. Tap the toggle when you're ready to keep going.");
      return;
    }
    startListening();
  };

  // Fired by the VAD when a listen opens but no speech ever arrives. Discard
  // the (silent) recording without a wasted transcription round-trip.
  const handleInitialTimeout = () => {
    session?.cancel();
    session = null;
    stopAudioAnalysis();
    setState("idle");
    if (isConversationModeOn()) {
      handleEmptyTurn();
    } else {
      // One-shot listen: the mic closing on its own with nothing to show for
      // it needs saying, or the orb just silently reverts to idle.
      showError("Didn't hear anything — tap the orb and try again.");
    }
  };

  const startConversation = () => {
    setConversationMode(true);
    updateToggleUI(true);
    consecutiveEmpties = 0;
    errorEl.hidden = true;
    // If a turn is in flight, the loop starts once it settles (onTurnSettled);
    // otherwise open the mic now.
    if (state === "idle") startListening();
  };

  const stopConversation = () => {
    setConversationMode(false);
    updateToggleUI(false);
    consecutiveEmpties = 0;
    // Tear down an in-flight listen. Mid think/speak we let the current reply
    // finish -- the onTurnSettled handler simply won't resume once the flag
    // is off; mid transcribe, the utterance still gets sent, it just won't
    // reopen the mic afterwards.
    if (state === "recording") {
      session?.cancel();
      session = null;
      stopAudioAnalysis();
      setState("idle");
    }
  };

  const handleRecordingStopped = async (blob: Blob) => {
    stopAudioAnalysis();
    // The recording is over -- the blob is already captured, so clear the
    // session handle. startListening() guards on session === null, so leaving
    // a stale one here would block the next (manual or hands-free) listen.
    session = null;
    setState("transcribing");
    input.disabled = true;
    try {
      const text = await transcribe(blob);
      if (text.trim()) {
        consecutiveEmpties = 0;
        input.value = text.trim();
        markNextReplyForAutoplay();
        input.disabled = false;
        // Speaking IS sending, in both modes -- what conversation mode adds on
        // top is reopening the mic afterwards. Esc is the undo if the
        // transcript came out wrong. Drives the normal chat submit path; its
        // turn-settled signal resumes the hands-free loop. Must be
        // requestSubmit (not submit) so the form's submit handler fires.
        form.requestSubmit();
        return;
      }
      if (isConversationModeOn()) {
        // Empty transcript in conversation mode: relisten silently.
        input.disabled = false;
        setState("idle");
        handleEmptyTurn();
        return;
      }
      showError("Didn't catch that -- no speech detected.");
      shakeOrb();
    } catch (err) {
      if (userCancelledTranscribe) {
        userCancelledTranscribe = false;
      } else {
        showError((err as Error).message);
        // STT failed (e.g. a 503 on a machine without on-device speech) --
        // don't keep looping into the same failure.
        if (isConversationModeOn()) stopConversation();
      }
    }
    input.disabled = false;
    setState("idle");
    input.focus();
  };

  const cancelActiveTurn = () => {
    switch (state) {
      case "recording":
        session?.cancel();
        session = null;
        stopAudioAnalysis();
        setState("idle");
        break;
      case "transcribing":
        userCancelledTranscribe = true;
        transcribeAbortController?.abort();
        break;
      case "thinking":
        cancelCurrentTurn();
        setState("idle");
        break;
      case "speaking":
        stopCurrentPlayback();
        setState("idle");
        break;
      default:
        break; // idle: nothing to cancel
    }
  };

  // The hands-free loop's resume signal: chat.ts emits once a turn's reply
  // (and any spoken audio) have fully settled. Only resume on a clean finish
  // and only while the toggle is still on; on an error pause the loop.
  onTurnSettled((outcome) => {
    if (!isConversationModeOn()) return;
    if (outcome === "spoken" || outcome === "reply-only") {
      startListening();
    } else if (outcome === "error") {
      stopConversation();
    }
    // "cancelled": Esc/new-chat already stopped the loop -- do nothing.
  });

  // chat.ts's new-chat handler (and main.ts's tab switch) ask the loop to
  // stand down through this seam, to avoid importing voice.ts directly.
  onConversationStop(() => stopConversation());

  toggle?.addEventListener("click", () => {
    if (toggle.disabled) return;
    if (isConversationModeOn()) stopConversation();
    else startConversation();
  });

  // Gate the toggle on whether this machine can actually do speech-to-text
  // (the sidecar reports whether it found a usable Whisper backend). The
  // runtime 503 handling in handleRecordingStopped is the safety net if this
  // probe is unavailable.
  if (toggle) {
    void getCapabilities()
      .then((caps) => {
        if (!caps.stt) {
          toggle.disabled = true;
          toggle.title =
            "Conversation mode needs on-device speech-to-text, which isn't installed on this machine.";
        }
      })
      .catch(() => {
        // Leave it enabled; a failed first transcription will stop the loop
        // with a clear message.
      });
  }

  button.addEventListener("click", () => {
    if (state === "idle") {
      startListening();
    } else if (state === "recording") {
      // "I'm done" -- stop now. In conversation mode this short-circuits the
      // 2-second silence wait; in manual mode it's the normal tap-to-stop.
      session?.stop();
    }
  });

  // Global rather than button-scoped: the button is disabled (and thus
  // unfocused) during transcribing/thinking/speaking, but Esc must still be
  // able to cancel those in-flight states. Space/Enter-to-activate needs no
  // extra code -- native <button> elements already fire a click for both.
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state !== "idle") {
      event.preventDefault();
      // One press stops everything: turn the loop off first so cancelling the
      // current turn doesn't immediately trigger a resume-listen.
      if (isConversationModeOn()) stopConversation();
      cancelActiveTurn();
    }
  });
}
