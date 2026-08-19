import {
  createVoiceSample,
  deleteVoiceSample,
  listVoiceSamples,
  voiceSampleAudioUrl,
  type VoiceSample,
} from "./sidecar";
import { extensionFor, mediaRecorderSupported, startRecording, type RecordingSession } from "./recorder";
import { getOwnerName } from "./settings";

// Templates, not the final strings -- PROMPTS[0] carries an {owner_name}
// token interpolated by buildPrompts() when initVoiceSamples() runs (after
// first-run onboarding has resolved). A module-level array evaluates at
// import time, before the owner's name is known, so the interpolation can't
// happen here.
const PROMPTS = [
  "Hi, I'm {owner_name}. This is what my voice sounds like -- calm, clear, and a little curious. I want this recording to sound like me on an ordinary day, just talking.",
  "The quick brown fox jumps over the lazy dog, while the five boxing wizards jump quickly. Pack my box with five dozen liquor jugs before the delivery arrives.",
  "I like to think out loud when I'm working through a hard problem, especially early in the morning. Sometimes I'll pace around the room, and other times I just stare at the screen until something clicks.",
  "Wait, did I already tell you about that? I feel like I bring it up constantly, but it's genuinely one of my favorite stories to share with people.",
  "One, two, three, four, five, six, seven, eight, nine, ten. Testing, testing -- how does this sound? I'm just reading numbers out loud to give the recording something steady and even to work with.",
  "Honestly, some days are great and some days are a mess, and that's just how it goes. I try not to take any single day too seriously.",
  "If I had to describe myself in a few sentences, I'd probably start with what I care about, then get sidetracked talking about something completely unrelated.",
  "Thanks for listening. That should be enough for the digital twin to get started, though I could probably keep talking for another few minutes without much effort.",
];

/** Total number of guided recording prompts -- reused by the Train Overview
 * tab so its "N / total samples recorded" stat doesn't duplicate this list. */
export const VOICE_PROMPT_COUNT = PROMPTS.length;

function buildPrompts(ownerName: string): string[] {
  return PROMPTS.map((prompt) => prompt.replace("{owner_name}", ownerName));
}

function renderPromptRow(
  prompt: string,
  existing: VoiceSample | undefined,
  showError: (message: string) => void,
  onChange: () => void,
): HTMLElement {
  const row = document.createElement("div");
  row.className = "voice-prompt-row";

  const text = document.createElement("p");
  text.className = "voice-prompt-text";
  text.textContent = prompt;

  const status = document.createElement("span");
  status.className = "voice-prompt-status";

  const recordButton = document.createElement("button");
  recordButton.type = "button";
  recordButton.className = "voice-prompt-record";

  const player = document.createElement("audio");
  player.controls = true;
  player.className = "voice-prompt-audio";

  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.textContent = "Delete";
  deleteButton.className = "voice-prompt-delete";

  let sample = existing;
  let session: RecordingSession | null = null;
  let recording = false;

  const updateUi = () => {
    status.textContent = sample ? "Recorded" : "Not recorded";
    status.classList.toggle("voice-prompt-status--done", Boolean(sample));
    row.classList.toggle("voice-prompt-row--pending", !sample);
    recordButton.textContent = recording ? "⏹ Stop" : sample ? "Re-record" : "Record";
    deleteButton.hidden = !sample || recording;
    if (sample) {
      player.hidden = false;
      player.src = voiceSampleAudioUrl(sample.id);
    } else {
      player.hidden = true;
      player.removeAttribute("src");
    }
    onChange();
  };

  recordButton.addEventListener("click", () => {
    if (recording) {
      session?.stop();
      return;
    }

    void startRecording(
      (blob) => {
        void (async () => {
          recordButton.disabled = true;
          try {
            const previous = sample;
            const created = await createVoiceSample(prompt, blob, extensionFor(blob.type));
            if (previous) await deleteVoiceSample(previous.id).catch(() => undefined);
            sample = created;
          } catch (err) {
            showError((err as Error).message);
          } finally {
            recording = false;
            recordButton.disabled = false;
            updateUi();
          }
        })();
      },
      (message) => {
        recording = false;
        recordButton.disabled = false;
        showError(message);
        updateUi();
      },
    ).then((started) => {
      if (started) {
        session = started;
        recording = true;
        updateUi();
      }
    });
  });

  deleteButton.addEventListener("click", () => {
    const current = sample;
    if (!current) return;
    void (async () => {
      deleteButton.disabled = true;
      try {
        await deleteVoiceSample(current.id);
        sample = undefined;
      } catch (err) {
        showError((err as Error).message);
      } finally {
        deleteButton.disabled = false;
        updateUi();
      }
    })();
  });

  updateUi();
  row.append(text, status, recordButton, player, deleteButton);
  return row;
}

export async function initVoiceSamples(): Promise<void> {
  const panel = document.querySelector<HTMLElement>("#voice-samples-panel");
  const list = document.querySelector<HTMLElement>("#voice-prompt-list");
  const errorEl = document.querySelector<HTMLElement>("#voice-samples-error");
  const countValue = document.querySelector<HTMLElement>("#voice-samples-count-value");
  const countTotal = document.querySelector<HTMLElement>("#voice-samples-count-total");
  const resetButton = document.querySelector<HTMLButtonElement>("#voice-samples-reset");
  if (!panel || !list || !errorEl) return;

  panel.hidden = false;
  if (countTotal) countTotal.textContent = String(VOICE_PROMPT_COUNT);
  const prompts = buildPrompts(getOwnerName());

  const refreshCount = () => {
    if (!countValue) return;
    countValue.textContent = String(list.querySelectorAll(".voice-prompt-status--done").length);
  };

  if (!mediaRecorderSupported()) {
    errorEl.textContent = "Microphone recording isn't supported in this window.";
    errorEl.hidden = false;
    return;
  }

  const showError = (message: string) => {
    errorEl.textContent = message;
    errorEl.hidden = false;
  };

  const renderList = (samples: VoiceSample[]) => {
    const byPrompt = new Map(samples.map((sample) => [sample.prompt, sample]));
    list.innerHTML = "";
    for (const prompt of prompts) {
      list.appendChild(renderPromptRow(prompt, byPrompt.get(prompt), showError, refreshCount));
    }
    refreshCount();
  };

  renderList(await listVoiceSamples());

  if (resetButton) {
    resetButton.addEventListener("click", () => {
      if (!confirm("Delete all recorded voice samples? This can't be undone -- you'll re-record them from scratch."))
        return;
      void (async () => {
        resetButton.disabled = true;
        try {
          const existing = await listVoiceSamples();
          for (const sample of existing) {
            await deleteVoiceSample(sample.id).catch(() => undefined);
          }
          renderList(await listVoiceSamples());
        } catch (err) {
          showError((err as Error).message);
        } finally {
          resetButton.disabled = false;
        }
      })();
    });
  }
}
