import { getPersonaDuelSummary, listMemories, listStyleEntries, listVoiceSamples } from "./sidecar";
import { VOICE_PROMPT_COUNT } from "./voiceSamples";

const RING_CIRCUMFERENCE = 2 * Math.PI * 46;

// Reasonable "looks reasonably fed" targets for the stat bars -- not hard
// caps, just enough to fill the bar and stop nudging once there's a solid
// amount of material.
const MEMORIES_TARGET = 30;
const STYLE_TARGET = 20;

interface NextStep {
  label: string;
  subtabId: string;
}

function pickNextStep(
  memoriesCount: number,
  styleCount: number,
  voiceCount: number,
  personaTotal: number,
  defaultArmRate: number | null,
): NextStep {
  if (memoriesCount === 0) return { label: "Add your first memory", subtabId: "subtab-memories" };
  if (styleCount === 0) return { label: "Add a writing sample or Q&A example", subtabId: "subtab-style" };
  if (voiceCount === 0) return { label: "Record your first voice sample", subtabId: "subtab-voice" };
  if (voiceCount < VOICE_PROMPT_COUNT) {
    const remaining = VOICE_PROMPT_COUNT - voiceCount;
    return { label: `Record ${remaining} more voice sample${remaining === 1 ? "" : "s"}`, subtabId: "subtab-voice" };
  }
  if (personaTotal === 0) return { label: "Run your first persona check", subtabId: "subtab-persona" };
  if (defaultArmRate !== null && defaultArmRate < 0.5) {
    // Below even odds, the CHALLENGER setup is winning outright -- that's a
    // signal to go look at persona check, not a generic "add more material"
    // nudge (which is what a modest-but-still-winning rate gets instead).
    return { label: "A different setup is beating your default — review persona check", subtabId: "subtab-persona" };
  }
  if (defaultArmRate !== null && defaultArmRate < 0.7) {
    return { label: "Add more memories or style examples", subtabId: "subtab-memories" };
  }
  return { label: "Keep training — check in on persona check", subtabId: "subtab-persona" };
}

export async function initOverview(): Promise<void> {
  const panel = document.querySelector<HTMLElement>("#overview-panel");
  const ringArc = document.querySelector<SVGCircleElement>("#overview-ring-arc");
  const ringPct = document.querySelector<SVGTextElement>("#overview-ring-pct");
  const heroTitle = document.querySelector<HTMLElement>("#overview-hero-title");
  const heroDesc = document.querySelector<HTMLElement>("#overview-hero-desc");
  const nextStepLabel = document.querySelector<HTMLElement>("#overview-next-step-label");
  const nextStepGo = document.querySelector<HTMLButtonElement>("#overview-next-step-go");
  const memoriesValue = document.querySelector<HTMLElement>("#overview-stat-memories-value");
  const memoriesBar = document.querySelector<HTMLElement>("#overview-stat-memories-bar");
  const styleValue = document.querySelector<HTMLElement>("#overview-stat-style-value");
  const styleBar = document.querySelector<HTMLElement>("#overview-stat-style-bar");
  const voiceValue = document.querySelector<HTMLElement>("#overview-stat-voice-value");
  const voiceTotal = document.querySelector<HTMLElement>("#overview-stat-voice-total");
  const voiceBar = document.querySelector<HTMLElement>("#overview-stat-voice-bar");
  const personaValue = document.querySelector<HTMLElement>("#overview-stat-persona-value");
  const personaCaption = document.querySelector<HTMLElement>("#overview-stat-persona-caption");
  const personaBar = document.querySelector<HTMLElement>("#overview-stat-persona-bar");
  if (
    !panel ||
    !ringArc ||
    !ringPct ||
    !heroTitle ||
    !heroDesc ||
    !nextStepLabel ||
    !nextStepGo ||
    !memoriesValue ||
    !memoriesBar ||
    !styleValue ||
    !styleBar ||
    !voiceValue ||
    !voiceTotal ||
    !voiceBar ||
    !personaValue ||
    !personaCaption ||
    !personaBar
  ) {
    return;
  }

  const [memories, styleEntries, voiceSamples, personaSummary] = await Promise.all([
    listMemories(),
    listStyleEntries(),
    listVoiceSamples(),
    getPersonaDuelSummary(),
  ]);

  memoriesValue.textContent = String(memories.length);
  memoriesBar.style.width = `${Math.min(100, (memories.length / MEMORIES_TARGET) * 100)}%`;

  styleValue.textContent = String(styleEntries.length);
  styleBar.style.width = `${Math.min(100, (styleEntries.length / STYLE_TARGET) * 100)}%`;

  voiceValue.textContent = String(voiceSamples.length);
  voiceTotal.textContent = String(VOICE_PROMPT_COUNT);
  voiceBar.style.width = `${Math.min(100, (voiceSamples.length / VOICE_PROMPT_COUNT) * 100)}%`;

  if (personaSummary.total > 0 && personaSummary.default_arm_rate !== null) {
    const pct = Math.round(personaSummary.default_arm_rate * 100);
    const defaultWins = personaSummary.arms.find((arm) => arm.arm === personaSummary.default_arm)?.wins ?? 0;
    personaValue.textContent = `${pct}%`;
    personaCaption.textContent = `${defaultWins} of ${personaSummary.decided} decisive checks`;
    personaBar.style.width = `${pct}%`;

    ringArc.style.strokeDashoffset = String(RING_CIRCUMFERENCE * (1 - pct / 100));
    ringPct.textContent = `${pct}%`;
    heroTitle.textContent = "Coming into focus";
    heroDesc.innerHTML = `The more you feed it, the more it sounds like you. Your default setup is winning the blind test <strong>${pct}%</strong> of the time — keep going and it'll hold up in every reply.`;
  } else {
    personaValue.textContent = "—%";
    personaCaption.textContent = "no checks yet";
    personaBar.style.width = "0%";
    ringArc.style.strokeDashoffset = String(RING_CIRCUMFERENCE);
    ringPct.textContent = "—%";
  }

  const nextStep = pickNextStep(
    memories.length,
    styleEntries.length,
    voiceSamples.length,
    personaSummary.total,
    personaSummary.default_arm_rate,
  );
  nextStepLabel.textContent = nextStep.label;
  nextStepGo.onclick = () => {
    document.querySelector<HTMLButtonElement>(`#${nextStep.subtabId}`)?.click();
  };

  panel.hidden = false;
}
