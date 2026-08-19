import {
  choosePersonaDuel,
  createPersonaDuel,
  createStyleEntry,
  getPersonaDuelSummary,
  type PersonaDuelSummary,
} from "./sidecar";
import { rollNumber } from "./numberRoll";

interface ScoreboardElements {
  winpct: HTMLElement;
  armTableBody: HTMLElement;
  recentChart: HTMLElement;
  summaryFallback: HTMLElement;
}

function getScoreboardElements(): ScoreboardElements | null {
  const winpct = document.querySelector<HTMLElement>("#persona-winpct");
  const armTableBody = document.querySelector<HTMLElement>("#persona-arm-table-body");
  const recentChart = document.querySelector<HTMLElement>("#persona-recent-chart");
  const summaryFallback = document.querySelector<HTMLElement>("#persona-summary-text");
  if (!winpct || !armTableBody || !recentChart || !summaryFallback) return null;
  return { winpct, armTableBody, recentChart, summaryFallback };
}

// Tracks the previously-displayed rate so a vote can roll from old -> new;
// `null` means "nothing shown yet" (page load), which should set instantly.
let lastDisplayedRate: number | null = null;

function labelFor(summary: PersonaDuelSummary, arm: string): string {
  return summary.arms.find((a) => a.arm === arm)?.label ?? arm;
}

function renderScoreboard(els: ScoreboardElements, summary: PersonaDuelSummary): void {
  els.summaryFallback.hidden = summary.total > 0;

  const headerValue = document.querySelector<HTMLElement>("#header-winrate-value");

  if (summary.total === 0 || summary.default_arm_rate === null) {
    els.winpct.textContent = "—%";
    if (headerValue) headerValue.textContent = "—%";
    lastDisplayedRate = null;
  } else {
    const rate = Math.round(summary.default_arm_rate * 100);
    if (lastDisplayedRate === null) {
      els.winpct.textContent = `${rate}%`;
      if (headerValue) headerValue.textContent = `${rate}%`;
    } else {
      rollNumber(els.winpct, lastDisplayedRate, rate, 400, "%");
      if (headerValue) rollNumber(headerValue, lastDisplayedRate, rate, 400, "%");
    }
    lastDisplayedRate = rate;
  }

  els.armTableBody.innerHTML = "";
  for (const arm of summary.arms) {
    const row = document.createElement("tr");
    const isDefault = arm.arm === summary.default_arm;
    row.className = isDefault ? "persona-arm-row persona-arm-row--default" : "persona-arm-row";

    const label = document.createElement("td");
    label.className = "persona-arm-label";
    label.textContent = isDefault ? `${arm.label} (default)` : arm.label;

    const record = document.createElement("td");
    record.className = "persona-arm-record";
    record.textContent = `${arm.wins}-${arm.losses}-${arm.ties}`;

    const rate = document.createElement("td");
    rate.className = "persona-arm-rate";
    rate.textContent = arm.win_rate !== null ? `${Math.round(arm.win_rate * 100)}%` : "—";

    row.append(label, record, rate);
    els.armTableBody.appendChild(row);
  }

  els.recentChart.innerHTML = "";
  for (const outcome of summary.recent) {
    const bar = document.createElement("div");
    bar.className =
      outcome === "default"
        ? "persona-recent-bar persona-recent-bar--default"
        : outcome === "tie"
          ? "persona-recent-bar persona-recent-bar--tie"
          : "persona-recent-bar";
    els.recentChart.appendChild(bar);
  }
}

interface CompareElements {
  form: HTMLFormElement;
  prompt: HTMLInputElement;
  compareButton: HTMLButtonElement;
  compareStatus: HTMLElement;
  errorEl: HTMLElement;
  answers: HTMLElement;
  answerA: HTMLElement;
  answerB: HTMLElement;
  chooseA: HTMLButtonElement;
  chooseB: HTMLButtonElement;
  chooseTie: HTMLButtonElement;
  reveal: HTMLElement;
}

function getCompareElements(): CompareElements | null {
  const form = document.querySelector<HTMLFormElement>("#persona-form");
  const prompt = document.querySelector<HTMLInputElement>("#persona-prompt");
  const compareButton = document.querySelector<HTMLButtonElement>("#persona-compare");
  const compareStatus = document.querySelector<HTMLElement>("#persona-compare-status");
  const errorEl = document.querySelector<HTMLElement>("#persona-error");
  const answers = document.querySelector<HTMLElement>("#persona-answers");
  const answerA = document.querySelector<HTMLElement>("#persona-answer-a");
  const answerB = document.querySelector<HTMLElement>("#persona-answer-b");
  const chooseA = document.querySelector<HTMLButtonElement>("#persona-choose-a");
  const chooseB = document.querySelector<HTMLButtonElement>("#persona-choose-b");
  const chooseTie = document.querySelector<HTMLButtonElement>("#persona-choose-tie");
  const reveal = document.querySelector<HTMLElement>("#persona-reveal");
  if (
    !form ||
    !prompt ||
    !compareButton ||
    !compareStatus ||
    !errorEl ||
    !answers ||
    !answerA ||
    !answerB ||
    !chooseA ||
    !chooseB ||
    !chooseTie ||
    !reveal
  ) {
    return null;
  }
  return { form, prompt, compareButton, compareStatus, errorEl, answers, answerA, answerB, chooseA, chooseB, chooseTie, reveal };
}

function setCompareStatus(els: CompareElements, text: string | null): void {
  if (text) {
    els.compareStatus.textContent = text;
    els.compareStatus.hidden = false;
  } else {
    els.compareStatus.hidden = true;
  }
}

// The composer captures what Alex would actually say, written BEFORE either
// generated answer exists on screen -- see main.py's create_style_entry
// invariant comment for why that ordering, not an edit-after-the-fact step,
// is what keeps a saved example genuinely human-authored.
interface ComposeElements {
  compose: HTMLElement;
  body: HTMLTextAreaElement;
  error: HTMLElement;
  revealButton: HTMLButtonElement;
  saveButton: HTMLButtonElement;
  dismissButton: HTMLButtonElement;
  saved: HTMLElement;
}

function getComposeElements(): ComposeElements | null {
  const compose = document.querySelector<HTMLElement>("#persona-compose");
  const body = document.querySelector<HTMLTextAreaElement>("#persona-compose-body");
  const error = document.querySelector<HTMLElement>("#persona-compose-error");
  const revealButton = document.querySelector<HTMLButtonElement>("#persona-reveal-answers");
  const saveButton = document.querySelector<HTMLButtonElement>("#persona-compose-save");
  const dismissButton = document.querySelector<HTMLButtonElement>("#persona-compose-dismiss");
  const saved = document.querySelector<HTMLElement>("#persona-compose-saved");
  if (!compose || !body || !error || !revealButton || !saveButton || !dismissButton || !saved) return null;
  return { compose, body, error, revealButton, saveButton, dismissButton, saved };
}

/** Resets the composer's internal state for a new round -- does NOT touch
 * `compose.hidden` itself, since the caller shows it immediately on submit,
 * before this reset even runs. */
function resetCompose(els: ComposeElements): void {
  els.body.value = "";
  els.body.readOnly = false;
  els.body.hidden = false;
  els.error.hidden = true;
  els.revealButton.hidden = false;
  els.revealButton.disabled = true;
  els.saveButton.hidden = true;
  els.dismissButton.hidden = true;
  els.saved.hidden = true;
}

export async function initPersonaEval(): Promise<void> {
  const panel = document.querySelector<HTMLElement>("#persona-panel");
  if (!panel) return;
  panel.hidden = false;

  const scoreboard = getScoreboardElements();
  const compare = getCompareElements();
  // The composer only makes sense alongside a working compare flow -- if
  // #persona-form's elements are missing, there's nothing for it to sit
  // between, so don't bother wiring it either.
  const compose = compare ? getComposeElements() : null;

  let currentId: string | null = null;
  let currentPrompt = "";

  const setChoosing = (enabled: boolean) => {
    if (!compare) return;
    compare.chooseA.disabled = !enabled;
    compare.chooseB.disabled = !enabled;
    compare.chooseTie.disabled = !enabled;
  };

  const wireSave = () => {
    if (!compose) return;

    compose.dismissButton.addEventListener("click", () => {
      compose.compose.hidden = true;
    });

    compose.saveButton.addEventListener("click", () => {
      const content = compose.body.value.trim();
      if (!content) {
        compose.error.textContent = "Write something before saving.";
        compose.error.hidden = false;
        return;
      }
      compose.error.hidden = true;
      compose.saveButton.disabled = true;
      compose.dismissButton.disabled = true;
      compose.saveButton.classList.add("btn-loading");

      void (async () => {
        try {
          await createStyleEntry({
            kind: "qa",
            prompt: currentPrompt || undefined,
            content,
            source: "own-answer",
          });
          compose.body.hidden = true;
          compose.saveButton.hidden = true;
          compose.dismissButton.hidden = true;
          compose.saved.hidden = false;
        } catch (err) {
          compose.error.textContent = (err as Error).message;
          compose.error.hidden = false;
        } finally {
          compose.saveButton.disabled = false;
          compose.dismissButton.disabled = false;
          compose.saveButton.classList.remove("btn-loading");
        }
      })();
    });
  };

  if (compare) {
    const choose = (choice: "a" | "b" | "tie") => {
      if (!currentId) return;
      const id = currentId;
      currentId = null;
      void (async () => {
        setChoosing(false);
        try {
          const summary = await choosePersonaDuel(id, choice);
          if (scoreboard) renderScoreboard(scoreboard, summary);

          const armA = summary.arm_a ?? "";
          const armB = summary.arm_b ?? "";
          compare.reveal.textContent = `A was ${labelFor(summary, armA)} · B was ${labelFor(summary, armB)}.`;
          compare.reveal.hidden = false;

          if (choice !== "tie") {
            const winnerAnswerEl = choice === "a" ? compare.answerA : compare.answerB;
            const winnerCard = winnerAnswerEl.closest<HTMLElement>(".persona-answer");
            if (winnerCard) {
              winnerCard.classList.add("persona-answer--winner-flash");
              setTimeout(() => winnerCard.classList.remove("persona-answer--winner-flash"), 700);
            }
          }

          // Offer to save Alex's own pre-written answer regardless of which
          // side won (or tied) -- it's his own words either way, not tied to
          // which generated answer he preferred.
          if (compose && compose.body.value.trim() && compose.saveButton.hidden) {
            compose.saveButton.hidden = false;
            compose.dismissButton.hidden = false;
          }
        } catch (err) {
          compare.errorEl.textContent = (err as Error).message;
          compare.errorEl.hidden = false;
        }
      })();
    };

    compare.chooseA.addEventListener("click", () => choose("a"));
    compare.chooseB.addEventListener("click", () => choose("b"));
    compare.chooseTie.addEventListener("click", () => choose("tie"));

    if (compose) {
      compose.revealButton.addEventListener("click", () => {
        compare.answers.hidden = false;
        compare.chooseTie.hidden = false;
        compose.revealButton.hidden = true;
        compose.body.readOnly = true;
        setChoosing(true);
      });
    }

    compare.form.addEventListener("submit", (event) => {
      event.preventDefault();
      const prompt = compare.prompt.value.trim();
      if (!prompt) return;

      compare.compareButton.disabled = true;
      compare.compareButton.classList.add("btn-loading");
      compare.errorEl.hidden = true;
      compare.reveal.hidden = true;
      compare.answers.hidden = true;
      compare.chooseTie.hidden = true;
      currentId = null;
      currentPrompt = prompt;

      // Shown the instant Compare is clicked, before either answer exists --
      // this is what makes anything written here provably predate exposure
      // to the model's own wording, not a race against generation finishing.
      if (compose) {
        resetCompose(compose);
        compose.compose.hidden = false;
      }

      // The backend generates both answers with two concurrent, fully-
      // blocking calls -- there's no real progress to back a determinate
      // bar with, so this is an honest indeterminate wait (the spinner) plus
      // an elapsed-time readout, not a fake percentage.
      const startedAt = Date.now();
      const tick = () => {
        const elapsed = Math.round((Date.now() - startedAt) / 1000);
        setCompareStatus(compare, elapsed >= 3 ? `Generating two answers… (${elapsed}s elapsed)` : "Generating two answers…");
      };
      tick();
      const ticker = setInterval(tick, 1000);

      void (async () => {
        try {
          const result = await createPersonaDuel(prompt);
          currentId = result.id;
          compare.answerA.textContent = result.a;
          compare.answerB.textContent = result.b;
          if (compose) compose.revealButton.disabled = false;
        } catch (err) {
          compare.errorEl.textContent = (err as Error).message;
          compare.errorEl.hidden = false;
          // No answers were generated, so there's nothing to "reveal" -- but
          // whatever Alex wrote is still his, and still worth offering to
          // save directly.
          if (compose) {
            compose.revealButton.hidden = true;
            compose.saveButton.hidden = false;
            compose.dismissButton.hidden = false;
          }
        } finally {
          clearInterval(ticker);
          setCompareStatus(compare, null);
          compare.compareButton.disabled = false;
          compare.compareButton.classList.remove("btn-loading");
        }
      })();
    });
  }

  wireSave();

  if (scoreboard) {
    try {
      const summary = await getPersonaDuelSummary();
      renderScoreboard(scoreboard, summary);
    } catch (err) {
      console.error("failed to load persona duel summary", err);
      scoreboard.summaryFallback.hidden = false;
    }
  }
}
