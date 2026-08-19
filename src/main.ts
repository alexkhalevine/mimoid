import { initArchitecture } from "./architecture";
import { initBackup } from "./backup";
import { runBootstrapWizard } from "./bootstrap";
import { initChat } from "./chat";
import { initErrorModal } from "./errorModal";
import { initGoogleCalendarModal } from "./googleCalendarModal";
import { initInfoTips } from "./infoTips";
import { startSystemMonitor } from "./monitor";
import { startNetworkActivityMonitor } from "./network";
import { initOpenRouterModal } from "./openrouterModal";
import { runNameOnboarding } from "./onboarding";
import { initOverview } from "./overview";
import { initPersonaEval } from "./personaEval";
import { initLanguageSelector, initOwnerNameField, initVoicePauseControl } from "./settings";
import { runSetupFlow } from "./setup";
import { initStatusPopover, startStatusPolling } from "./status";
import { initStyle } from "./style";
import { initSystemInfo } from "./systemInfo";
import { initTools } from "./tools";
import { initTrain } from "./train";
import { initVault } from "./vault";
import { initVoice } from "./voice";
import { initVoiceSamples } from "./voiceSamples";
import { requestConversationStop } from "./conversationMode";

function wireTabs(): void {
  const winratePill = document.querySelector<HTMLElement>("#header-winrate-pill");
  const modes = [
    { tabId: "tab-talk", viewId: "talk-view", showWinrate: false },
    { tabId: "tab-train", viewId: "train-view", showWinrate: true },
    { tabId: "tab-tools", viewId: "tools-view", showWinrate: false },
    { tabId: "tab-system", viewId: "system-view", showWinrate: false },
    { tabId: "tab-architecture", viewId: "architecture-view", showWinrate: false },
  ].map(({ tabId, viewId, showWinrate }) => ({
    tab: document.querySelector<HTMLButtonElement>(`#${tabId}`),
    view: document.querySelector<HTMLElement>(`#${viewId}`),
    showWinrate,
  }));
  if (modes.some(({ tab, view }) => !tab || !view)) return;

  const show = (target: (typeof modes)[number]) => {
    for (const mode of modes) {
      const active = mode === target;
      mode.view!.hidden = !active;
      mode.tab!.classList.toggle("view-tab--active", active);
    }
    if (winratePill) winratePill.hidden = !target.showWinrate;
    // Leaving the Talk tab must not leave the hands-free mic open on another tab.
    if (target.view!.id !== "talk-view") requestConversationStop();
  };

  for (const mode of modes) {
    mode.tab!.addEventListener("click", () => show(mode));
  }
}

interface Subtab {
  tabId: string;
  panelId: string;
}

/** Wires an N-way tab switcher. Returns a function that activates the
 * first subtab (the default), for callers to re-assert it once all panels'
 * own init*() calls have finished revealing themselves independently. */
function wireSubtabs(subtabs: Subtab[]): (() => void) | null {
  const entries = subtabs.map(({ tabId, panelId }) => ({
    tab: document.querySelector<HTMLButtonElement>(`#${tabId}`),
    panel: document.querySelector<HTMLElement>(`#${panelId}`),
  }));
  if (entries.some(({ tab, panel }) => !tab || !panel)) return null;

  const show = (target: (typeof entries)[number]) => {
    for (const entry of entries) {
      const active = entry === target;
      entry.panel!.hidden = !active;
      entry.tab!.classList.toggle("subtab--active", active);
    }
  };

  for (const entry of entries) {
    entry.tab!.addEventListener("click", () => show(entry));
  }

  return () => show(entries[0]);
}

window.addEventListener("DOMContentLoaded", () => {
  initErrorModal();
  initInfoTips();
  startStatusPolling();
  startSystemMonitor();
  startNetworkActivityMonitor();
  initStatusPopover();
  wireTabs();
  const showDefaultSubtab = wireSubtabs([
    { tabId: "subtab-overview", panelId: "overview-panel" },
    { tabId: "subtab-memories", panelId: "train-panel" },
    { tabId: "subtab-voice", panelId: "voice-samples-panel" },
    { tabId: "subtab-style", panelId: "style-panel" },
    { tabId: "subtab-vault", panelId: "vault-panel" },
    { tabId: "subtab-persona", panelId: "persona-panel" },
    { tabId: "subtab-backup", panelId: "backup-panel" },
  ]);

  void (async () => {
    await runBootstrapWizard();
    await runSetupFlow(() => {
      void (async () => {
        // Blocks until the owner's name is known -- resolves immediately on
        // every launch after the first. Everything below (the persona
        // pipeline in particular) assumes a real name is already saved.
        await runNameOnboarding();

        void initLanguageSelector();
        void initVoicePauseControl();
        void initChat();
        // Each init*() call reveals its own panel (panel.hidden = false)
        // independently, so once they're all done, explicitly re-assert which
        // Train subtab should actually be showing.
        void Promise.all([
          initOverview(),
          initTrain(),
          initVoiceSamples(),
          initStyle(),
          initVault(),
          initPersonaEval(),
          initBackup(),
          initOpenRouterModal(),
          initGoogleCalendarModal(),
        ]).then(() => {
          showDefaultSubtab?.();
        });
        void initSystemInfo();
        initTools();
        void initOwnerNameField();
        void initArchitecture();
        initVoice();
      })();
    });
  })();
});
