import { saveBlobAs } from "./fileSave";
import { downloadBackup, getBackupConfig, runBackup, updateBackupConfig } from "./sidecar";

const SAVED_PASSWORD_PLACEHOLDER = "•••••••• (saved — leave blank to keep)";

interface BackupElements {
  panel: HTMLElement;
  configForm: HTMLFormElement;
  url: HTMLInputElement;
  username: HTMLInputElement;
  password: HTMLInputElement;
  configError: HTMLElement;
  runButton: HTMLButtonElement;
  status: HTMLElement;
  runError: HTMLElement;
  downloadButton: HTMLButtonElement;
  downloadError: HTMLElement;
}

function getElements(): BackupElements | null {
  const panel = document.querySelector<HTMLElement>("#backup-panel");
  const configForm = document.querySelector<HTMLFormElement>("#backup-config-form");
  const url = document.querySelector<HTMLInputElement>("#backup-url");
  const username = document.querySelector<HTMLInputElement>("#backup-username");
  const password = document.querySelector<HTMLInputElement>("#backup-password");
  const configError = document.querySelector<HTMLElement>("#backup-config-error");
  const runButton = document.querySelector<HTMLButtonElement>("#backup-run");
  const status = document.querySelector<HTMLElement>("#backup-status");
  const runError = document.querySelector<HTMLElement>("#backup-error");
  const downloadButton = document.querySelector<HTMLButtonElement>("#backup-download");
  const downloadError = document.querySelector<HTMLElement>("#backup-download-error");
  if (
    !panel ||
    !configForm ||
    !url ||
    !username ||
    !password ||
    !configError ||
    !runButton ||
    !status ||
    !runError ||
    !downloadButton ||
    !downloadError
  ) {
    return null;
  }
  return {
    panel,
    configForm,
    url,
    username,
    password,
    configError,
    runButton,
    status,
    runError,
    downloadButton,
    downloadError,
  };
}

function renderStatus(els: BackupElements, lastBackupAt: string | null): void {
  els.status.textContent = lastBackupAt
    ? `Last backed up: ${new Date(lastBackupAt).toLocaleString()}`
    : "Never backed up yet.";
}

export async function initBackup(): Promise<void> {
  const els = getElements();
  if (!els) return;

  els.panel.hidden = false;

  // Wire event listeners before the initial config fetch resolves, so a
  // form submit/click can never race ahead of preventDefault() and fall
  // through to the browser's native (page-reloading) form submission.
  els.configForm.addEventListener("submit", (event) => {
    event.preventDefault();
    els.configError.hidden = true;
    const submitButton = els.configForm.querySelector<HTMLButtonElement>("#backup-save");
    if (submitButton) {
      submitButton.disabled = true;
      submitButton.classList.add("btn-loading");
    }

    void (async () => {
      try {
        const config = await updateBackupConfig({
          url: els.url.value.trim(),
          username: els.username.value.trim(),
          password: els.password.value.trim() || undefined,
        });
        els.password.value = "";
        els.password.placeholder = config.has_password ? SAVED_PASSWORD_PLACEHOLDER : "App password";
      } catch (err) {
        els.configError.textContent = (err as Error).message;
        els.configError.hidden = false;
      } finally {
        if (submitButton) {
          submitButton.disabled = false;
          submitButton.classList.remove("btn-loading");
        }
      }
    })();
  });

  els.runButton.addEventListener("click", () => {
    void (async () => {
      els.runButton.disabled = true;
      els.runError.hidden = true;
      els.status.textContent = "Backing up…";
      try {
        const result = await runBackup();
        renderStatus(els, result.last_backup_at);
      } catch (err) {
        els.runError.textContent = (err as Error).message;
        els.runError.hidden = false;
        try {
          const config = await getBackupConfig();
          renderStatus(els, config.last_backup_at);
        } catch {
          // keep the "Backing up…" text rather than compound the failure
        }
      } finally {
        els.runButton.disabled = false;
      }
    })();
  });

  els.downloadButton.addEventListener("click", () => {
    void (async () => {
      els.downloadButton.disabled = true;
      els.downloadButton.classList.add("btn-loading");
      els.downloadError.hidden = true;
      try {
        const blob = await downloadBackup();
        const stamp = new Date()
          .toISOString()
          .replace(/[-:]/g, "")
          .replace(/\..+/, "")
          .replace("T", "-");
        await saveBlobAs(blob, `mimoid-backup-${stamp}.zip`, ["zip"]);
      } catch (err) {
        els.downloadError.textContent = (err as Error).message;
        els.downloadError.hidden = false;
      } finally {
        els.downloadButton.disabled = false;
        els.downloadButton.classList.remove("btn-loading");
      }
    })();
  });

  try {
    const config = await getBackupConfig();
    els.url.value = config.url;
    els.username.value = config.username;
    els.password.placeholder = config.has_password ? SAVED_PASSWORD_PLACEHOLDER : "App password";
    renderStatus(els, config.last_backup_at);
  } catch (err) {
    console.error("failed to load backup config", err);
  }
}
