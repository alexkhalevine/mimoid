import { openUrl } from "@tauri-apps/plugin-opener";
import {
  connectGoogleCalendar,
  disconnectGoogleCalendar,
  getGoogleStatus,
  saveGoogleCredentials,
  type GoogleStatus,
} from "./sidecar";

interface GoogleModalElements {
  dialog: HTMLDialogElement;
  credentialsForm: HTMLFormElement;
  clientId: HTMLInputElement;
  clientSecret: HTMLInputElement;
  credentialsError: HTMLElement;
  statusText: HTMLElement;
  connectButton: HTMLButtonElement;
  disconnectButton: HTMLButtonElement;
  connectError: HTMLElement;
  close: HTMLButtonElement;
}

function getElements(): GoogleModalElements | null {
  const dialog = document.querySelector<HTMLDialogElement>("#google-calendar-modal");
  const credentialsForm = document.querySelector<HTMLFormElement>("#google-credentials-form");
  const clientId = document.querySelector<HTMLInputElement>("#google-client-id");
  const clientSecret = document.querySelector<HTMLInputElement>("#google-client-secret");
  const credentialsError = document.querySelector<HTMLElement>("#google-credentials-error");
  const statusText = document.querySelector<HTMLElement>("#google-status-text");
  const connectButton = document.querySelector<HTMLButtonElement>("#google-connect-button");
  const disconnectButton = document.querySelector<HTMLButtonElement>("#google-disconnect-button");
  const connectError = document.querySelector<HTMLElement>("#google-connect-error");
  const close = document.querySelector<HTMLButtonElement>("#google-modal-close");
  if (
    !dialog ||
    !credentialsForm ||
    !clientId ||
    !clientSecret ||
    !credentialsError ||
    !statusText ||
    !connectButton ||
    !disconnectButton ||
    !connectError ||
    !close
  )
    return null;
  return {
    dialog,
    credentialsForm,
    clientId,
    clientSecret,
    credentialsError,
    statusText,
    connectButton,
    disconnectButton,
    connectError,
    close,
  };
}

// While a connect attempt is in flight, the modal polls the sidecar rather
// than waiting on a single request -- the sidecar's own loopback listener
// is what's actually waiting on Google, out of band, potentially for
// minutes while the user picks an account in the browser.
const POLL_INTERVAL_MS = 1500;
let pollTimer: ReturnType<typeof setTimeout> | null = null;

function stopPolling(): void {
  if (pollTimer !== null) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
}

let onStatusChanged: ((status: GoogleStatus) => void) | null = null;

function render(els: GoogleModalElements, status: GoogleStatus): void {
  if (status.connecting) {
    els.statusText.textContent = "Waiting for Google — check the browser window that just opened…";
    els.connectButton.hidden = true;
    els.disconnectButton.textContent = "Cancel";
    els.disconnectButton.hidden = false;
  } else if (status.connected) {
    els.statusText.textContent = "Connected.";
    els.connectButton.hidden = true;
    els.disconnectButton.textContent = "Disconnect";
    els.disconnectButton.hidden = false;
  } else {
    els.statusText.textContent = status.configured
      ? "Not connected yet."
      : "Add a client ID and secret below, then connect.";
    els.connectButton.hidden = !status.configured;
    els.disconnectButton.hidden = true;
  }
  if (status.error) {
    els.connectError.textContent = status.error;
    els.connectError.hidden = false;
  } else {
    els.connectError.hidden = true;
  }
  onStatusChanged?.(status);
}

async function refresh(els: GoogleModalElements): Promise<GoogleStatus | null> {
  try {
    const status = await getGoogleStatus();
    render(els, status);
    return status;
  } catch (err) {
    els.connectError.textContent = (err as Error).message;
    els.connectError.hidden = false;
    return null;
  }
}

function pollWhileConnecting(els: GoogleModalElements): void {
  stopPolling();
  const tick = () => {
    void (async () => {
      const status = await refresh(els);
      if (status?.connecting) pollTimer = setTimeout(tick, POLL_INTERVAL_MS);
    })();
  };
  pollTimer = setTimeout(tick, POLL_INTERVAL_MS);
}

/** Opens the modal and refreshes its status. `onSaved`, if given, fires
 * every time the connection status actually changes (connected, error, or
 * disconnected) -- used by the Tools tab to keep its own summary in sync
 * without polling twice. */
export function openGoogleCalendarModal(onChanged?: (status: GoogleStatus) => void): void {
  const els = getElements();
  if (!els) return;
  onStatusChanged = onChanged ?? null;
  els.credentialsError.hidden = true;
  els.connectError.hidden = true;
  els.clientId.value = "";
  els.clientSecret.value = "";
  els.dialog.showModal();
  void refresh(els).then((status) => {
    if (status?.connecting) pollWhileConnecting(els);
  });
}

export async function initGoogleCalendarModal(): Promise<void> {
  const els = getElements();
  if (!els) return;

  els.close.addEventListener("click", () => {
    stopPolling();
    els.dialog.close();
  });
  els.dialog.addEventListener("close", stopPolling);

  els.credentialsForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const clientId = els.clientId.value.trim();
    const clientSecret = els.clientSecret.value.trim();
    if (!clientId || !clientSecret) return;

    els.credentialsError.hidden = true;
    const saveButton = els.credentialsForm.querySelector<HTMLButtonElement>("button[type=submit]");
    if (saveButton) saveButton.disabled = true;

    void (async () => {
      try {
        const status = await saveGoogleCredentials(clientId, clientSecret);
        els.clientId.value = "";
        els.clientSecret.value = "";
        render(els, status);
      } catch (err) {
        els.credentialsError.textContent = (err as Error).message;
        els.credentialsError.hidden = false;
      } finally {
        if (saveButton) saveButton.disabled = false;
      }
    })();
  });

  els.connectButton.addEventListener("click", () => {
    els.connectError.hidden = true;
    els.connectButton.disabled = true;
    void (async () => {
      try {
        const { auth_url } = await connectGoogleCalendar();
        // First (and so far only) use of the Tauri opener plugin in this
        // app -- the consent screen has to run in the system browser, not
        // this app's own webview, both because Google refuses to render it
        // in an embedded webview and because the whole point is picking a
        // real signed-in browser session.
        await openUrl(auth_url);
        await refresh(els);
        pollWhileConnecting(els);
      } catch (err) {
        els.connectError.textContent = (err as Error).message;
        els.connectError.hidden = false;
      } finally {
        els.connectButton.disabled = false;
      }
    })();
  });

  els.disconnectButton.addEventListener("click", () => {
    stopPolling();
    els.disconnectButton.disabled = true;
    void (async () => {
      try {
        const status = await disconnectGoogleCalendar();
        render(els, status);
      } catch (err) {
        els.connectError.textContent = (err as Error).message;
        els.connectError.hidden = false;
      } finally {
        els.disconnectButton.disabled = false;
      }
    })();
  });

  await refresh(els);
}
