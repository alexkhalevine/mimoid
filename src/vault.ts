import {
  deleteVaultDocument,
  listVaultDocuments,
  uploadVaultDocument,
  type VaultDocument,
} from "./sidecar";

function renderDocument(doc: VaultDocument, onChange: () => void): HTMLElement {
  const item = document.createElement("article");
  item.className = "memory-item";

  const meta = document.createElement("div");
  meta.className = "memory-item-meta";
  const date = doc.created_at ? doc.created_at.slice(0, 10) : "";
  meta.textContent = [doc.filename, `${doc.chunk_count} ${doc.chunk_count === 1 ? "chunk" : "chunks"}`, date]
    .filter(Boolean)
    .join(" · ");

  const content = document.createElement("p");
  content.className = "memory-item-content";
  content.textContent = doc.title;

  const actions = document.createElement("div");
  actions.className = "memory-item-actions";

  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.textContent = "Delete";
  deleteButton.addEventListener("click", () => {
    void (async () => {
      deleteButton.disabled = true;
      try {
        await deleteVaultDocument(doc.id);
        onChange();
      } catch (err) {
        deleteButton.disabled = false;
        console.error("failed to delete document", err);
      }
    })();
  });

  actions.append(deleteButton);
  item.append(meta, content, actions);
  return item;
}

async function refreshList(list: HTMLElement): Promise<void> {
  const documents = await listVaultDocuments();
  list.innerHTML = "";
  if (documents.length === 0) {
    const empty = document.createElement("p");
    empty.className = "memory-list-empty";
    empty.textContent = "No documents yet — your twin is running on its base knowledge alone.";
    list.appendChild(empty);
    return;
  }
  for (const doc of documents) {
    list.appendChild(renderDocument(doc, () => void refreshList(list)));
  }
}

export async function initVault(): Promise<void> {
  const panel = document.querySelector<HTMLElement>("#vault-panel");
  const form = document.querySelector<HTMLFormElement>("#vault-form");
  const fileInput = document.querySelector<HTMLInputElement>("#vault-file");
  const uploadButton = document.querySelector<HTMLButtonElement>("#vault-upload");
  const statusEl = document.querySelector<HTMLElement>("#vault-status");
  const errorEl = document.querySelector<HTMLElement>("#vault-error");
  const list = document.querySelector<HTMLElement>("#vault-list");
  if (!panel || !form || !fileInput || !uploadButton || !statusEl || !errorEl || !list) return;

  panel.hidden = false;

  const setStatus = (text: string | null) => {
    statusEl.textContent = text ?? "";
    statusEl.hidden = !text;
  };

  const setError = (text: string | null) => {
    errorEl.textContent = text ?? "";
    errorEl.hidden = !text;
  };

  fileInput.addEventListener("change", () => {
    uploadButton.disabled = !fileInput.files?.length;
    setError(null);
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const file = fileInput.files?.[0];
    if (!file) return;

    void (async () => {
      uploadButton.disabled = true;
      fileInput.disabled = true;
      setError(null);
      setStatus(`Indexing "${file.name}"… large books can take a few minutes.`);
      try {
        const doc = await uploadVaultDocument(file);
        setStatus(`Added "${doc.title}" (${doc.chunk_count} ${doc.chunk_count === 1 ? "chunk" : "chunks"}).`);
        form.reset();
        await refreshList(list);
      } catch (err) {
        setStatus(null);
        setError((err as Error).message);
      } finally {
        fileInput.disabled = false;
        uploadButton.disabled = !fileInput.files?.length;
      }
    })();
  });

  await refreshList(list);
}
