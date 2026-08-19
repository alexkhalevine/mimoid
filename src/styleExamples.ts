import type { StyleEntry } from "./sidecar";
import { deriveRowText, formatRelativeDay, matchesQuery, typeLabel } from "./styleFormat";

type TypeFilter = "all" | "text" | "qa";

const TYPE_FILTER_CYCLE: TypeFilter[] = ["all", "qa", "text"];
const TYPE_FILTER_LABEL: Record<TypeFilter, string> = { all: "All types", qa: "Q&A only", text: "Writing only" };

interface StyleExamplesElements {
  search: HTMLInputElement;
  typeFilter: HTMLButtonElement;
  sortButton: HTMLButtonElement;
  addButton: HTMLButtonElement;
  listError: HTMLElement;
  list: HTMLElement;
}

function getElements(): StyleExamplesElements | null {
  const search = document.querySelector<HTMLInputElement>("#style-search");
  const typeFilter = document.querySelector<HTMLButtonElement>("#style-type-filter");
  const sortButton = document.querySelector<HTMLButtonElement>("#style-sort");
  const addButton = document.querySelector<HTMLButtonElement>("#style-add");
  const listError = document.querySelector<HTMLElement>("#style-list-error");
  const list = document.querySelector<HTMLElement>("#style-list");
  if (!search || !typeFilter || !sortButton || !addButton || !listError || !list) return null;
  return { search, typeFilter, sortButton, addButton, listError, list };
}

export interface StyleExamplesCallbacks {
  getEntries: () => StyleEntry[];
  onOpenNew: () => void;
  onOpenEntry: (id: string) => void;
}

export interface StyleExamplesView {
  render: () => void;
}

function renderRow(entry: StyleEntry, onOpenEntry: (id: string) => void): HTMLElement {
  const { subject } = deriveRowText(entry);
  const row = document.createElement("button");
  row.type = "button";
  row.className = "style-row";
  row.dataset.id = entry.id;

  const type = document.createElement("span");
  type.className = "style-row-type";
  type.textContent = typeLabel(entry.kind);

  const subjectEl = document.createElement("span");
  subjectEl.className = "style-row-subject";
  subjectEl.textContent = subject;

  const date = document.createElement("span");
  date.className = "style-row-date";
  date.textContent = formatRelativeDay(entry.updated_at);

  row.append(type, subjectEl, date);
  row.addEventListener("click", () => onOpenEntry(entry.id));
  return row;
}

export function initStyleExamples(callbacks: StyleExamplesCallbacks): StyleExamplesView | null {
  const els = getElements();
  if (!els) return null;

  let query = "";
  let typeFilter: TypeFilter = "all";
  let sortDesc = true;

  const render = () => {
    // #style-list-error is a fetch-failure banner owned by the orchestrator
    // (it knows whether the last listStyleEntries() call succeeded) --
    // render() only filters/sorts/displays whatever entries it's given, so
    // it must never blindly clear that banner itself.
    let entries = callbacks.getEntries();

    if (typeFilter !== "all") entries = entries.filter((e) => e.kind === typeFilter);
    if (query) entries = entries.filter((e) => matchesQuery(e, query));

    entries = [...entries].sort((a, b) => {
      const diff = Date.parse(a.created_at) - Date.parse(b.created_at);
      return sortDesc ? -diff : diff;
    });

    els.list.innerHTML = "";
    if (entries.length === 0) {
      const empty = document.createElement("p");
      empty.className = "memory-list-empty";
      empty.textContent = query || typeFilter !== "all" ? "No matching examples." : "No style examples yet.";
      els.list.appendChild(empty);
      return;
    }
    for (const entry of entries) {
      els.list.appendChild(renderRow(entry, callbacks.onOpenEntry));
    }
  };

  let searchTimer: ReturnType<typeof setTimeout> | undefined;
  els.search.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      query = els.search.value.trim();
      render();
    }, 200);
  });

  els.typeFilter.addEventListener("click", () => {
    const nextIndex = (TYPE_FILTER_CYCLE.indexOf(typeFilter) + 1) % TYPE_FILTER_CYCLE.length;
    typeFilter = TYPE_FILTER_CYCLE[nextIndex];
    els.typeFilter.textContent = TYPE_FILTER_LABEL[typeFilter];
    render();
  });

  els.sortButton.addEventListener("click", () => {
    sortDesc = !sortDesc;
    els.sortButton.textContent = sortDesc ? "Newest first ↓" : "Oldest first ↑";
    render();
  });

  els.addButton.addEventListener("click", () => callbacks.onOpenNew());

  return { render };
}
