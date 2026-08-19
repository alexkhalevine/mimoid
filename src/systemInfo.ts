import { getSidecarConfig, type SidecarConfig } from "./sidecar";

interface ComponentCard {
  title: string;
  badge: "local" | "cloud";
  badgeLabel: string;
  value: (config: SidecarConfig) => string;
  description: string;
}

const CARDS: ComponentCard[] = [
  {
    title: "Brain (LLM)",
    badge: "local",
    badgeLabel: "LOCAL",
    value: (c) => c.model,
    description: "Generates every reply in Talk mode. Runs entirely on this machine via Ollama.",
  },
  {
    title: "Memory embeddings",
    badge: "local",
    badgeLabel: "LOCAL",
    value: (c) => c.embedding_model,
    description:
      "Turns your memories and style examples into vectors so the twin can pull in the most relevant ones for each question (stored in ChromaDB).",
  },
  {
    title: "Local model runtime",
    badge: "local",
    badgeLabel: "LOCAL",
    value: () => "Ollama",
    description:
      "Runs the LLM and embedding model above — nothing about a chat or memory ever leaves this machine.",
  },
  {
    title: "Speech-to-text",
    badge: "local",
    badgeLabel: "LOCAL",
    value: (c) => c.whisper_model,
    description:
      "Transcribes your voice input to text in Talk mode — Whisper on Apple's MLX on Apple Silicon, on PyTorch (CUDA where there's an NVIDIA GPU) everywhere else.",
  },
  {
    title: "Voice cloning (TTS)",
    badge: "local",
    badgeLabel: "LOCAL",
    value: (c) => c.tts_model,
    description: "Clones your voice from recorded samples in Train mode, and speaks replies aloud in your voice.",
  },
  {
    title: "Local storage",
    badge: "local",
    badgeLabel: "LOCAL",
    value: () => "SQLite + ChromaDB",
    description:
      "Conversations, memories, voice samples, and style examples live in a local SQLite database; ChromaDB holds a separate embedded index for fast memory retrieval.",
  },
  {
    title: "Local style distillation",
    badge: "local",
    badgeLabel: "LOCAL",
    value: (c) => c.model,
    description:
      "Condenses writing samples into a compact style guide using the configured Ollama model. It runs entirely on this Mac.",
  },
  {
    title: "OpenRouter style distillation",
    badge: "cloud",
    badgeLabel: "CLOUD · OPTIONAL",
    value: (c) => c.openrouter_model,
    description:
      "An optional paid alternative (Train → Style → \"Distill with OpenRouter\") that sends writing examples to OpenRouter to create a compact style guide. It requires an API key.",
  },
];

function renderCard(card: ComponentCard, config: SidecarConfig): HTMLElement {
  const el = document.createElement("article");
  el.className = "system-card";

  const header = document.createElement("div");
  header.className = "system-card-header";

  const badge = document.createElement("span");
  badge.className = `system-card-badge system-card-badge--${card.badge}`;
  badge.textContent = card.badgeLabel;

  const title = document.createElement("h3");
  title.className = "system-card-title";
  title.textContent = card.title;

  const value = document.createElement("span");
  value.className = "system-card-value";
  value.textContent =
    card.title === "OpenRouter style distillation" && !config.openrouter_configured
      ? "Not configured"
      : card.value(config);

  header.append(badge, title, value);

  const desc = document.createElement("p");
  desc.className = "system-card-desc";
  desc.textContent = card.description;

  el.append(header, desc);
  return el;
}

export async function initSystemInfo(): Promise<void> {
  const view = document.querySelector<HTMLElement>("#system-view");
  const list = document.querySelector<HTMLElement>("#system-card-list");
  const errorEl = document.querySelector<HTMLElement>("#system-info-error");
  if (!view || !list || !errorEl) return;

  try {
    const config = await getSidecarConfig();
    list.innerHTML = "";
    for (const card of CARDS) {
      list.appendChild(renderCard(card, config));
    }
  } catch (err) {
    errorEl.textContent = (err as Error).message;
    errorEl.hidden = false;
  }
}
