import { SIDECAR_BASE_URL } from "./config";

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
}

export interface OllamaModel {
  name: string;
}

export interface Memory {
  id: string;
  content: string;
  topic: string | null;
  source: string;
  occurred_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface MemoryInput {
  content: string;
  topic?: string;
  occurred_at?: string;
}

export interface VoiceSample {
  id: string;
  prompt: string;
  created_at: string;
}

export interface VaultDocument {
  id: string;
  title: string;
  filename: string;
  chunk_count: number;
  created_at: string;
}

export interface StyleEntry {
  id: string;
  kind: "text" | "qa";
  prompt: string | null;
  content: string;
  source: string;
  created_at: string;
  updated_at: string;
}

export interface StyleEntryInput {
  kind: "text" | "qa";
  content: string;
  prompt?: string;
  /** Defaults to "manual" server-side when omitted. "own-answer" tags
   * entries saved from the Persona check tab's composer -- text Alex wrote
   * himself, before seeing either generated duel answer. See main.py's
   * create_style_entry invariant comment: this table must stay
   * human-authored, so a model-generated source is never valid here. */
  source?: "manual" | "own-answer";
}

export interface StyleGuide {
  content: string;
  entry_count: number;
  updated_at: string;
}

export interface ImportBatch {
  id: string;
  source: "whatsapp" | "signal";
  filename: string;
  entry_count: number;
  created_at: string;
}

export interface PersonaDuelResult {
  id: string;
  a: string;
  b: string;
}

export interface PersonaArmStats {
  arm: string;
  label: string;
  wins: number;
  losses: number;
  ties: number;
  appearances: number;
  win_rate: number | null;
}

export interface PersonaDuelSummary {
  total: number;
  decided: number;
  tie: number;
  default_arm: string;
  default_arm_label: string;
  /** The default arm's win rate against whichever challenger it faced,
   * ties excluded from the denominator -- "is the setup live chat actually
   * uses winning?" */
  default_arm_rate: number | null;
  arms: PersonaArmStats[];
  /** Only present on the response to choosePersonaDuel() -- withheld from
   * createPersonaDuel() so the comparison stays blind until a choice is made. */
  arm_a?: string;
  arm_b?: string;
  /** Last up to 8 decided duels, chronological, for the scoreboard's mini
   * bar chart -- "default" (the default arm won), "challenger", or "tie". */
  recent: ("default" | "challenger" | "tie")[];
}

export type PullProgress = Record<string, unknown>;

async function readNdjsonStream(
  body: ReadableStream<Uint8Array>,
  onLine: (line: PullProgress) => void,
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.trim()) onLine(JSON.parse(line) as PullProgress);
    }
  }
}

export interface SidecarConfig {
  model: string;
  embedding_model: string;
  whisper_model: string;
  tts_model: string;
  openrouter_model: string;
  openrouter_configured: boolean;
}

export async function getSidecarConfig(): Promise<SidecarConfig> {
  const res = await fetch(`${SIDECAR_BASE_URL}/config`);
  if (!res.ok) throw new Error("failed to load sidecar config");
  return res.json();
}

export interface StorageInfo {
  chroma_bytes: number;
}

export async function getStorageInfo(): Promise<StorageInfo> {
  const res = await fetch(`${SIDECAR_BASE_URL}/storage`);
  if (!res.ok) throw new Error("failed to load storage info");
  return res.json();
}

export interface Capabilities {
  stt: boolean;
  tts: boolean;
}

export async function getCapabilities(): Promise<Capabilities> {
  const res = await fetch(`${SIDECAR_BASE_URL}/capabilities`);
  if (!res.ok) throw new Error("failed to load capabilities");
  return res.json();
}

export async function listModels(): Promise<OllamaModel[]> {
  const res = await fetch(`${SIDECAR_BASE_URL}/ollama/models`);
  if (!res.ok) throw new Error("failed to list models");
  const data = (await res.json()) as { models: OllamaModel[] };
  return data.models;
}

export async function pullModel(
  name: string,
  onProgress: (line: PullProgress) => void,
): Promise<void> {
  const res = await fetch(`${SIDECAR_BASE_URL}/ollama/pull`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok || !res.body) throw new Error("failed to pull model");
  await readNdjsonStream(res.body, onProgress);
}

export async function listConversations(): Promise<Conversation[]> {
  const res = await fetch(`${SIDECAR_BASE_URL}/conversations`);
  if (!res.ok) throw new Error("failed to list conversations");
  const data = (await res.json()) as { conversations: Conversation[] };
  return data.conversations;
}

export async function createConversation(): Promise<Conversation> {
  const res = await fetch(`${SIDECAR_BASE_URL}/conversations`, { method: "POST" });
  if (!res.ok) throw new Error("failed to create conversation");
  return res.json();
}

export async function resetConversation(): Promise<Conversation> {
  const res = await fetch(`${SIDECAR_BASE_URL}/conversations/reset`, { method: "POST" });
  if (!res.ok) throw new Error("failed to start a new conversation");
  return res.json();
}

export async function getMessages(conversationId: string): Promise<ChatMessage[]> {
  const res = await fetch(`${SIDECAR_BASE_URL}/conversations/${conversationId}/messages`);
  if (!res.ok) throw new Error("failed to load messages");
  const data = (await res.json()) as { messages: ChatMessage[] };
  return data.messages;
}

export async function sendMessage(
  conversationId: string,
  content: string,
  onToken: (token: string) => void,
  signal?: AbortSignal,
  onRetrievalCounts?: (memoryCount: number, historyCount: number) => void,
): Promise<{ turnStartedAt: string | null }> {
  const res = await fetch(`${SIDECAR_BASE_URL}/conversations/${conversationId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error("failed to send message");

  if (onRetrievalCounts) {
    const memoryCount = Number(res.headers.get("X-Memory-Count"));
    const historyCount = Number(res.headers.get("X-History-Count"));
    if (Number.isFinite(memoryCount) || Number.isFinite(historyCount)) {
      onRetrievalCounts(
        Number.isFinite(memoryCount) ? memoryCount : 0,
        Number.isFinite(historyCount) ? historyCount : 0,
      );
    }
  }

  // Server-issued, so it's directly comparable against stored tool-run
  // timestamps; a browser-generated one would differ in ISO spelling and
  // sort wrong against them.
  const turnStartedAt = res.headers.get("X-Turn-Started");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    onToken(decoder.decode(value, { stream: true }));
  }

  return { turnStartedAt };
}

export interface ToolRun {
  id: string;
  conversation_id: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  result: string;
  created_at: string;
}

/** `after` is an ISO timestamp; the Talk tab passes the moment it sent the
 * message so it only sees tool runs from the reply it just received. */
export async function listToolRuns(conversationId: string, after?: string): Promise<ToolRun[]> {
  const url = new URL(`${SIDECAR_BASE_URL}/conversations/${conversationId}/tool-runs`);
  if (after) url.searchParams.set("after", after);
  const res = await fetch(url);
  if (!res.ok) throw new Error("failed to load tool runs");
  const data = (await res.json()) as { tool_runs: ToolRun[] };
  return data.tool_runs;
}

export interface WeatherLocation {
  name: string;
  latitude: number;
  longitude: number;
}

export interface ToolSettings {
  model: string;
  model_supports_tools: boolean;
  clock_enabled: boolean;
  timezone: string;
  weather_enabled: boolean;
  weather_locations: WeatherLocation[];
  calendar_enabled: boolean;
  calendar_confirm: boolean;
  calendar_configured: boolean;
  calendar_connected: boolean;
}

export async function getToolSettings(): Promise<ToolSettings> {
  const res = await fetch(`${SIDECAR_BASE_URL}/tools`);
  if (!res.ok) throw new Error("failed to load tool settings");
  return (await res.json()) as ToolSettings;
}

export async function updateToolSettings(patch: Partial<ToolSettings>): Promise<ToolSettings> {
  const res = await fetch(`${SIDECAR_BASE_URL}/tools`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to update tool settings"));
  return (await res.json()) as ToolSettings;
}

export interface GoogleStatus {
  configured: boolean;
  connected: boolean;
  connecting: boolean;
  error: string | null;
}

export async function getGoogleStatus(): Promise<GoogleStatus> {
  const res = await fetch(`${SIDECAR_BASE_URL}/tools/google/status`);
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to load Google Calendar status"));
  return (await res.json()) as GoogleStatus;
}

export async function saveGoogleCredentials(clientId: string, clientSecret: string): Promise<GoogleStatus> {
  const res = await fetch(`${SIDECAR_BASE_URL}/tools/google/credentials`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ client_id: clientId, client_secret: clientSecret }),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to save Google credentials"));
  return (await res.json()) as GoogleStatus;
}

/** Starts the loopback OAuth flow and returns the consent URL to open in
 * the system browser -- the sidecar's background listener then waits for
 * Google's redirect; poll getGoogleStatus() to learn when it lands. */
export async function connectGoogleCalendar(): Promise<{ auth_url: string }> {
  const res = await fetch(`${SIDECAR_BASE_URL}/tools/google/connect`, { method: "POST" });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to start connecting Google Calendar"));
  return (await res.json()) as { auth_url: string };
}

export async function disconnectGoogleCalendar(): Promise<GoogleStatus> {
  const res = await fetch(`${SIDECAR_BASE_URL}/tools/google/disconnect`, { method: "POST" });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to disconnect Google Calendar"));
  return (await res.json()) as GoogleStatus;
}

export interface PendingCalendarEvent {
  id: string;
  conversation_id: string;
  title: string;
  start_at: string;
  duration_minutes: number;
  description: string;
  created_at: string;
}

export async function listPendingCalendarEvents(
  conversationId: string,
  after?: string,
): Promise<PendingCalendarEvent[]> {
  const url = new URL(`${SIDECAR_BASE_URL}/conversations/${conversationId}/calendar/pending`);
  if (after) url.searchParams.set("after", after);
  const res = await fetch(url);
  if (!res.ok) throw new Error("failed to load pending calendar events");
  const data = (await res.json()) as { pending_events: PendingCalendarEvent[] };
  return data.pending_events;
}

export async function confirmPendingCalendarEvent(eventId: string): Promise<{ html_link: string }> {
  const res = await fetch(`${SIDECAR_BASE_URL}/calendar/pending/${eventId}/confirm`, { method: "POST" });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to create the calendar event"));
  return (await res.json()) as { html_link: string };
}

export async function discardPendingCalendarEvent(eventId: string): Promise<void> {
  const res = await fetch(`${SIDECAR_BASE_URL}/calendar/pending/${eventId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to discard the draft event"));
}

async function parseErrorDetail(res: Response, fallback: string): Promise<string> {
  try {
    const data = (await res.json()) as { detail?: string };
    return data.detail ?? fallback;
  } catch {
    return fallback;
  }
}

export async function listMemories(query?: string): Promise<Memory[]> {
  const url = new URL(`${SIDECAR_BASE_URL}/memories`);
  if (query) url.searchParams.set("q", query);
  const res = await fetch(url);
  if (!res.ok) throw new Error("failed to list memories");
  const data = (await res.json()) as { memories: Memory[] };
  return data.memories;
}

export async function createMemory(input: MemoryInput): Promise<Memory> {
  const res = await fetch(`${SIDECAR_BASE_URL}/memories`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to create memory"));
  return res.json();
}

export async function updateMemory(id: string, input: MemoryInput): Promise<Memory> {
  const res = await fetch(`${SIDECAR_BASE_URL}/memories/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to update memory"));
  return res.json();
}

export async function deleteMemory(id: string): Promise<void> {
  const res = await fetch(`${SIDECAR_BASE_URL}/memories/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to delete memory"));
}

export async function exportMemories(): Promise<Blob> {
  const res = await fetch(`${SIDECAR_BASE_URL}/memories/export`);
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to export memories"));
  return res.blob();
}

export async function exportConversation(conversationId: string): Promise<Blob> {
  const res = await fetch(`${SIDECAR_BASE_URL}/conversations/${conversationId}/export`);
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to export chat"));
  return res.blob();
}

export interface MemoryImportResult {
  created: number;
  skipped_duplicates: number;
  skipped_invalid: number;
}

export async function importMemories(file: File): Promise<MemoryImportResult> {
  const formData = new FormData();
  formData.append("file", file, file.name);
  const res = await fetch(`${SIDECAR_BASE_URL}/memories/import`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to import memories"));
  return res.json();
}

export async function listVoiceSamples(): Promise<VoiceSample[]> {
  const res = await fetch(`${SIDECAR_BASE_URL}/voice-samples`);
  if (!res.ok) throw new Error("failed to list voice samples");
  const data = (await res.json()) as { samples: VoiceSample[] };
  return data.samples;
}

export async function createVoiceSample(prompt: string, blob: Blob, extension: string): Promise<VoiceSample> {
  const formData = new FormData();
  formData.append("file", blob, `sample.${extension}`);
  formData.append("prompt", prompt);
  const res = await fetch(`${SIDECAR_BASE_URL}/voice-samples`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to save voice sample"));
  return res.json();
}

export async function deleteVoiceSample(id: string): Promise<void> {
  const res = await fetch(`${SIDECAR_BASE_URL}/voice-samples/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to delete voice sample"));
}

export function voiceSampleAudioUrl(id: string): string {
  return `${SIDECAR_BASE_URL}/voice-samples/${id}/audio`;
}

export async function listVaultDocuments(): Promise<VaultDocument[]> {
  const res = await fetch(`${SIDECAR_BASE_URL}/vault/documents`);
  if (!res.ok) throw new Error("failed to list documents");
  const data = (await res.json()) as { documents: VaultDocument[] };
  return data.documents;
}

export async function uploadVaultDocument(file: File): Promise<VaultDocument> {
  const formData = new FormData();
  formData.append("file", file, file.name);
  const res = await fetch(`${SIDECAR_BASE_URL}/vault/documents`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to import the document"));
  return res.json();
}

export async function deleteVaultDocument(id: string): Promise<void> {
  const res = await fetch(`${SIDECAR_BASE_URL}/vault/documents/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to delete the document"));
}

export async function listStyleEntries(): Promise<StyleEntry[]> {
  const res = await fetch(`${SIDECAR_BASE_URL}/style-entries`);
  if (!res.ok) throw new Error("failed to list style entries");
  const data = (await res.json()) as { entries: StyleEntry[] };
  return data.entries;
}

export async function createStyleEntry(input: StyleEntryInput): Promise<StyleEntry> {
  const res = await fetch(`${SIDECAR_BASE_URL}/style-entries`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to create style entry"));
  return res.json();
}

export async function updateStyleEntry(id: string, input: StyleEntryInput): Promise<StyleEntry> {
  const res = await fetch(`${SIDECAR_BASE_URL}/style-entries/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to update style entry"));
  return res.json();
}

export async function deleteStyleEntry(id: string): Promise<void> {
  const res = await fetch(`${SIDECAR_BASE_URL}/style-entries/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to delete style entry"));
}

export async function deleteStyleEntriesBySource(source: string): Promise<number> {
  const url = new URL(`${SIDECAR_BASE_URL}/style-entries`);
  url.searchParams.set("source", source);
  const res = await fetch(url, { method: "DELETE" });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to delete imported entries"));
  const data = (await res.json()) as { deleted: number };
  return data.deleted;
}

export interface WhatsappParticipant {
  name: string;
  message_count: number;
}

export interface WhatsappParseResult {
  upload_id: string;
  participants: WhatsappParticipant[];
  total_messages: number;
}

export interface WhatsappImportResult {
  created: number;
  qa_count: number;
  text_count: number;
}

export async function parseWhatsappExport(file: File): Promise<WhatsappParseResult> {
  const formData = new FormData();
  formData.append("file", file, file.name);
  const res = await fetch(`${SIDECAR_BASE_URL}/whatsapp/parse`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to read the WhatsApp export"));
  return res.json();
}

export async function importWhatsappMessages(uploadId: string, me: string): Promise<WhatsappImportResult> {
  const res = await fetch(`${SIDECAR_BASE_URL}/whatsapp/import`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ upload_id: uploadId, me }),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to import messages"));
  return res.json();
}

export interface SignalConversationSummary {
  name: string;
  message_count: number;
}

export interface SignalImportResult {
  created: number;
  qa_count: number;
  text_count: number;
  conversations: SignalConversationSummary[];
}

export async function importSignalChats(files: File[]): Promise<SignalImportResult> {
  const formData = new FormData();
  for (const file of files) formData.append("files", file, file.name);
  const res = await fetch(`${SIDECAR_BASE_URL}/signal/import`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to import Signal chats"));
  return res.json();
}

export async function getStyleGuide(): Promise<StyleGuide | null> {
  const res = await fetch(`${SIDECAR_BASE_URL}/style-guide`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error("failed to load style guide");
  return res.json();
}

export async function distillStyleGuide(): Promise<StyleGuide> {
  const res = await fetch(`${SIDECAR_BASE_URL}/style-guide/distill`, { method: "POST" });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to distill style guide"));
  return res.json();
}

export async function distillStyleGuideLocally(): Promise<StyleGuide> {
  const res = await fetch(`${SIDECAR_BASE_URL}/style-guide/distill/local`, { method: "POST" });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to distill style guide locally"));
  return res.json();
}

export async function deleteStyleGuide(): Promise<void> {
  const res = await fetch(`${SIDECAR_BASE_URL}/style-guide`, { method: "DELETE" });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to delete style guide"));
}

export async function listImportBatches(): Promise<ImportBatch[]> {
  const res = await fetch(`${SIDECAR_BASE_URL}/import-history`);
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to load import history"));
  const data = (await res.json()) as { batches: ImportBatch[] };
  return data.batches;
}

export async function createPersonaDuel(prompt: string): Promise<PersonaDuelResult> {
  const res = await fetch(`${SIDECAR_BASE_URL}/persona/duels`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to generate comparison"));
  return res.json();
}

export async function choosePersonaDuel(
  id: string,
  choice: "a" | "b" | "tie",
): Promise<PersonaDuelSummary> {
  const res = await fetch(`${SIDECAR_BASE_URL}/persona/duels/${id}/choice`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ choice }),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to record choice"));
  return res.json();
}

export async function getPersonaDuelSummary(): Promise<PersonaDuelSummary> {
  const res = await fetch(`${SIDECAR_BASE_URL}/persona/duel-summary`);
  if (!res.ok) throw new Error("failed to load duel summary");
  return res.json();
}

export interface BackupConfig {
  url: string;
  username: string;
  has_password: boolean;
  last_backup_at: string | null;
}

export interface BackupConfigInput {
  url: string;
  username: string;
  password?: string;
}

export interface BackupResult {
  uploaded_to: string;
  size_bytes: number;
  last_backup_at: string;
}

export async function getBackupConfig(): Promise<BackupConfig> {
  const res = await fetch(`${SIDECAR_BASE_URL}/backup/config`);
  if (!res.ok) throw new Error("failed to load backup config");
  return res.json();
}

export async function updateBackupConfig(input: BackupConfigInput): Promise<BackupConfig> {
  const res = await fetch(`${SIDECAR_BASE_URL}/backup/config`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to save backup settings"));
  return res.json();
}

export async function runBackup(): Promise<BackupResult> {
  const res = await fetch(`${SIDECAR_BASE_URL}/backup/run`, { method: "POST" });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "backup failed"));
  return res.json();
}

export async function downloadBackup(): Promise<Blob> {
  const res = await fetch(`${SIDECAR_BASE_URL}/backup/download`, { method: "POST" });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "backup download failed"));
  return res.blob();
}

export interface OpenRouterConfig {
  has_api_key: boolean;
}

export async function getOpenRouterConfig(): Promise<OpenRouterConfig> {
  const res = await fetch(`${SIDECAR_BASE_URL}/openrouter/config`);
  if (!res.ok) throw new Error("failed to load OpenRouter config");
  return res.json();
}

export async function saveOpenRouterApiKey(apiKey: string): Promise<OpenRouterConfig> {
  const res = await fetch(`${SIDECAR_BASE_URL}/openrouter/config`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey }),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res, "failed to save OpenRouter API key"));
  return res.json();
}
