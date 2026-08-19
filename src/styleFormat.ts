// Pure, DOM-free helpers for the Style Examples table. Deliberately built
// with only indexOf/slice/lastIndexOf and a plain \s+ regex -- no
// lookbehind/lookahead/Intl.Segmenter -- since this ships to WebKitGTK on
// Linux dev machines as well as macOS/WebView2, and the portable version
// sidesteps the question of engine support entirely.

export type StyleEntryKind = "text" | "qa";

export interface DerivableEntry {
  kind: string;
  prompt: string | null;
  content: string;
}

export interface DerivedRow {
  subject: string;
  snippet: string;
}

// A row's SUBJECT column is capped at this many characters; MIN_SUBJECT is
// the earliest position a sentence-ending punctuation mark is allowed to
// count as a break (shorter, and "3.5" / "Hi." / most bare URLs would get
// mistaken for a full sentence).
const SUBJECT_MAX = 80;
const MIN_SUBJECT = 12;
// Only look for a natural break within this many leading characters --
// searching the whole (possibly huge) content for a rare match is wasted
// work once nothing suitable is going to be found early anyway.
const SEARCH_WINDOW = 160;

function inline(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

/** Index right after a natural break point in `window` (terminator
 * included), or -1 if none is found. A break is either the first newline,
 * or the first `.`/`!`/`?` followed by whitespace or end-of-string, at an
 * offset >= MIN_SUBJECT. */
function findBreak(window: string): number {
  const newlineIdx = window.indexOf("\n");
  if (newlineIdx !== -1) return newlineIdx;

  for (let i = MIN_SUBJECT; i < window.length; i++) {
    const ch = window[i];
    if (ch !== "." && ch !== "!" && ch !== "?") continue;
    const next = window[i + 1];
    if (next === undefined || /\s/.test(next)) return i + 1;
  }
  return -1;
}

/** Derives a table row's SUBJECT + snippet from a style entry. Q&A entries
 * with a real question use the question as the subject; everything else
 * (writing samples, and Q&A entries with no question) derives from the
 * content itself. */
export function deriveRowText(entry: DerivableEntry): DerivedRow {
  if (entry.kind === "qa" && entry.prompt && entry.prompt.trim()) {
    return { subject: inline(entry.prompt), snippet: inline(entry.content) };
  }

  const content = entry.content;
  if (!content.trim()) return { subject: "(empty)", snippet: "" };

  const breakIdx = findBreak(content.slice(0, SEARCH_WINDOW));

  if (breakIdx !== -1 && breakIdx <= SUBJECT_MAX) {
    return { subject: inline(content.slice(0, breakIdx)), snippet: inline(content.slice(breakIdx)) };
  }

  if (content.length <= SUBJECT_MAX) {
    // Short enough that the subject already shows everything -- repeating
    // it as a snippet too would just duplicate the same text twice.
    return { subject: inline(content), snippet: "" };
  }

  const head = content.slice(0, SUBJECT_MAX);
  const lastSpace = head.lastIndexOf(" ");
  const cut = lastSpace > 0 ? lastSpace : SUBJECT_MAX;
  return { subject: `${inline(content.slice(0, cut))}…`, snippet: inline(content) };
}

export function typeLabel(kind: string): string {
  return kind === "qa" ? "Q&A" : "Write";
}

/** Search matches raw prompt+content, not the derived subject -- otherwise
 * a hit past the 80-char subject cut would never surface. */
export function matchesQuery(entry: DerivableEntry, query: string): boolean {
  const trimmed = query.trim();
  if (!trimmed) return true;
  const haystack = `${entry.prompt ?? ""} ${entry.content}`.toLowerCase();
  return haystack.includes(trimmed.toLowerCase());
}

const WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function startOfDay(d: Date): number {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
}

/** Today / Yesterday / weekday name (2-6 days ago) / "Jul 12" (same year) /
 * "Dec 3, 2025" (older). Compares local calendar days, not elapsed
 * milliseconds, so a DST transition never miscounts a day boundary. `now`
 * is injectable so callers (and tests) can pin the clock. */
export function formatRelativeDay(iso: string, now: Date = new Date()): string {
  const date = new Date(iso);
  const dayDiff = Math.round((startOfDay(now) - startOfDay(date)) / 86_400_000);

  if (dayDiff === 0) return "Today";
  if (dayDiff === 1) return "Yesterday";
  if (dayDiff >= 2 && dayDiff <= 6) return WEEKDAYS[date.getDay()];
  if (date.getFullYear() === now.getFullYear()) return `${MONTHS[date.getMonth()]} ${date.getDate()}`;
  return `${MONTHS[date.getMonth()]} ${date.getDate()}, ${date.getFullYear()}`;
}
