// The Mimoid error catalog -- every user-facing error shown via the
// global error modal (errorModal.ts) comes from an entry here. The `code`
// is the primary thing a coder (or an AI agent looking at a bug report/
// screenshot) should search the codebase for: `grep -rn "SYM-CHAT-SEND"` in
// this file finds the copy, and in the call site finds the actual failure
// path. Titles are deliberately poetic (Solaris's sentient ocean, not a
// generic toast) -- bodies stay honest and non-destructive: what actually
// happened, and what was (or wasn't) lost.

export interface ErrorDefinition {
  code: string;
  title: string;
  body: string;
  /** Label for the primary recovery action (e.g. "Retry", "Reconnect").
   * Omitted entirely when there's nothing to retry -- the modal then only
   * shows Dismiss. */
  primaryLabel?: string;
}

export const ERROR_CATALOG = {
  "SYM-CHAT-SEND-FAILED": {
    code: "SYM-CHAT-SEND-FAILED",
    title: "Ollama went quiet",
    body: "The local model didn't respond in time. Nothing was lost — your message is safe, try again when it settles.",
    primaryLabel: "Retry",
  },
  "SYM-MEMORY-DELETE-FAILED": {
    code: "SYM-MEMORY-DELETE-FAILED",
    title: "That memory held on",
    body: "The delete didn't go through. It's still safe in your library, untouched — try again when you're ready.",
    primaryLabel: "Retry",
  },
  "SYM-STYLE-DELETE-FAILED": {
    code: "SYM-STYLE-DELETE-FAILED",
    title: "That example held on",
    body: "The delete didn't go through. It's still safe in your corpus, untouched — try again when you're ready.",
    primaryLabel: "Retry",
  },
  "SYM-CHAT-EXPORT-FAILED": {
    code: "SYM-CHAT-EXPORT-FAILED",
    title: "The transcript didn't come out",
    body: "Exporting this chat didn't go through. Nothing in the conversation was touched — try again when you're ready.",
    primaryLabel: "Retry",
  },
  "SYM-CALENDAR-CONNECT-FAILED": {
    code: "SYM-CALENDAR-CONNECT-FAILED",
    title: "Google didn't pick up",
    body: "Connecting to Google Calendar didn't finish. Nothing was saved — try connecting again.",
    primaryLabel: "Retry",
  },
  "SYM-CALENDAR-EVENT-FAILED": {
    code: "SYM-CALENDAR-EVENT-FAILED",
    title: "That event didn't land",
    body: "Creating the calendar event didn't go through. The draft is still here, untouched — try again when you're ready.",
    primaryLabel: "Retry",
  },
  "SYM-UNKNOWN": {
    code: "SYM-UNKNOWN",
    title: "Something surfaced wrong",
    body: "An unexpected disturbance — nothing was lost, but this one doesn't have a name yet.",
  },
} as const satisfies Record<string, ErrorDefinition>;

export type ErrorCode = keyof typeof ERROR_CATALOG;

export function getErrorDefinition(code: ErrorCode): ErrorDefinition {
  return ERROR_CATALOG[code] ?? ERROR_CATALOG["SYM-UNKNOWN"];
}
