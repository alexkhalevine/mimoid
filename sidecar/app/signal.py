"""Parses Signal Desktop chat exports (one .txt per conversation) into
style corpus entries. Pure functions, no I/O beyond reading the bytes
handed in -- keeps it easy to unit-test.

Unlike WhatsApp's export, Signal's already labels which side is the
export's owner ("From: You" vs "From: <contact>"), so there's no need for
a "which participant is me" step -- messages map to style entries in one
pass. The message->entry mapping itself (burst-merging, qa-vs-text
classification, quality filters, dedupe, cap-to-N-keep-longest) is shared
with the WhatsApp importer via `whatsapp._collect_candidates` /
`whatsapp._cap_and_reorder`, since none of that logic is WhatsApp-specific.
"""
import re

from . import whatsapp

ME = "You"

# A message header paragraph starts with "From:"; a bare "Type: ..."
# paragraph (no "From:") is a system event (keychange, etc.) with no body.
_HEADER_LINE = re.compile(r"^(From|Type|Sent|Received):\s?(.*)$")
_CONVERSATION_LINE = re.compile(r"^Conversation:\s*(.*?)\s*$")

_MESSAGE_TYPES = {"incoming", "outgoing"}


def _paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return [p for p in re.split(r"\n\s*\n", normalized) if p.strip()]


def _parse_header(paragraph: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in paragraph.split("\n"):
        match = _HEADER_LINE.match(line)
        if match:
            fields[match.group(1)] = match.group(2).strip()
    return fields


def conversation_name(text: str) -> str | None:
    """Reads the leading "Conversation: <name>" paragraph, if present."""
    paragraphs = _paragraphs(text)
    if not paragraphs:
        return None
    first_line = paragraphs[0].split("\n", 1)[0]
    match = _CONVERSATION_LINE.match(first_line)
    return match.group(1) or None if match else None


def parse_export(text: str) -> list[dict]:
    """Parses raw export text into ordered {sender, text} messages.
    A paragraph starting with "From:" is a message header; the next
    paragraph, if it isn't itself a header, is that message's body. A bare
    "Type: ..." paragraph (no "From:") is a system event -- skipped, no
    body follows. Only "incoming"/"outgoing" types are kept as messages."""
    paragraphs = _paragraphs(text)
    messages: list[dict] = []

    index = 0
    if paragraphs and paragraphs[0].startswith("Conversation:"):
        index = 1

    while index < len(paragraphs):
        paragraph = paragraphs[index]
        first_line = paragraph.split("\n", 1)[0]

        if not first_line.startswith("From:"):
            # Bare "Type: ..." system event, or unrecognized/stray content --
            # neither carries a body paragraph of its own.
            index += 1
            continue

        fields = _parse_header(paragraph)
        sender = fields.get("From", "").strip()
        msg_type = fields.get("Type", "")

        has_body = False
        if msg_type in _MESSAGE_TYPES and index + 1 < len(paragraphs):
            next_first_line = paragraphs[index + 1].split("\n", 1)[0]
            has_body = not next_first_line.startswith(("From:", "Type:"))

        if has_body:
            messages.append({"sender": sender, "text": paragraphs[index + 1].strip()})
            index += 2
        else:
            index += 1

    return messages


def select_entries_for_files(
    per_file_messages: list[list[dict]],
    cap: int = whatsapp.DEFAULT_CAP,
    existing_contents: set[str] | None = None,
) -> list[dict]:
    """Aggregates candidate style entries across multiple conversation
    files, sharing one dedupe set so the same line said in two exports
    only counts once, then applies the cap once over the pooled result
    (rather than per file, which could let each file contribute up to
    `cap` entries)."""
    existing = set(existing_contents or set())
    seen: set[str] = set()
    candidates: list[dict] = []

    # `_index` is only unique within one file's message list; offset each
    # file's candidates so the final chronological-order restore in
    # `_cap_and_reorder` can't interleave entries from different files.
    offset = 0
    for messages in per_file_messages:
        file_candidates = whatsapp._collect_candidates(messages, ME, seen, existing)
        for candidate in file_candidates:
            candidate["_index"] += offset
        candidates.extend(file_candidates)
        offset += len(messages) + 1

    return whatsapp._cap_and_reorder(candidates, cap)
