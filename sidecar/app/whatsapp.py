"""Parses WhatsApp chat exports ("Export chat" -> .txt, optionally zipped)
into style corpus entries. Pure functions, no I/O beyond reading the bytes
handed in -- keeps it easy to unit-test."""
import io
import re
import zipfile

# Structural line-start patterns. We deliberately do NOT parse the date
# itself -- date formats vary by locale (d/m/y vs m/d/y, 12h vs 24h, `.`
# vs `/` separators) and getting that right per-locale is fragile. What's
# stable across every export is the shape: a timestamp, then " - " (Android)
# or a "[...]" bracket (iOS), then "Sender: message".
_ANDROID_START = re.compile(
    r"^\d{1,4}[./-]\d{1,4}[./-]\d{1,4},?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s?[apAP]\.?[mM]\.?)?\s+-\s+(?P<rest>.*)$"
)
_IOS_START = re.compile(r"^\[[^\]]{6,}\]\s+(?P<rest>.*)$")

# Bodies that are media placeholders or deleted messages -- not real text.
_SKIP_BODIES = {
    "<media omitted>",
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "gif omitted",
    "document omitted",
    "contact card omitted",
    "this message was deleted",
    "you deleted this message",
    "null",
}

_URL_ONLY = re.compile(r"^https?://\S+$")

# iOS prefixes many lines with these bidirectional marks; strip them so the
# structural regexes and body comparisons see clean text. Written as escapes
# (LEFT-TO-RIGHT MARK, RIGHT-TO-LEFT MARK, LEFT-TO-RIGHT/RIGHT-TO-LEFT
# EMBEDDING, POP DIRECTIONAL FORMATTING) rather than literal invisible
# characters, so the source stays legible instead of hiding control
# characters inline.
_BIDI_MARKS = str.maketrans("", "", "\u200e\u200f\u202a\u202b\u202c")

MIN_CONTENT_CHARS = 25
STANDALONE_MIN_CHARS = 100
DEFAULT_CAP = 100


def extract_chat_text(filename: str, data: bytes) -> str:
    """Returns the chat .txt text from an uploaded export, unzipping if
    needed. Raises ValueError if no usable text file is found."""
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            chat_name = next((n for n in names if n.endswith("_chat.txt")), None)
            if chat_name is None:
                chat_name = next((n for n in names if n.lower().endswith(".txt")), None)
            if chat_name is None:
                raise ValueError("The zip doesn't contain a chat .txt file.")
            data = archive.read(chat_name)
    return data.decode("utf-8-sig", errors="replace")


def _clean(line: str) -> str:
    return line.translate(_BIDI_MARKS).rstrip("\n")


def parse_export(text: str) -> list[dict]:
    """Parses raw export text into ordered {sender, text} messages.
    System messages (no "Sender: " after the timestamp) and media/deleted
    bodies are skipped; unrecognized lines are treated as continuations of
    the previous message."""
    messages: list[dict] = []

    for raw_line in text.split("\n"):
        line = _clean(raw_line)
        match = _ANDROID_START.match(line) or _IOS_START.match(line)

        if not match:
            # Continuation of the previous message (multi-line body).
            if messages and line.strip():
                messages[-1]["text"] += "\n" + line
            continue

        rest = match.group("rest")
        sender, sep, body = rest.partition(": ")
        if not sep:
            # Timestamped line with no "Sender: " -- a system message
            # (encryption notice, "X added Y", etc.). Skip it.
            continue

        body = body.strip()
        if (
            body.lower() in _SKIP_BODIES
            or body.startswith(("\u200e", "<attached:"))
            or _URL_ONLY.match(body)
        ):
            continue

        messages.append({"sender": sender.strip(), "text": body})

    return messages


def participants(messages: list[dict]) -> list[dict]:
    counts: dict[str, int] = {}
    for message in messages:
        counts[message["sender"]] = counts.get(message["sender"], 0) + 1
    return [
        {"name": name, "message_count": count}
        for name, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    ]


def _merge_bursts(messages: list[dict]) -> list[dict]:
    """WhatsApp conversations are bursty -- someone sends several messages
    in a row. Merge consecutive same-sender messages into one so a burst
    reads as a single turn."""
    merged: list[dict] = []
    for message in messages:
        if merged and merged[-1]["sender"] == message["sender"]:
            merged[-1]["text"] += "\n" + message["text"]
        else:
            merged.append({"sender": message["sender"], "text": message["text"]})
    return merged


def _is_low_quality(content: str) -> bool:
    if len(content) < MIN_CONTENT_CHARS:
        return True
    lines = [line for line in content.splitlines() if line.strip()]
    return bool(lines) and all(_URL_ONLY.match(line.strip()) for line in lines)


def _collect_candidates(
    messages: list[dict], me: str, seen: set[str], existing: set[str]
) -> list[dict]:
    """Maps one conversation's messages to candidate style-corpus entries
    for the participant `me`: a reply to someone else becomes a `qa`
    example (their message -> my reply), a standalone message of mine
    becomes a `text` sample. Applies quality filters and dedupes against
    `seen` (mutated in place, so callers can share it across conversations)
    and `existing`. Not specific to WhatsApp -- reused by signal.py."""
    merged = _merge_bursts(messages)
    candidates: list[dict] = []

    for index, message in enumerate(merged):
        if message["sender"] != me:
            continue

        content = message["text"].strip()
        if _is_low_quality(content):
            continue

        normalized = content.lower()
        if normalized in seen or normalized in existing:
            continue

        prev = merged[index - 1] if index > 0 else None
        if prev is not None and prev["sender"] != me:
            prompt = prev["text"].strip()
            candidates.append({"kind": "qa", "prompt": prompt, "content": content, "_index": index})
            seen.add(normalized)
        elif len(content) >= STANDALONE_MIN_CHARS:
            candidates.append({"kind": "text", "prompt": None, "content": content, "_index": index})
            seen.add(normalized)

    return candidates


def _cap_and_reorder(candidates: list[dict], cap: int) -> list[dict]:
    """Keeps the `cap` longest candidates, then restores chronological
    order (by original `_index`) so entries read naturally. Not specific
    to WhatsApp -- reused by signal.py."""
    candidates.sort(key=lambda c: (len(c["content"]), -c["_index"]), reverse=True)
    kept = candidates[:cap]
    kept.sort(key=lambda c: c["_index"])
    for entry in kept:
        del entry["_index"]
    return kept


def select_entries(
    messages: list[dict],
    me: str,
    cap: int = DEFAULT_CAP,
    existing_contents: set[str] | None = None,
) -> list[dict]:
    """Maps parsed messages to style corpus entries for the participant
    `me`. See `_collect_candidates` / `_cap_and_reorder` for the mapping
    and capping rules."""
    existing = set(existing_contents or set())
    seen: set[str] = set()
    candidates = _collect_candidates(messages, me, seen, existing)
    return _cap_and_reorder(candidates, cap)
