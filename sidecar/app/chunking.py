DEFAULT_CHUNK_CHARS = 800
DEFAULT_OVERLAP_CHARS = 100


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Splits text into overlapping chunks of at most max_chars, breaking on
    whitespace where possible. Most user-entered memories are short enough to
    fit in a single chunk; this only kicks in for longer entries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks
