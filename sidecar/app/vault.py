"""Text extraction for world-knowledge ("historical vault") uploads.

Supports .md/.txt (plain decode) and .epub. EPUB is just a zip of XHTML
plus an OPF manifest, so it's parsed entirely with the standard library:
container.xml -> OPF path -> spine order -> tag-stripped chapter text.
"""

import posixpath
import zipfile
from html.parser import HTMLParser
from io import BytesIO
from pathlib import PurePosixPath
from xml.etree import ElementTree

TEXT_EXTENSIONS = {".md", ".txt"}

# Tags whose text content is never prose.
_SKIPPED_TAGS = {"script", "style"}
# Tags that imply a line break when stripped, so headings/paragraphs don't
# run together into one giant line.
_BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "tr", "blockquote"}


class VaultExtractionError(Exception):
    """Raised when an uploaded document can't be parsed into text."""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


def _strip_html(markup: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(markup)
    return extractor.text()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _extract_epub(data: bytes, fallback_title: str) -> tuple[str, str]:
    try:
        archive = zipfile.ZipFile(BytesIO(data))
    except zipfile.BadZipFile as err:
        raise VaultExtractionError("not a valid EPUB file (couldn't open as a zip)") from err

    with archive:
        try:
            container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
            opf_path = next(
                el.get("full-path")
                for el in container.iter()
                if _local_name(el.tag) == "rootfile" and el.get("full-path")
            )
            opf = ElementTree.fromstring(archive.read(opf_path))
        except (KeyError, StopIteration, ElementTree.ParseError) as err:
            raise VaultExtractionError("not a valid EPUB file (missing or broken manifest)") from err

        title = fallback_title
        for el in opf.iter():
            if _local_name(el.tag) == "title" and el.text and el.text.strip():
                title = el.text.strip()
                break

        manifest = {
            item.get("id"): item.get("href")
            for item in opf.iter()
            if _local_name(item.tag) == "item" and item.get("id") and item.get("href")
        }
        spine_hrefs = [
            manifest[ref.get("idref")]
            for ref in opf.iter()
            if _local_name(ref.tag) == "itemref" and ref.get("idref") in manifest
        ]

        opf_dir = str(PurePosixPath(opf_path).parent)
        chapters = []
        for href in spine_hrefs:
            entry = posixpath.normpath(posixpath.join(opf_dir, href)) if opf_dir != "." else href
            try:
                markup = archive.read(entry).decode("utf-8", errors="replace")
            except KeyError:
                continue  # manifest points at a missing file; skip that chapter
            chapter_text = _strip_html(markup)
            if chapter_text:
                chapters.append(chapter_text)

    return title, "\n\n".join(chapters)


def extract_text(filename: str, data: bytes) -> tuple[str, str]:
    """Returns (title, text) for an uploaded document, or raises
    VaultExtractionError with a user-presentable message."""
    path = PurePosixPath(filename)
    extension = path.suffix.lower()
    fallback_title = path.stem or "Untitled document"

    if extension in TEXT_EXTENSIONS:
        text = data.decode("utf-8", errors="replace")
        title = fallback_title
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip() or fallback_title
                break
            if stripped:
                break  # first real line isn't a heading; keep the filename
        result = title, text
    elif extension == ".epub":
        result = _extract_epub(data, fallback_title)
    else:
        raise VaultExtractionError(
            f"unsupported file type '{extension or filename}' -- upload .md, .txt, or .epub"
        )

    title, text = result
    if not text.strip():
        raise VaultExtractionError("no readable text found in the document")
    return title, text
