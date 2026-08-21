import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from . import config, ollama
from .chunking import chunk_text

# Personal memories live in a cosine-space collection so search_memories()
# can apply a relevance floor (config.MEMORY_MAX_DISTANCE) on the same fixed
# [0, 2] scale the vault uses. The original "memories" collection was created
# in Chroma's default L2 space, whose distances are unbounded and
# magnitude-dependent -- no principled threshold exists on that scale, and
# Chroma can't change a collection's space in place. Hence a new name plus a
# one-time rebuild from SQLite (reindex_memories_to_cosine below);
# LEGACY_MEMORIES_COLLECTION is only known to that migration.
MEMORIES_COLLECTION = "memories_cosine"
LEGACY_MEMORIES_COLLECTION = "memories"
VAULT_COLLECTION = "historical_vault"
STYLE_COLLECTION = "style_corpus"
_COSINE_COLLECTIONS = {MEMORIES_COLLECTION, VAULT_COLLECTION}

_client: chromadb.ClientAPI | None = None
_collections: dict[str, Collection] = {}


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def disk_usage_bytes() -> int:
    """Total size on disk of the ChromaDB persistence directory (its sqlite
    index plus the per-collection HNSW segment files). Returns 0 before the
    directory exists (fresh install, nothing indexed yet)."""
    if not config.CHROMA_DIR.exists():
        return 0
    return sum(f.stat().st_size for f in config.CHROMA_DIR.rglob("*") if f.is_file())


def get_collection(name: str = MEMORIES_COLLECTION) -> Collection:
    if name not in _collections:
        # Cosine space puts distances on a fixed [0, 2] scale that a relevance
        # threshold can be applied to -- see _COSINE_COLLECTIONS above. The
        # style corpus stays on Chroma's default (L2) space: its retrieval is
        # unfiltered top-K by design (examples are voice reference, not facts,
        # so a loose match is harmless), and switching it would mean another
        # rebuild for no behavioral gain.
        metadata = {"hnsw:space": "cosine"} if name in _COSINE_COLLECTIONS else None
        _collections[name] = _get_client().get_or_create_collection(name, metadata=metadata)
    return _collections[name]


async def index_memory(memory_id: str, content: str, metadata: dict) -> None:
    """Chunks, embeds, and stores a memory. Replaces any existing chunks for
    this memory_id first, so re-indexing an edited memory doesn't leave
    stale chunks behind."""
    deindex_memory(memory_id)

    chunks = chunk_text(content)
    if not chunks:
        return

    embeddings = [await ollama.embed(chunk) for chunk in chunks]
    ids = [f"{memory_id}:{i}" for i in range(len(chunks))]
    metadatas = [{**metadata, "memory_id": memory_id, "chunk_index": i} for i in range(len(chunks))]

    get_collection().add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)


def deindex_memory(memory_id: str) -> None:
    collection = get_collection()
    if collection.count() == 0:
        return
    collection.delete(where={"memory_id": memory_id})


def needs_cosine_reindex() -> bool:
    """True when a legacy L2 memories collection still holds the only copy of
    the index. Cheap enough to call on every startup."""
    client = _get_client()
    existing = {collection.name for collection in client.list_collections()}
    if LEGACY_MEMORIES_COLLECTION not in existing:
        return False
    # Already rebuilt (or rebuilt far enough to be useful) -- the legacy
    # collection is dropped at the end of a successful rebuild, so finding
    # both means a previous attempt died partway.
    return get_collection().count() == 0


async def reindex_memories_to_cosine(memories: list[dict]) -> int:
    """Rebuilds the memories index in the cosine collection from SQLite, which
    is the source of truth (the Chroma index is derived and disposable -- see
    backup.py's archive note). Returns how many memories were indexed.

    Deliberately builds the new collection FIRST and only drops the legacy one
    once every memory is in. Embedding needs Ollama, which may be down at
    startup; failing partway then leaves the legacy collection untouched and
    the next launch simply tries again, rather than leaving the twin with no
    memory index at all. Callers treat any exception as "try again next
    launch" -- see main.py's lifespan."""
    if not memories:
        # Nothing to carry over, but a stale empty legacy collection should
        # still go, so this doesn't re-run forever on a fresh install.
        _drop_legacy_memories_collection()
        return 0

    indexed = 0
    for record in memories:
        metadata = {
            "topic": record.get("topic") or "",
            "occurred_at": record.get("occurred_at") or "",
        }
        await index_memory(record["id"], record["content"], metadata)
        indexed += 1

    _drop_legacy_memories_collection()
    return indexed


def _drop_legacy_memories_collection() -> None:
    client = _get_client()
    if LEGACY_MEMORIES_COLLECTION not in {c.name for c in client.list_collections()}:
        return
    client.delete_collection(LEGACY_MEMORIES_COLLECTION)
    _collections.pop(LEGACY_MEMORIES_COLLECTION, None)


async def search_memories(
    query: str,
    top_k: int = config.MEMORY_TOP_K,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """Semantic search over personal memories, keeping only results within
    MEMORY_MAX_DISTANCE.

    That relevance floor is what lets the twin admit it doesn't know
    something. This used to be unfiltered top-K: every question pulled back
    MEMORY_TOP_K memories whenever the collection was non-empty, and
    format_memories_section() then presented them as things that actually
    happened -- so a question about something never recorded still arrived
    with four unrelated pieces of a real life attached, and the model
    answered from them. Returning [] here is a real answer, not a failure:
    it means nothing stored is relevant, and the prompt says so explicitly."""
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    if query_embedding is None:
        query_embedding = await ollama.embed(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    return [
        {"content": document, "metadata": metadata, "distance": distance}
        for document, metadata, distance in zip(documents, metadatas, distances, strict=True)
        if distance <= config.MEMORY_MAX_DISTANCE
    ]


async def index_vault_document(doc_id: str, title: str, content: str) -> int:
    """Chunks, embeds, and stores a world-knowledge document in the vault
    collection. Returns the number of chunks indexed. Replace semantics like
    index_memory, so re-uploading a document never leaves stale chunks."""
    deindex_vault_document(doc_id)

    chunks = chunk_text(content, config.VAULT_CHUNK_CHARS, config.VAULT_OVERLAP_CHARS)
    if not chunks:
        return 0

    embeddings = [await ollama.embed(chunk) for chunk in chunks]
    ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "title": title, "chunk_index": i} for i in range(len(chunks))]

    get_collection(VAULT_COLLECTION).add(
        ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
    )
    return len(chunks)


def deindex_vault_document(doc_id: str) -> None:
    collection = get_collection(VAULT_COLLECTION)
    if collection.count() == 0:
        return
    collection.delete(where={"doc_id": doc_id})


async def search_vault(
    query: str,
    top_k: int = config.VAULT_TOP_K,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """Semantic search over the historical vault, keeping only results within
    VAULT_MAX_DISTANCE. That relevance gate is what routes: a personal
    question lands far from all history chunks and gets nothing back, while a
    history question lands close and pulls in real material -- no separate
    routing model needed."""
    collection = get_collection(VAULT_COLLECTION)
    count = collection.count()
    if count == 0:
        return []

    if query_embedding is None:
        query_embedding = await ollama.embed(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    return [
        {"content": document, "metadata": metadata, "distance": distance}
        for document, metadata, distance in zip(documents, metadatas, distances, strict=True)
        if distance <= config.VAULT_MAX_DISTANCE
    ]


async def index_style_entry(entry_id: str, content: str, kind: str, prompt: str | None) -> None:
    """Indexes one style entry as a single embedding -- deliberately not
    chunked, unlike memories/vault documents: a style entry retrieved as a
    few-shot demonstration needs to stay one complete, coherent example, not
    a fragment. Replace semantics like index_memory, so re-editing an entry
    doesn't leave a stale duplicate behind.

    For Q&A entries the *question* (prompt) is embedded, not the answer. This
    aligns retrieval with how the index is searched: the user's incoming
    question is compared against each entry to find the most topically
    relevant examples, so question-to-question similarity is what we want --
    not question-to-answer similarity. Writing-sample entries (kind="text")
    embed the content as before."""
    deindex_style_entry(entry_id)
    text_to_embed = prompt if (kind == "qa" and prompt) else content
    embedding = await ollama.embed(text_to_embed)
    get_collection(STYLE_COLLECTION).add(
        ids=[entry_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[{"kind": kind, "prompt": prompt or ""}],
    )


def deindex_style_entry(entry_id: str) -> None:
    collection = get_collection(STYLE_COLLECTION)
    if collection.count() == 0:
        return
    collection.delete(ids=[entry_id])


async def search_style_examples(
    query: str,
    top_k: int = config.STYLE_FEWSHOT_TOP_K,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """Finds style entries most relevant to `query`, shaped as
    {kind, prompt, content} -- the same shape persona.py's
    _format_style_examples() already expects for the raw-corpus fallback, so
    the few-shot section reuses that formatter directly."""
    collection = get_collection(STYLE_COLLECTION)
    count = collection.count()
    if count == 0:
        return []

    if query_embedding is None:
        query_embedding = await ollama.embed(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    return [
        {"kind": metadata.get("kind", "text"), "prompt": metadata.get("prompt") or None, "content": document}
        for document, metadata in zip(documents, metadatas, strict=True)
    ]


NO_RELEVANT_MEMORIES_SECTION = (
    "You have no stored memories that relate to this question. Ordinary "
    "conversation, opinions, and general knowledge are all still fine to "
    "answer normally. What you must not do is state a specific personal fact, "
    "name, date, or event about your own life as though you remembered it -- "
    "if that's what's being asked for, say plainly that you don't know or "
    "don't recall it."
)

MEMORY_LOOKUP_UNAVAILABLE_SECTION = (
    "Your recall isn't reachable this turn, so you have nothing in front of "
    "you to answer from. Talk normally, but don't state any specific personal "
    "fact, name, date, or event about your own life -- if asked for one, say "
    "you can't bring it to mind right now. Say that as yourself; don't "
    "explain it as a technical problem."
)


def format_memories_section(memories: list[dict]) -> str:
    """Formats retrieved memories as a system-prompt section.

    Framed assertively -- these are the twin's OWN real memories to answer
    from, not optional trivia. The earlier "use these if helpful, don't force
    them in if irrelevant" wording actively licensed a small local model to
    ignore relevant memories, so the assertive framing has to stay.

    What changed alongside the relevance floor in search_memories(): back when
    retrieval was unfiltered top-K, this section had to hedge ("only leave out
    ones that are genuinely unrelated") because some of what it listed
    genuinely didn't fit the question -- while simultaneously calling all of it
    things that actually happened. Now that what arrives is actually relevant,
    the hedge is gone and the boundary is explicit instead: these are the only
    personal history available, and anything outside them is a "don't know".

    An empty list is a real signal, not an absence: it means the relevance
    floor filtered everything out, and the twin is told so explicitly rather
    than the section silently vanishing from the prompt (persona.py used to
    skip blank sections, which read to the model as no constraint at all)."""
    if not memories:
        return NO_RELEVANT_MEMORIES_SECTION

    memory_lines = "\n".join(f"- {memory['content']}" for memory in memories)
    return (
        "These are your own real memories and experiences -- things that "
        "actually happened to you, picked out because they relate to what's "
        "being asked. Draw on them and answer from them, in the first person "
        "as yourself. They are also the only personal history you have in "
        "front of you right now: if the question asks for a personal detail "
        "these don't cover, say plainly that you don't know or don't recall "
        f"it instead of filling the gap with something that sounds right:\n{memory_lines}"
    )


def format_history_section(snippets: list[dict]) -> str:
    """Formats retrieved world-knowledge snippets as a system-prompt section.
    Empty string if nothing survived the relevance gate.

    Framed as things the owner themself knows and would retell, not as
    citations to quote -- the original "reference notes... stick to these
    facts" wording instructed factual fidelity but said nothing about voice,
    so the model tended to paraphrase chunks in textbook prose instead of
    the owner's own."""
    if not snippets:
        return ""

    snippet_lines = "\n".join(f"- {snippet['content']}" for snippet in snippets)
    return (
        "Things you know about world history, relevant here (use them when "
        "the question touches world events). Retell them in your own words "
        "and your own voice, like you're explaining it to a friend -- never "
        f"quote them verbatim or sound like an encyclopedia:\n{snippet_lines}"
    )
