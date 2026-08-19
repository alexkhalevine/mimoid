import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from . import config, ollama
from .chunking import chunk_text

MEMORIES_COLLECTION = "memories"
VAULT_COLLECTION = "historical_vault"
STYLE_COLLECTION = "style_corpus"

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
        # The vault gets cosine space so its distances live on a fixed [0, 2]
        # scale that a relevance threshold (VAULT_MAX_DISTANCE) can be applied
        # to; the memories collection keeps Chroma's default (L2) space it has
        # always had, so existing personal retrieval behavior is unchanged.
        metadata = {"hnsw:space": "cosine"} if name == VAULT_COLLECTION else None
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


async def search_memories(
    query: str,
    top_k: int = config.MEMORY_TOP_K,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    if query_embedding is None:
        query_embedding = await ollama.embed(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
    )

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    return [
        {"content": document, "metadata": metadata}
        for document, metadata in zip(documents[0], metadatas[0], strict=True)
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


def format_memories_section(memories: list[dict]) -> str:
    """Formats retrieved memories as a system-prompt section. Empty string
    if there's nothing relevant -- persona.py skips blank sections.

    Framed assertively -- these are the twin's OWN real memories to answer
    from, not optional trivia. The earlier "use these if helpful, don't force
    them in if irrelevant" wording actively licensed a small local model to
    ignore relevant memories; this tells it to draw on them when the question
    touches its life, while still allowing it to skip genuinely unrelated ones
    (retrieval is unfiltered top-K, so some may not fit the question)."""
    if not memories:
        return ""

    memory_lines = "\n".join(f"- {memory['content']}" for memory in memories)
    return (
        "These are your own real memories and experiences -- things that "
        "actually happened to you. When the question touches your life, draw on "
        "them and answer from them, in the first person as yourself. Only leave "
        f"out ones that are genuinely unrelated to what's being asked:\n{memory_lines}"
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
