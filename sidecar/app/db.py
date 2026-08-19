import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    topic TEXT,
    source TEXT NOT NULL DEFAULT 'manual',
    occurred_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vault_documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    filename TEXT NOT NULL,
    content TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS voice_samples (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS style_entries (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('text', 'qa')),
    prompt TEXT,
    content TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS style_guide (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    content TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS import_batches (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL CHECK(source IN ('whatsapp', 'signal')),
    filename TEXT NOT NULL,
    entry_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Every tool the twin invokes on your behalf, kept as a durable record for
-- the same reason network_activity keeps its history: in an offline-first
-- app, the moments it reaches outside itself should never be invisible.
-- Also what the Talk tab reads back to show which tools a reply used.
CREATE TABLE IF NOT EXISTS tool_runs (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments TEXT NOT NULL,
    result TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

-- A calendar event the twin has drafted but not yet created, when
-- tools_calendar_confirm is on (the default). Alex confirms or discards it
-- from the Talk tab; nothing reaches Google until then. Rows are deleted on
-- both outcomes -- confirm and discard -- so this table only ever holds
-- events still awaiting a decision, which is exactly what the Talk tab
-- needs to poll for per turn.
CREATE TABLE IF NOT EXISTS pending_calendar_events (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    title TEXT NOT NULL,
    start_at TEXT NOT NULL,
    duration_minutes INTEGER NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS persona_eval (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    persona_side TEXT NOT NULL CHECK(persona_side IN ('a', 'b')),
    persona_answer TEXT NOT NULL,
    generic_answer TEXT NOT NULL,
    choice TEXT CHECK(choice IN ('a', 'b', 'tie')),
    created_at TEXT NOT NULL,
    decided_at TEXT
);

-- Replaces persona_eval's persona-vs-generic comparison (a test the persona
-- side always won, since a generic assistant gives itself away instantly)
-- with a blind duel between two persona system-prompt "arms". persona_eval
-- is left in place, unreferenced -- its rows are real history and dropping
-- a table isn't worth the migration risk -- but nothing new reads it.
CREATE TABLE IF NOT EXISTS persona_duels (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    language TEXT NOT NULL,
    arm_a TEXT NOT NULL,
    arm_b TEXT NOT NULL,
    model_a TEXT NOT NULL,
    model_b TEXT NOT NULL,
    answer_a TEXT NOT NULL,
    answer_b TEXT NOT NULL,
    choice TEXT CHECK(choice IN ('a', 'b', 'tie')),
    created_at TEXT NOT NULL,
    decided_at TEXT
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    category TEXT NOT NULL,
    prompt TEXT NOT NULL,
    reply TEXT NOT NULL,
    gates_failed_count INTEGER NOT NULL,
    gates_detail TEXT NOT NULL,
    judge_score INTEGER,
    judge_notes TEXT,
    model TEXT NOT NULL,
    judge_model TEXT,
    created_at TEXT NOT NULL
);
"""


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Schema changes that can't be expressed as `CREATE TABLE IF NOT
    EXISTS` (there's no migration framework here, so this is just a set of
    idempotent ALTER TABLEs guarded against already having run)."""
    try:
        conn.execute("ALTER TABLE style_entries ADD COLUMN source TEXT NOT NULL DEFAULT 'manual'")
    except sqlite3.OperationalError:
        pass  # column already exists

    try:
        conn.execute("ALTER TABLE conversations ADD COLUMN archived_at TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists

    try:
        conn.execute(
            "ALTER TABLE style_guide ADD COLUMN entry_count INTEGER NOT NULL DEFAULT 0"
        )
    except sqlite3.OperationalError:
        pass  # column already exists


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def create_conversation(title: str = "New conversation") -> dict:
    conversation_id = str(uuid.uuid4())
    created_at = _now()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at) VALUES (?, ?, ?)",
            (conversation_id, title, created_at),
        )
    return {"id": conversation_id, "title": title, "created_at": created_at}


def list_conversations() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM conversations "
            "WHERE archived_at IS NULL ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def archive_conversation(conversation_id: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE conversations SET archived_at = ? WHERE id = ?",
            (_now(), conversation_id),
        )


def conversation_exists(conversation_id: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
    return row is not None


def get_messages(conversation_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def add_message(conversation_id: str, role: str, content: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, _now()),
        )


def create_memory(
    content: str,
    topic: str | None = None,
    source: str = "manual",
    occurred_at: str | None = None,
    created_at: str | None = None,
) -> dict:
    """`created_at` is normally left to default to now, but importing a JSON
    export (see main.py's /memories/import) passes the original timestamp
    through so re-imported memories keep their real place in `created_at
    DESC` ordering instead of all bunching up at "just now". `updated_at`
    always reflects this row's actual insert time regardless -- it's not
    something an export round-trip should preserve."""
    memory_id = str(uuid.uuid4())
    now = _now()
    created_at = created_at or now
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO memories (id, content, topic, source, occurred_at, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (memory_id, content, topic, source, occurred_at, created_at, now),
        )
    return {
        "id": memory_id,
        "content": content,
        "topic": topic,
        "source": source,
        "occurred_at": occurred_at,
        "created_at": created_at,
        "updated_at": now,
    }


def get_memory(memory_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    return dict(row) if row else None


def list_memories(query: str | None = None) -> list[dict]:
    with get_connection() as conn:
        if query:
            rows = conn.execute(
                "SELECT * FROM memories WHERE content LIKE ? OR topic LIKE ? "
                "ORDER BY created_at DESC",
                (f"%{query}%", f"%{query}%"),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM memories ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def update_memory(
    memory_id: str,
    content: str,
    topic: str | None = None,
    occurred_at: str | None = None,
) -> dict | None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE memories SET content = ?, topic = ?, occurred_at = ?, updated_at = ? "
            "WHERE id = ?",
            (content, topic, occurred_at, _now(), memory_id),
        )
    return get_memory(memory_id)


def delete_memory(memory_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))


def list_memory_contents() -> set[str]:
    """Normalized (stripped, lowercased) content of every existing memory --
    used by /memories/import to skip re-adding a memory whose text is
    already present, the same normalization list_style_entry_contents_by_source
    uses for the WhatsApp/Signal importers."""
    with get_connection() as conn:
        rows = conn.execute("SELECT content FROM memories").fetchall()
    return {row["content"].strip().lower() for row in rows}


def create_vault_document(title: str, filename: str, content: str) -> dict:
    doc_id = str(uuid.uuid4())
    created_at = _now()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO vault_documents (id, title, filename, content, chunk_count, created_at) "
            "VALUES (?, ?, ?, ?, 0, ?)",
            (doc_id, title, filename, content, created_at),
        )
    return {
        "id": doc_id,
        "title": title,
        "filename": filename,
        "chunk_count": 0,
        "created_at": created_at,
    }


def list_vault_documents() -> list[dict]:
    # content deliberately excluded -- a single document can be a whole book,
    # and the listing only drives the browse UI.
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, title, filename, chunk_count, created_at FROM vault_documents "
            "ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_vault_document(doc_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM vault_documents WHERE id = ?", (doc_id,)).fetchone()
    return dict(row) if row else None


def set_vault_document_chunk_count(doc_id: str, chunk_count: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE vault_documents SET chunk_count = ? WHERE id = ?",
            (chunk_count, doc_id),
        )


def delete_vault_document(doc_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM vault_documents WHERE id = ?", (doc_id,))


def create_voice_sample(prompt: str, file_path: str) -> dict:
    sample_id = str(uuid.uuid4())
    created_at = _now()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO voice_samples (id, prompt, file_path, created_at) VALUES (?, ?, ?, ?)",
            (sample_id, prompt, file_path, created_at),
        )
    return {"id": sample_id, "prompt": prompt, "file_path": file_path, "created_at": created_at}


def list_voice_samples() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM voice_samples ORDER BY created_at ASC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_voice_sample(sample_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM voice_samples WHERE id = ?", (sample_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_voice_sample(sample_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM voice_samples WHERE id = ?", (sample_id,))


def create_style_entry(
    kind: str, content: str, prompt: str | None = None, source: str = "manual"
) -> dict:
    entry_id = str(uuid.uuid4())
    now = _now()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO style_entries (id, kind, prompt, content, source, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (entry_id, kind, prompt, content, source, now, now),
        )
    return {
        "id": entry_id,
        "kind": kind,
        "prompt": prompt,
        "content": content,
        "source": source,
        "created_at": now,
        "updated_at": now,
    }


def get_style_entry(entry_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM style_entries WHERE id = ?", (entry_id,)
        ).fetchone()
    return dict(row) if row else None


def list_style_entries() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM style_entries ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def update_style_entry(
    entry_id: str, kind: str, content: str, prompt: str | None = None
) -> dict | None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE style_entries SET kind = ?, prompt = ?, content = ?, updated_at = ? "
            "WHERE id = ?",
            (kind, prompt, content, _now(), entry_id),
        )
    return get_style_entry(entry_id)


def delete_style_entry(entry_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM style_entries WHERE id = ?", (entry_id,))


def delete_style_entries_by_source(source: str) -> int:
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM style_entries WHERE source = ?", (source,))
    return cursor.rowcount


def list_style_entry_contents_by_source(source: str) -> set[str]:
    """Normalized (stripped, lowercased) content of existing entries with
    the given source -- used by the WhatsApp importer to dedupe against
    content it already imported previously."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT content FROM style_entries WHERE source = ?", (source,)
        ).fetchall()
    return {row["content"].strip().lower() for row in rows}


def get_style_guide() -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM style_guide WHERE id = 1").fetchone()
    return dict(row) if row else None


def set_style_guide(content: str, entry_count: int = 0) -> dict:
    now = _now()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO style_guide (id, content, entry_count, updated_at) VALUES (1, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET content = excluded.content, "
            "entry_count = excluded.entry_count, updated_at = excluded.updated_at",
            (content, entry_count, now),
        )
    return {"content": content, "entry_count": entry_count, "updated_at": now}


def delete_style_guide() -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM style_guide WHERE id = 1")


def create_import_batch(source: str, filename: str, entry_count: int) -> dict:
    batch_id = str(uuid.uuid4())
    created_at = _now()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO import_batches (id, source, filename, entry_count, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (batch_id, source, filename, entry_count, created_at),
        )
    return {
        "id": batch_id,
        "source": source,
        "filename": filename,
        "entry_count": entry_count,
        "created_at": created_at,
    }


def list_import_batches() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM import_batches ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def delete_import_batches_by_source(source: str) -> int:
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM import_batches WHERE source = ?", (source,))
    return cursor.rowcount


def create_tool_run(conversation_id: str, tool_name: str, arguments: dict, result: str) -> dict:
    run_id = str(uuid.uuid4())
    created_at = _now()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO tool_runs (id, conversation_id, tool_name, arguments, result, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, conversation_id, tool_name, json.dumps(arguments), result, created_at),
        )
    return {
        "id": run_id,
        "conversation_id": conversation_id,
        "tool_name": tool_name,
        "arguments": arguments,
        "result": result,
        "created_at": created_at,
    }


def list_tool_runs(conversation_id: str, after: str | None = None) -> list[dict]:
    """Tool runs for a conversation, oldest first. `after` (an ISO timestamp)
    narrows to a single turn -- the Talk tab passes the moment it sent the
    message so it only shows chips for the reply just received."""
    query = "SELECT * FROM tool_runs WHERE conversation_id = ?"
    params: list[str] = [conversation_id]
    if after:
        query += " AND created_at >= ?"
        params.append(after)
    with get_connection() as conn:
        rows = conn.execute(query + " ORDER BY created_at", params).fetchall()
    return [{**dict(row), "arguments": json.loads(row["arguments"])} for row in rows]


def create_pending_calendar_event(
    conversation_id: str, title: str, start_at: str, duration_minutes: int, description: str = ""
) -> dict:
    event_id = str(uuid.uuid4())
    created_at = _now()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO pending_calendar_events "
            "(id, conversation_id, title, start_at, duration_minutes, description, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (event_id, conversation_id, title, start_at, duration_minutes, description, created_at),
        )
    return {
        "id": event_id,
        "conversation_id": conversation_id,
        "title": title,
        "start_at": start_at,
        "duration_minutes": duration_minutes,
        "description": description,
        "created_at": created_at,
    }


def get_pending_calendar_event(event_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM pending_calendar_events WHERE id = ?", (event_id,)
        ).fetchone()
    return dict(row) if row else None


def list_pending_calendar_events(conversation_id: str, after: str | None = None) -> list[dict]:
    """Same `after`-filters-to-one-turn shape as list_tool_runs -- the Talk
    tab uses both together to render a turn's chips and its pending-event
    card from one X-Turn-Started value."""
    query = "SELECT * FROM pending_calendar_events WHERE conversation_id = ?"
    params: list[str] = [conversation_id]
    if after:
        query += " AND created_at >= ?"
        params.append(after)
    with get_connection() as conn:
        rows = conn.execute(query + " ORDER BY created_at", params).fetchall()
    return [dict(row) for row in rows]


def delete_pending_calendar_event(event_id: str) -> None:
    """Used for both outcomes -- confirmed (the real event now exists in
    Google Calendar) and discarded -- so a row's presence always means
    "still awaiting a decision"."""
    with get_connection() as conn:
        conn.execute("DELETE FROM pending_calendar_events WHERE id = ?", (event_id,))


def get_setting(key: str, default: str) -> str:
    with get_connection() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def create_persona_eval(
    prompt: str, persona_side: str, persona_answer: str, generic_answer: str
) -> dict:
    eval_id = str(uuid.uuid4())
    created_at = _now()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO persona_eval "
            "(id, prompt, persona_side, persona_answer, generic_answer, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (eval_id, prompt, persona_side, persona_answer, generic_answer, created_at),
        )
    return {"id": eval_id, "persona_side": persona_side}


def get_persona_eval(eval_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM persona_eval WHERE id = ?", (eval_id,)
        ).fetchone()
    return dict(row) if row else None


def record_persona_eval_choice(eval_id: str, choice: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE persona_eval SET choice = ?, decided_at = ? WHERE id = ?",
            (choice, _now(), eval_id),
        )


def _eval_outcome(persona_side: str, choice: str) -> str:
    if choice == "tie":
        return "tie"
    return "persona" if choice == persona_side else "generic"


def persona_eval_summary() -> dict:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT persona_side, choice FROM persona_eval WHERE choice IS NOT NULL"
        ).fetchall()
        recent_rows = conn.execute(
            "SELECT persona_side, choice FROM persona_eval "
            "WHERE choice IS NOT NULL ORDER BY decided_at DESC LIMIT 8"
        ).fetchall()

    persona_count = 0
    generic_count = 0
    tie_count = 0
    for row in rows:
        outcome = _eval_outcome(row["persona_side"], row["choice"])
        if outcome == "tie":
            tie_count += 1
        elif outcome == "persona":
            persona_count += 1
        else:
            generic_count += 1

    total = persona_count + generic_count + tie_count
    decided = persona_count + generic_count  # excludes ties
    # recent_rows is newest-first (for LIMIT to keep the most recent 8);
    # reverse so the scoreboard's bar chart reads left-to-right chronologically.
    recent = [_eval_outcome(row["persona_side"], row["choice"]) for row in reversed(recent_rows)]
    return {
        "persona": persona_count,
        "generic": generic_count,
        "tie": tie_count,
        "total": total,
        # persona_rate (unchanged) folds ties into the denominator, which
        # silently dilutes it -- a run with many ties looks worse even though
        # the twin never actually lost outright. decisive_rate and tie_rate
        # are reported alongside it so both "how often does it win outright"
        # and "how often is it indistinguishable from generic" are visible,
        # rather than conflated into one number.
        "persona_rate": persona_count / total if total else None,
        "decisive_rate": persona_count / decided if decided else None,
        "tie_rate": tie_count / total if total else None,
        "recent": recent,
    }


def create_persona_duel(
    prompt: str,
    language: str,
    arm_a: str,
    arm_b: str,
    model_a: str,
    model_b: str,
    answer_a: str,
    answer_b: str,
) -> dict:
    duel_id = str(uuid.uuid4())
    created_at = _now()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO persona_duels "
            "(id, prompt, language, arm_a, arm_b, model_a, model_b, answer_a, answer_b, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (duel_id, prompt, language, arm_a, arm_b, model_a, model_b, answer_a, answer_b, created_at),
        )
    return {"id": duel_id, "arm_a": arm_a, "arm_b": arm_b}


def get_persona_duel(duel_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM persona_duels WHERE id = ?", (duel_id,)).fetchone()
    return dict(row) if row else None


def record_persona_duel_choice(duel_id: str, choice: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE persona_duels SET choice = ?, decided_at = ? WHERE id = ?",
            (choice, _now(), duel_id),
        )


def persona_duel_arm_appearances() -> dict[str, int]:
    """How many recorded duels each arm has appeared in (decided or not) --
    drives round-robin challenger selection so no single challenger arm
    starves against the default."""
    with get_connection() as conn:
        rows = conn.execute("SELECT arm_a, arm_b FROM persona_duels").fetchall()
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["arm_a"]] = counts.get(row["arm_a"], 0) + 1
        counts[row["arm_b"]] = counts.get(row["arm_b"], 0) + 1
    return counts


def _duel_outcome(arm_a: str, arm_b: str, choice: str, default_arm: str) -> str:
    if choice == "tie":
        return "tie"
    winner_arm = arm_a if choice == "a" else arm_b
    return "default" if winner_arm == default_arm else "challenger"


def persona_duel_summary(default_arm: str, arm_labels: dict[str, str]) -> dict:
    """`default_arm`/`arm_labels` are passed in (rather than imported from
    persona.py) so this module stays agnostic of what the arms actually
    are -- it just aggregates whatever arm strings show up in the table."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT arm_a, arm_b, choice FROM persona_duels WHERE choice IS NOT NULL"
        ).fetchall()
        recent_rows = conn.execute(
            "SELECT arm_a, arm_b, choice FROM persona_duels "
            "WHERE choice IS NOT NULL ORDER BY decided_at DESC LIMIT 8"
        ).fetchall()

    stats: dict[str, dict[str, int]] = {}

    def _touch(arm: str) -> dict[str, int]:
        return stats.setdefault(arm, {"wins": 0, "losses": 0, "ties": 0, "appearances": 0})

    for row in rows:
        arm_a, arm_b, choice = row["arm_a"], row["arm_b"], row["choice"]
        stats_a, stats_b = _touch(arm_a), _touch(arm_b)
        stats_a["appearances"] += 1
        stats_b["appearances"] += 1
        if choice == "tie":
            stats_a["ties"] += 1
            stats_b["ties"] += 1
        elif choice == "a":
            stats_a["wins"] += 1
            stats_b["losses"] += 1
        else:
            stats_b["wins"] += 1
            stats_a["losses"] += 1

    total = len(rows)
    tie_count = sum(1 for row in rows if row["choice"] == "tie")

    default_stats = stats.get(default_arm, {"wins": 0, "losses": 0, "ties": 0, "appearances": 0})
    default_decided = default_stats["wins"] + default_stats["losses"]

    # recent_rows is newest-first (for LIMIT to keep the most recent 8);
    # reverse so the scoreboard's bar chart reads left-to-right chronologically.
    recent = [
        _duel_outcome(row["arm_a"], row["arm_b"], row["choice"], default_arm)
        for row in reversed(recent_rows)
    ]

    arms = [
        {
            "arm": arm,
            "label": arm_labels.get(arm, arm),
            "wins": s["wins"],
            "losses": s["losses"],
            "ties": s["ties"],
            "appearances": s["appearances"],
            "win_rate": s["wins"] / (s["wins"] + s["losses"]) if (s["wins"] + s["losses"]) else None,
        }
        for arm, s in sorted(stats.items())
    ]

    return {
        "total": total,
        "decided": total - tie_count,
        "tie": tie_count,
        "default_arm": default_arm,
        "default_arm_label": arm_labels.get(default_arm, default_arm),
        "default_arm_rate": default_stats["wins"] / default_decided if default_decided else None,
        "arms": arms,
        "recent": recent,
    }


def create_eval_run(
    batch_id: str,
    prompt_id: str,
    category: str,
    prompt: str,
    reply: str,
    gates: list[dict],
    model: str,
    judge_score: int | None = None,
    judge_notes: str | None = None,
    judge_model: str | None = None,
) -> dict:
    run_id = str(uuid.uuid4())
    created_at = _now()
    gates_failed_count = sum(1 for gate in gates if not gate["passed"])
    gates_detail = json.dumps(gates)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO eval_runs "
            "(id, batch_id, prompt_id, category, prompt, reply, gates_failed_count, "
            "gates_detail, judge_score, judge_notes, model, judge_model, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                batch_id,
                prompt_id,
                category,
                prompt,
                reply,
                gates_failed_count,
                gates_detail,
                judge_score,
                judge_notes,
                model,
                judge_model,
                created_at,
            ),
        )
    return {
        "id": run_id,
        "batch_id": batch_id,
        "prompt_id": prompt_id,
        "category": category,
        "prompt": prompt,
        "reply": reply,
        "gates_failed_count": gates_failed_count,
        "gates_detail": gates,
        "judge_score": judge_score,
        "judge_notes": judge_notes,
        "model": model,
        "judge_model": judge_model,
        "created_at": created_at,
    }


def list_eval_runs(batch_id: str | None = None) -> list[dict]:
    with get_connection() as conn:
        if batch_id:
            rows = conn.execute(
                "SELECT * FROM eval_runs WHERE batch_id = ? ORDER BY created_at ASC",
                (batch_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM eval_runs ORDER BY created_at DESC").fetchall()
    runs = []
    for row in rows:
        run = dict(row)
        run["gates_detail"] = json.loads(run["gates_detail"])
        runs.append(run)
    return runs
