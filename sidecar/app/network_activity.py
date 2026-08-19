"""Tracks the sidecar's outbound calls that leave the local machine --
OpenRouter, Nextcloud/WebDAV backup, and Ollama model downloads --
deliberately excluding local Ollama chat/embedding traffic to 127.0.0.1,
which never reaches the internet. Powers a footer indicator so external
network access in this offline-first app is never silent. Same shape as
activity.py's tracker, extended with a bounded history since the frontend
needs to show recent internet accesses, not just the current one."""
import itertools
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

HISTORY_LIMIT = 20

_lock = threading.Lock()
_active: dict[int, str] = {}
_counter = itertools.count()
_history: list[dict] = []


def current() -> dict | None:
    """The most recently started active call, as {id, label}, or None if idle."""
    with _lock:
        if not _active:
            return None
        last_token = next(reversed(_active))
        return {"id": last_token, "label": _active[last_token]}


def history() -> list[dict]:
    """Finished calls, newest first, as {id, label, at} (at = unix epoch seconds)."""
    with _lock:
        return list(reversed(_history))


@contextmanager
def track(label: str) -> Iterator[None]:
    """Marks `label` as an active internet call for the duration of the
    block, then records it in history on exit. Safe to nest/overlap."""
    with _lock:
        token = next(_counter)
        _active[token] = label
        started_at = time.time()
    try:
        yield
    finally:
        with _lock:
            _active.pop(token, None)
            _history.append({"id": token, "label": label, "at": started_at})
            del _history[:-HISTORY_LIMIT]


def reset() -> None:
    """Clears all tracked activity. For tests."""
    with _lock:
        _active.clear()
        _history.clear()
