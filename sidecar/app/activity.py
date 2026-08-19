"""Tracks what backend task the sidecar is currently working on, so the
desktop app can show a live "current task" label. Handlers can overlap
(FastAPI serves requests concurrently), so this keeps an ordered set of
active labels and reports the most recently started one; `None` when idle."""
import itertools
import threading
from collections.abc import Iterator
from contextlib import contextmanager

_lock = threading.Lock()
_active: dict[int, str] = {}
_counter = itertools.count()


def current() -> str | None:
    """The most recently started active task label, or None if idle."""
    with _lock:
        if not _active:
            return None
        # dicts preserve insertion order, so the last key is the newest task.
        last_token = next(reversed(_active))
        return _active[last_token]


@contextmanager
def track(label: str) -> Iterator[None]:
    """Marks `label` as the current task for the duration of the block.
    Safe to nest and to overlap across requests; each use is independent."""
    with _lock:
        token = next(_counter)
        _active[token] = label
    try:
        yield
    finally:
        with _lock:
            _active.pop(token, None)


def reset() -> None:
    """Clears all tracked activity. For tests."""
    with _lock:
        _active.clear()
