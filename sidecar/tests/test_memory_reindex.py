import unittest
from unittest.mock import AsyncMock, patch

from app import memory


class _FakeClient:
    """Minimal stand-in for the Chroma client: just enough to model which
    collections exist and let the tests assert the legacy one is dropped
    only at the right moment."""

    def __init__(self, names: list[str]):
        self.names = list(names)
        self.deleted: list[str] = []

    def list_collections(self):
        return [type("C", (), {"name": name})() for name in self.names]

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)
        if name in self.names:
            self.names.remove(name)


class _FakeCollection:
    def __init__(self, count: int = 0):
        self._count = count

    def count(self) -> int:
        return self._count


class NeedsCosineReindexTests(unittest.TestCase):
    def _patch(self, client: _FakeClient, cosine_count: int) -> None:
        client_patcher = patch.object(memory, "_get_client", return_value=client)
        client_patcher.start()
        self.addCleanup(client_patcher.stop)
        collection_patcher = patch.object(
            memory, "get_collection", return_value=_FakeCollection(cosine_count)
        )
        collection_patcher.start()
        self.addCleanup(collection_patcher.stop)

    def test_legacy_present_and_new_empty_needs_migrating(self):
        self._patch(_FakeClient([memory.LEGACY_MEMORIES_COLLECTION]), cosine_count=0)

        self.assertTrue(memory.needs_cosine_reindex())

    def test_no_legacy_collection_is_a_fresh_install(self):
        self._patch(_FakeClient([memory.MEMORIES_COLLECTION]), cosine_count=0)

        self.assertFalse(memory.needs_cosine_reindex())

    def test_already_rebuilt_does_not_run_again(self):
        """A populated cosine collection means a previous run got there --
        re-running would re-embed the whole corpus for nothing."""
        self._patch(
            _FakeClient([memory.LEGACY_MEMORIES_COLLECTION, memory.MEMORIES_COLLECTION]),
            cosine_count=12,
        )

        self.assertFalse(memory.needs_cosine_reindex())


class ReindexMemoriesTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = _FakeClient([memory.LEGACY_MEMORIES_COLLECTION])
        client_patcher = patch.object(memory, "_get_client", return_value=self.client)
        client_patcher.start()
        self.addCleanup(client_patcher.stop)

    async def test_indexes_every_memory_and_then_drops_the_legacy_collection(self):
        indexed: list[tuple[str, str, dict]] = []

        async def fake_index(memory_id, content, metadata):
            indexed.append((memory_id, content, metadata))

        with patch.object(memory, "index_memory", new=fake_index):
            count = await memory.reindex_memories_to_cosine(
                [
                    {"id": "1", "content": "first", "topic": "life", "occurred_at": "2020"},
                    {"id": "2", "content": "second", "topic": None, "occurred_at": None},
                ]
            )

        self.assertEqual(count, 2)
        self.assertEqual([row[0] for row in indexed], ["1", "2"])
        # Metadata has to match what the live create path writes, or pulled
        # memories index differently from typed ones.
        self.assertEqual(indexed[0][2], {"topic": "life", "occurred_at": "2020"})
        self.assertEqual(indexed[1][2], {"topic": "", "occurred_at": ""})
        self.assertEqual(self.client.deleted, [memory.LEGACY_MEMORIES_COLLECTION])

    async def test_failure_partway_leaves_the_legacy_collection_intact(self):
        """The whole reason the rebuild writes to a new collection first: if
        Ollama dies mid-embed, the twin must still have a usable index on the
        next launch rather than nothing at all."""
        calls = {"n": 0}

        async def flaky_index(memory_id, content, metadata):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("ollama went away")

        with (
            patch.object(memory, "index_memory", new=flaky_index),
            self.assertRaises(RuntimeError),
        ):
            await memory.reindex_memories_to_cosine(
                [
                    {"id": "1", "content": "first", "topic": "", "occurred_at": ""},
                    {"id": "2", "content": "second", "topic": "", "occurred_at": ""},
                ]
            )

        self.assertEqual(self.client.deleted, [])
        self.assertIn(memory.LEGACY_MEMORIES_COLLECTION, self.client.names)

    async def test_no_memories_still_clears_a_stale_legacy_collection(self):
        """Otherwise needs_cosine_reindex() keeps returning True forever on an
        install that has no memories saved."""
        with patch.object(memory, "index_memory", new=AsyncMock()):
            count = await memory.reindex_memories_to_cosine([])

        self.assertEqual(count, 0)
        self.assertEqual(self.client.deleted, [memory.LEGACY_MEMORIES_COLLECTION])

    async def test_dropping_is_idempotent_when_legacy_is_already_gone(self):
        self.client.names = [memory.MEMORIES_COLLECTION]

        with patch.object(memory, "index_memory", new=AsyncMock()):
            await memory.reindex_memories_to_cosine([])

        self.assertEqual(self.client.deleted, [])


class CollectionSpaceTests(unittest.TestCase):
    def test_memories_and_vault_are_cosine_the_style_corpus_is_not(self):
        """The threshold in search_memories() is only meaningful on cosine's
        fixed [0, 2] scale -- on L2 it would be comparing against an
        unbounded, embedding-dependent number."""
        self.assertIn(memory.MEMORIES_COLLECTION, memory._COSINE_COLLECTIONS)
        self.assertIn(memory.VAULT_COLLECTION, memory._COSINE_COLLECTIONS)
        self.assertNotIn(memory.STYLE_COLLECTION, memory._COSINE_COLLECTIONS)

    def test_live_collection_name_is_not_the_legacy_one(self):
        self.assertNotEqual(memory.MEMORIES_COLLECTION, memory.LEGACY_MEMORIES_COLLECTION)


if __name__ == "__main__":
    unittest.main()
