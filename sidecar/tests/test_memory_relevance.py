import unittest
from unittest.mock import AsyncMock, patch

from app import config, memory, persona


class _FakeCollection:
    """Stands in for a Chroma collection. Records the kwargs query() was
    called with, so the tests can assert distances are actually requested --
    the gate is silently inert if `include` omits them."""

    def __init__(self, rows: list[tuple[str, float]]):
        self._rows = rows
        self.query_kwargs: dict | None = None

    def count(self) -> int:
        return len(self._rows)

    def query(self, **kwargs) -> dict:
        self.query_kwargs = kwargs
        n = min(kwargs.get("n_results", len(self._rows)), len(self._rows))
        selected = self._rows[:n]
        return {
            "documents": [[content for content, _ in selected]],
            "metadatas": [[{"topic": ""} for _ in selected]],
            "distances": [[distance for _, distance in selected]],
        }


class SearchMemoriesRelevanceTests(unittest.IsolatedAsyncioTestCase):
    """The relevance floor is what lets the twin say "I don't know": without
    it, search_memories() returned top-K regardless of distance and
    format_memories_section() then presented whatever came back as things
    that actually happened, so any question at all arrived with four pieces
    of a real life attached."""

    def _patch_collection(self, collection: _FakeCollection) -> None:
        patcher = patch.object(memory, "get_collection", return_value=collection)
        patcher.start()
        self.addCleanup(patcher.stop)
        embed_patcher = patch.object(memory.ollama, "embed", new=AsyncMock(return_value=[0.1, 0.2]))
        embed_patcher.start()
        self.addCleanup(embed_patcher.stop)

    async def test_keeps_results_within_the_threshold(self):
        collection = _FakeCollection([("a close memory", config.MEMORY_MAX_DISTANCE - 0.1)])
        self._patch_collection(collection)

        results = await memory.search_memories("anything")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "a close memory")
        self.assertIn("distance", results[0])

    async def test_drops_results_past_the_threshold(self):
        collection = _FakeCollection([("an unrelated memory", config.MEMORY_MAX_DISTANCE + 0.1)])
        self._patch_collection(collection)

        self.assertEqual(await memory.search_memories("anything"), [])

    async def test_boundary_distance_is_kept(self):
        """<= not <, matching search_vault -- worth pinning so a later edit
        can't quietly flip the comparison."""
        collection = _FakeCollection([("exactly at the limit", config.MEMORY_MAX_DISTANCE)])
        self._patch_collection(collection)

        self.assertEqual(len(await memory.search_memories("anything")), 1)

    async def test_mixed_results_keep_only_the_relevant_ones(self):
        collection = _FakeCollection(
            [
                ("relevant", config.MEMORY_MAX_DISTANCE - 0.2),
                ("irrelevant", config.MEMORY_MAX_DISTANCE + 0.4),
            ]
        )
        self._patch_collection(collection)

        results = await memory.search_memories("anything")

        self.assertEqual([row["content"] for row in results], ["relevant"])

    async def test_requests_distances_from_chroma(self):
        collection = _FakeCollection([("a memory", 0.1)])
        self._patch_collection(collection)

        await memory.search_memories("anything")

        self.assertIsNotNone(collection.query_kwargs)
        self.assertIn("distances", collection.query_kwargs["include"])

    async def test_empty_collection_returns_nothing(self):
        self._patch_collection(_FakeCollection([]))

        self.assertEqual(await memory.search_memories("anything"), [])


class FormatMemoriesSectionTests(unittest.TestCase):
    def test_empty_list_states_it_has_nothing_rather_than_vanishing(self):
        """An omitted section reads to the model as no constraint at all --
        which is what let it improvise a life story when retrieval came back
        empty. The empty case is now an explicit instruction."""
        section = memory.format_memories_section([])

        self.assertEqual(section, memory.NO_RELEVANT_MEMORIES_SECTION)
        self.assertNotEqual(section, "")
        self.assertIn("no stored memories", section.lower())
        self.assertIn("don't know", section.lower())

    def test_empty_case_still_allows_ordinary_conversation(self):
        """The failure mode to avoid on the other side: a twin that answers
        "I don't know" to "how are you?" because it has no memories."""
        section = memory.format_memories_section([])

        self.assertIn("general knowledge", section.lower())
        self.assertIn("normally", section.lower())

    def test_populated_list_keeps_the_assertive_framing(self):
        section = memory.format_memories_section([{"content": "I moved house in June"}])

        self.assertIn("I moved house in June", section)
        self.assertIn("actually happened to you", section)
        self.assertIn("first person", section)

    def test_populated_list_states_the_boundary(self):
        section = memory.format_memories_section([{"content": "I moved house in June"}])

        self.assertIn("only personal history", section.lower())
        self.assertIn("don't recall", section.lower())

    def test_retired_unfiltered_era_wording_is_gone(self):
        """That hedge only made sense while retrieval was unfiltered top-K.
        Asserting it stays gone so the old framing can't quietly return with
        the gate still in place."""
        section = memory.format_memories_section([{"content": "anything"}])

        self.assertNotIn("Only leave out ones that are genuinely unrelated", section)


class MemoryLookupFailureSectionTests(unittest.IsolatedAsyncioTestCase):
    """"Nothing relevant is stored" and "the lookup broke" are different
    claims about the twin's own past -- saying "I have no memory of that"
    when retrieval is simply down would be a confident false statement."""

    async def test_failure_flag_uses_the_unavailable_wording(self):
        prompt = await persona.build_system_prompt(
            [], [], config.DEFAULT_LANGUAGE, [], owner_name="Jordan", memory_lookup_failed=True
        )

        self.assertIn(memory.MEMORY_LOOKUP_UNAVAILABLE_SECTION, prompt)
        self.assertNotIn(memory.NO_RELEVANT_MEMORIES_SECTION, prompt)

    async def test_no_memories_without_failure_uses_the_empty_wording(self):
        prompt = await persona.build_system_prompt(
            [], [], config.DEFAULT_LANGUAGE, [], owner_name="Jordan"
        )

        self.assertIn(memory.NO_RELEVANT_MEMORIES_SECTION, prompt)
        self.assertNotIn(memory.MEMORY_LOOKUP_UNAVAILABLE_SECTION, prompt)

    async def test_unavailable_wording_declines_in_character(self):
        """It should decline as the person, not announce a malfunction --
        "my retrieval backend is down" is its own kind of persona break."""
        self.assertIn("as yourself", memory.MEMORY_LOOKUP_UNAVAILABLE_SECTION.lower())
        self.assertIn("technical", memory.MEMORY_LOOKUP_UNAVAILABLE_SECTION.lower())


if __name__ == "__main__":
    unittest.main()
