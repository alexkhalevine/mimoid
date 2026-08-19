import unittest
from unittest.mock import patch

from app import persona


class StyleSectionFramingTests(unittest.IsolatedAsyncioTestCase):
    """persona._style_section() must only ever inject examples framed as
    voice-only reference -- never as facts, memories, or a transcript to
    analyze. Guards against a small local model copying an unrelated
    example verbatim as if it were the actual answer, or
    describing/analyzing injected examples instead of answering."""

    async def test_relevant_examples_use_voice_only_framing(self):
        style_examples = [{"kind": "text", "prompt": None, "content": "I love hiking on weekends."}]
        with patch.object(persona.db, "get_style_guide", return_value=None):
            section = await persona._style_section(style_examples)

        self.assertIn(persona._STYLE_EXAMPLES_FRAMING, section)
        self.assertIn("I love hiking on weekends.", section)
        # The old vague framing must be gone -- it's what let a small model
        # treat samples as facts/transcript instead of voice reference.
        self.assertNotIn("match this voice", section)
        self.assertNotIn("Examples of how you write and talk", section)

    async def test_relevant_examples_take_priority_over_unrelated_recency_dump(self):
        """The core regurgitation bug: with no guide, relevant retrieved
        examples must be used exclusively -- entries that only exist because
        they're recent (and have nothing to do with the question) must NOT
        also be injected alongside them."""
        style_examples = [{"kind": "text", "prompt": None, "content": "RELEVANT_EXAMPLE_TEXT"}]
        unrelated_entries = [
            {"kind": "text", "prompt": None, "content": f"unrelated recent entry {i}"} for i in range(50)
        ]
        with (
            patch.object(persona.db, "get_style_guide", return_value=None),
            patch.object(persona.db, "list_style_entries", return_value=unrelated_entries),
        ):
            section = await persona._style_section(style_examples)

        self.assertIn("RELEVANT_EXAMPLE_TEXT", section)
        self.assertNotIn("unrelated recent entry", section)

    async def test_no_relevant_examples_falls_back_to_recency_dump_still_framed(self):
        entries = [{"kind": "text", "prompt": None, "content": f"entry {i}"} for i in range(3)]
        with (
            patch.object(persona.db, "get_style_guide", return_value=None),
            patch.object(persona.db, "list_style_entries", return_value=entries),
        ):
            section = await persona._style_section(style_examples=[])

        self.assertIn(persona._STYLE_EXAMPLES_FRAMING, section)
        self.assertIn("entry 0", section)

    async def test_no_guide_no_examples_no_entries_emits_nothing(self):
        with (
            patch.object(persona.db, "get_style_guide", return_value=None),
            patch.object(persona.db, "list_style_entries", return_value=[]),
        ):
            section = await persona._style_section(style_examples=[])

        self.assertEqual(section, "")

    async def test_guide_with_examples_uses_voice_only_framing(self):
        guide = {"content": "You write short sentences.", "updated_at": "2026-01-01T00:00:00+00:00"}
        style_examples = [{"kind": "text", "prompt": None, "content": "Short and punchy."}]
        with patch.object(persona.db, "get_style_guide", return_value=guide):
            section = await persona._style_section(style_examples)

        self.assertIn("You write short sentences.", section)
        self.assertIn(persona._STYLE_EXAMPLES_FRAMING, section)
        self.assertIn("Short and punchy.", section)
        self.assertNotIn("match this voice", section)


class LocalDistillationCorpusCapTests(unittest.IsolatedAsyncioTestCase):
    """distill_style_guide_locally() must never silently hand Ollama more
    corpus than its context window can hold -- Ollama truncates silently,
    which would quietly degrade the guide with no signal anything was cut."""

    def test_small_corpus_is_not_truncated(self):
        entries = [{"kind": "text", "prompt": None, "content": "short"} for _ in range(5)]
        kept = persona._cap_corpus_for_local_distillation(entries)
        self.assertEqual(len(kept), len(entries))

    def test_oversized_corpus_is_capped_to_fit_the_budget(self):
        # Each entry ~1000 chars; OLLAMA_NUM_CTX defaults to 8192 tokens --
        # comfortably less than 300 * 1000 chars, so this must truncate.
        entries = [{"kind": "text", "prompt": None, "content": "x" * 1000} for _ in range(300)]
        kept = persona._cap_corpus_for_local_distillation(entries)
        self.assertLess(len(kept), len(entries))
        self.assertGreater(len(kept), 0)
        # Kept entries must be a prefix (the newest-first ordering
        # db.list_style_entries() already returns) -- not an arbitrary subset.
        self.assertEqual(kept, entries[: len(kept)])

    async def test_distill_locally_sends_capped_corpus_to_ollama(self):
        entries = [{"kind": "text", "prompt": None, "content": "x" * 1000} for _ in range(300)]
        captured = {}

        async def fake_chat(messages, model=None):
            captured["messages"] = messages
            return "a guide"

        with (
            patch.object(persona.db, "list_style_entries", return_value=entries),
            patch.object(persona.ollama, "chat", fake_chat),
        ):
            result = await persona.distill_style_guide_locally()

        self.assertEqual(result, "a guide")
        sent_corpus = captured["messages"][1]["content"]
        full_corpus_len = len(persona._format_style_examples(entries))
        self.assertLess(len(sent_corpus), full_corpus_len)


if __name__ == "__main__":
    unittest.main()
