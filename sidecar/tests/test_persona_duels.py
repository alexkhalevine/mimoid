import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException

from app import db, main, persona


class ArmEligibilityTests(unittest.TestCase):
    """eligible_arms() must reflect what was actually retrieved for THIS
    question, not just whether the corpus has any guide/examples at all --
    guide_fewshot degenerates into guide_only the moment no examples were
    retrieved for a particular prompt, and must not be offered as if it
    were a distinct arm in that case."""

    def test_guide_and_examples_makes_all_four_arms_eligible(self):
        arms = persona.eligible_arms(has_guide=True, has_examples=True)
        self.assertEqual(
            set(arms),
            {persona.ARM_PLAIN, persona.ARM_FEWSHOT_ONLY, persona.ARM_GUIDE_ONLY, persona.ARM_GUIDE_FEWSHOT},
        )

    def test_guide_without_examples_excludes_guide_fewshot(self):
        arms = persona.eligible_arms(has_guide=True, has_examples=False)
        self.assertEqual(set(arms), {persona.ARM_PLAIN, persona.ARM_GUIDE_ONLY})

    def test_examples_without_guide_excludes_guide_arms(self):
        arms = persona.eligible_arms(has_guide=False, has_examples=True)
        self.assertEqual(set(arms), {persona.ARM_PLAIN, persona.ARM_FEWSHOT_ONLY})

    def test_neither_guide_nor_examples_leaves_only_plain(self):
        arms = persona.eligible_arms(has_guide=False, has_examples=False)
        self.assertEqual(arms, [persona.ARM_PLAIN])


class StyleSectionArmTests(unittest.IsolatedAsyncioTestCase):
    """_style_section's use_guide/use_examples params must actually change
    what's assembled per arm, and critically must NOT let plain/guide_only
    fall through to the raw recency-dump fallback -- that would silently
    reintroduce a style signal the arm exists to test the absence of."""

    async def test_plain_emits_no_style_text_even_with_a_large_corpus(self):
        entries = [{"kind": "text", "prompt": None, "content": f"entry {i}"} for i in range(5)]
        with (
            patch.object(persona.db, "get_style_guide", return_value=None),
            patch.object(persona.db, "list_style_entries", return_value=entries),
        ):
            section = await persona._style_section(style_examples=[], use_guide=False, use_examples=False)
        self.assertEqual(section, "")

    async def test_guide_only_omits_the_framing_block_even_with_examples_available(self):
        guide = {"content": "Write plainly.", "updated_at": "2026-01-01T00:00:00+00:00"}
        style_examples = [{"kind": "text", "prompt": None, "content": "an example"}]
        with patch.object(persona.db, "get_style_guide", return_value=guide):
            section = await persona._style_section(style_examples, use_guide=True, use_examples=False)
        self.assertIn("Write plainly.", section)
        self.assertNotIn(persona._STYLE_EXAMPLES_FRAMING, section)
        self.assertNotIn("an example", section)

    async def test_guide_only_does_not_fall_back_to_recency_dump_without_a_guide(self):
        entries = [{"kind": "text", "prompt": None, "content": f"entry {i}"} for i in range(5)]
        with (
            patch.object(persona.db, "get_style_guide", return_value=None),
            patch.object(persona.db, "list_style_entries", return_value=entries),
        ):
            section = await persona._style_section(style_examples=[], use_guide=True, use_examples=False)
        self.assertEqual(section, "")

    async def test_fewshot_only_omits_the_guide_line(self):
        guide = {"content": "Write plainly.", "updated_at": "2026-01-01T00:00:00+00:00"}
        style_examples = [{"kind": "text", "prompt": None, "content": "an example"}]
        with patch.object(persona.db, "get_style_guide", return_value=guide):
            section = await persona._style_section(style_examples, use_guide=False, use_examples=True)
        self.assertNotIn("Write plainly.", section)
        self.assertIn("an example", section)

    async def test_guide_fewshot_includes_both(self):
        guide = {"content": "Write plainly.", "updated_at": "2026-01-01T00:00:00+00:00"}
        style_examples = [{"kind": "text", "prompt": None, "content": "an example"}]
        with patch.object(persona.db, "get_style_guide", return_value=guide):
            section = await persona._style_section(style_examples, use_guide=True, use_examples=True)
        self.assertIn("Write plainly.", section)
        self.assertIn("an example", section)


class MatchupSelectionTests(unittest.TestCase):
    def test_default_arm_faces_the_least_appeared_eligible_challenger(self):
        appearances = {
            persona.ARM_GUIDE_FEWSHOT: 10,
            persona.ARM_GUIDE_ONLY: 8,
            persona.ARM_FEWSHOT_ONLY: 1,
            persona.ARM_PLAIN: 3,
        }
        eligible = [persona.ARM_GUIDE_FEWSHOT, persona.ARM_GUIDE_ONLY, persona.ARM_FEWSHOT_ONLY, persona.ARM_PLAIN]
        with patch.object(persona.db, "persona_duel_arm_appearances", return_value=appearances):
            default_arm, challenger = persona._pick_matchup(eligible)
        self.assertEqual(default_arm, persona.ARM_GUIDE_FEWSHOT)
        self.assertEqual(challenger, persona.ARM_FEWSHOT_ONLY)

    def test_unseen_arms_default_to_zero_appearances(self):
        eligible = [persona.ARM_GUIDE_FEWSHOT, persona.ARM_PLAIN]
        with patch.object(persona.db, "persona_duel_arm_appearances", return_value={}):
            default_arm, challenger = persona._pick_matchup(eligible)
        self.assertEqual(default_arm, persona.ARM_GUIDE_FEWSHOT)
        self.assertEqual(challenger, persona.ARM_PLAIN)

    def test_falls_back_to_two_least_appeared_when_default_not_eligible(self):
        appearances = {persona.ARM_FEWSHOT_ONLY: 5, persona.ARM_PLAIN: 1}
        eligible = [persona.ARM_FEWSHOT_ONLY, persona.ARM_PLAIN]
        with patch.object(persona.db, "persona_duel_arm_appearances", return_value=appearances):
            arm_x, arm_y = persona._pick_matchup(eligible)
        self.assertEqual({arm_x, arm_y}, {persona.ARM_FEWSHOT_ONLY, persona.ARM_PLAIN})
        self.assertEqual(arm_x, persona.ARM_PLAIN)  # fewer appearances ranked first


class CompareArmsTests(unittest.IsolatedAsyncioTestCase):
    async def test_embeds_the_prompt_exactly_once(self):
        with (
            patch.object(persona.ollama, "embed", AsyncMock(return_value=[0.1, 0.2])) as embed_mock,
            patch.object(persona.memory, "search_memories", AsyncMock(return_value=[])),
            patch.object(persona.memory, "search_style_examples", AsyncMock(return_value=[{"kind": "text", "prompt": None, "content": "x"}])),
            patch.object(persona.db, "get_style_guide", return_value=None),
            patch.object(persona.db, "persona_duel_arm_appearances", return_value={}),
            patch.object(persona.ollama, "chat", AsyncMock(return_value="an answer")),
        ):
            await persona.compare_arms("What's your favorite food?")
        embed_mock.assert_awaited_once_with("What's your favorite food?")

    async def test_raises_duel_unavailable_when_fewer_than_two_arms_eligible(self):
        with (
            patch.object(persona.ollama, "embed", AsyncMock(return_value=[0.1])),
            patch.object(persona.memory, "search_memories", AsyncMock(return_value=[])),
            patch.object(persona.memory, "search_style_examples", AsyncMock(return_value=[])),
            patch.object(persona.db, "get_style_guide", return_value=None),
            self.assertRaises(persona.DuelUnavailable),
        ):
            await persona.compare_arms("Anything")

    async def test_generations_run_concurrently_not_sequentially(self):
        # Wall-clock timing thresholds are flaky on shared/loaded CI runners
        # (this used to assert elapsed < 0.09s against two 0.05s sleeps,
        # which is a fine margin locally but not against CI jitter) -- a
        # peak-concurrency counter proves the same thing deterministically,
        # matching the pattern already used for tools.dispatch_all and
        # google_calendar's multi-calendar fan-out.
        active = 0
        peak = 0

        async def slow_chat(messages, model=None):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
            active -= 1
            return "an answer"

        with (
            patch.object(persona.ollama, "embed", AsyncMock(return_value=[0.1])),
            patch.object(persona.memory, "search_memories", AsyncMock(return_value=[])),
            patch.object(persona.memory, "search_style_examples", AsyncMock(return_value=[{"kind": "text", "prompt": None, "content": "x"}])),
            patch.object(persona.db, "get_style_guide", return_value=None),
            patch.object(persona.db, "persona_duel_arm_appearances", return_value={}),
            patch.object(persona.ollama, "chat", slow_chat),
        ):
            await persona.compare_arms("Anything")
        self.assertEqual(peak, 2, "the two arms' generations ran sequentially, not concurrently")


class PersonaDuelRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_duel_returns_400_when_arms_unavailable(self):
        with (
            patch.object(main.persona, "compare_arms", AsyncMock(side_effect=main.persona.DuelUnavailable)),
            self.assertRaises(HTTPException) as caught,
        ):
            await main.create_persona_duel(main.CompareRequest(prompt="hi"))
        self.assertEqual(caught.exception.status_code, 400)

    async def test_create_duel_returns_503_on_ollama_failure(self):
        with (
            patch.object(main.persona, "compare_arms", AsyncMock(side_effect=httpx.ConnectError("offline"))),
            self.assertRaises(HTTPException) as caught,
        ):
            await main.create_persona_duel(main.CompareRequest(prompt="hi"))
        self.assertEqual(caught.exception.status_code, 503)

    async def test_create_duel_does_not_reveal_arms(self):
        result = {"arm_a": "plain", "arm_b": "guide_only", "answer_a": "A", "answer_b": "B"}
        with (
            patch.object(main.persona, "compare_arms", AsyncMock(return_value=result)),
            patch.object(main.db, "create_persona_duel", return_value={"id": "d1"}),
        ):
            response = await main.create_persona_duel(main.CompareRequest(prompt="hi"))
        self.assertEqual(set(response.keys()), {"id", "a", "b"})
        self.assertEqual(response["a"], "A")
        self.assertEqual(response["b"], "B")

    def test_choice_route_rejects_a_second_choice_for_the_same_duel(self):
        decided = {"id": "d1", "arm_a": "plain", "arm_b": "guide_only", "choice": "a"}
        with (
            patch.object(main.db, "get_persona_duel", return_value=decided),
            self.assertRaises(HTTPException) as caught,
        ):
            main.choose_persona_duel("d1", main.DuelChoiceRequest(choice="b"))
        self.assertEqual(caught.exception.status_code, 409)

    def test_choice_route_404s_for_an_unknown_duel(self):
        with (
            patch.object(main.db, "get_persona_duel", return_value=None),
            self.assertRaises(HTTPException) as caught,
        ):
            main.choose_persona_duel("missing", main.DuelChoiceRequest(choice="a"))
        self.assertEqual(caught.exception.status_code, 404)

    def test_choice_route_reveals_arms_after_deciding(self):
        undecided = {"id": "d1", "arm_a": "plain", "arm_b": "guide_only", "choice": None}
        summary = {"total": 1, "decided": 1, "tie": 0}
        with (
            patch.object(main.db, "get_persona_duel", return_value=undecided),
            patch.object(main.db, "record_persona_duel_choice") as record_mock,
            patch.object(main.db, "persona_duel_summary", return_value=summary),
        ):
            response = main.choose_persona_duel("d1", main.DuelChoiceRequest(choice="a"))
        record_mock.assert_called_once_with("d1", "a")
        self.assertEqual(response["arm_a"], "plain")
        self.assertEqual(response["arm_b"], "guide_only")
        self.assertEqual(response["total"], 1)


class StyleEntrySourceAllowlistTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_an_arbitrary_source(self):
        body = main.StyleEntryRequest(kind="text", content="hi", source="not-a-real-source")
        with self.assertRaises(HTTPException) as caught:
            await main.create_style_entry(body)
        self.assertEqual(caught.exception.status_code, 400)

    async def test_rejects_the_retired_persona_check_source(self):
        # "persona-check" used to be a legitimate promotion source (saving a
        # duel's winning ANSWER, i.e. model output) -- that was the
        # self-training bug this allowlist now closes. Any client (including
        # a stale frontend build) attempting to write it must be rejected,
        # not silently accepted.
        body = main.StyleEntryRequest(kind="qa", content="hi", prompt="q", source="persona-check")
        with self.assertRaises(HTTPException) as caught:
            await main.create_style_entry(body)
        self.assertEqual(caught.exception.status_code, 400)

    async def test_accepts_own_answer_as_a_source(self):
        record = {"id": "e1", "kind": "qa", "content": "hi", "prompt": "q", "source": "own-answer"}
        with (
            patch.object(main.db, "create_style_entry", return_value=record) as create_mock,
            patch.object(main, "memory") as memory_mock,
        ):
            memory_mock.index_style_entry = AsyncMock(return_value=None)
            body = main.StyleEntryRequest(kind="qa", content="hi", prompt="q", source="own-answer")
            result = await main.create_style_entry(body)
        create_mock.assert_called_once_with("qa", "hi", "q", source="own-answer")
        self.assertEqual(result, record)


class PersonaDuelDbTests(unittest.TestCase):
    """Exercises persona_duels' SQL against a real (temp) SQLite file --
    thin enough SQL that mocking db would just re-assert the implementation."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmp_dir.name) / "test.db"
        self._patcher = patch.object(db.config, "DB_PATH", self._db_path)
        self._patcher.start()
        db.init_db()

    def tearDown(self):
        self._patcher.stop()
        self._tmp_dir.cleanup()

    def _make_duel(self, arm_a, arm_b, choice=None):
        record = db.create_persona_duel(
            prompt="hi",
            language="en",
            arm_a=arm_a,
            arm_b=arm_b,
            model_a="llama3",
            model_b="llama3",
            answer_a="A",
            answer_b="B",
        )
        if choice:
            db.record_persona_duel_choice(record["id"], choice)
        return record

    def test_arm_appearances_counts_both_slots(self):
        self._make_duel("guide_fewshot", "plain")
        self._make_duel("guide_fewshot", "fewshot_only")
        appearances = db.persona_duel_arm_appearances()
        self.assertEqual(appearances["guide_fewshot"], 2)
        self.assertEqual(appearances["plain"], 1)
        self.assertEqual(appearances["fewshot_only"], 1)

    def test_summary_is_all_none_rates_at_zero_rounds(self):
        summary = db.persona_duel_summary("guide_fewshot", {"guide_fewshot": "Style guide + matched examples"})
        self.assertEqual(summary["total"], 0)
        self.assertIsNone(summary["default_arm_rate"])
        self.assertEqual(summary["arms"], [])
        self.assertEqual(summary["recent"], [])

    def test_default_arm_rate_excludes_ties_from_the_denominator(self):
        self._make_duel("guide_fewshot", "plain", choice="a")  # default wins
        self._make_duel("guide_fewshot", "plain", choice="b")  # default loses
        self._make_duel("guide_fewshot", "plain", choice="tie")
        self._make_duel("guide_fewshot", "plain", choice="tie")

        summary = db.persona_duel_summary("guide_fewshot", {})
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["tie"], 2)
        self.assertEqual(summary["decided"], 2)
        # 1 win / (1 win + 1 loss) == 0.5, ties excluded from the denominator.
        self.assertEqual(summary["default_arm_rate"], 0.5)

    def test_undecided_duels_are_excluded_from_the_summary(self):
        self._make_duel("guide_fewshot", "plain")  # no choice recorded
        summary = db.persona_duel_summary("guide_fewshot", {})
        self.assertEqual(summary["total"], 0)

    def test_recent_reads_oldest_to_newest_and_labels_default_vs_challenger(self):
        self._make_duel("guide_fewshot", "plain", choice="a")  # default won
        self._make_duel("plain", "guide_fewshot", choice="a")  # challenger (plain) won
        self._make_duel("guide_fewshot", "plain", choice="tie")

        summary = db.persona_duel_summary("guide_fewshot", {})
        self.assertEqual(summary["recent"], ["default", "challenger", "tie"])

    def test_per_arm_stats_track_wins_losses_ties_and_appearances(self):
        self._make_duel("guide_fewshot", "plain", choice="a")
        self._make_duel("guide_fewshot", "plain", choice="b")

        summary = db.persona_duel_summary("guide_fewshot", {"plain": "No style layer"})
        by_arm = {row["arm"]: row for row in summary["arms"]}
        self.assertEqual(by_arm["guide_fewshot"]["wins"], 1)
        self.assertEqual(by_arm["guide_fewshot"]["losses"], 1)
        self.assertEqual(by_arm["guide_fewshot"]["appearances"], 2)
        self.assertEqual(by_arm["plain"]["label"], "No style layer")

    def test_choice_idempotency_guard_reads_from_get_persona_duel(self):
        record = self._make_duel("guide_fewshot", "plain", choice="a")
        fetched = db.get_persona_duel(record["id"])
        self.assertEqual(fetched["choice"], "a")
        self.assertIsNotNone(fetched["decided_at"])


if __name__ == "__main__":
    unittest.main()
