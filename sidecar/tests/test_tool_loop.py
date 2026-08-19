import asyncio
import unittest
from unittest.mock import patch

from app import config, main


def _token(text):
    return {"type": "token", "text": text}


def _calls(*names):
    return {
        "type": "tool_calls",
        "calls": [{"function": {"name": name, "arguments": {}}} for name in names],
    }


class ToolLoopTests(unittest.TestCase):
    """The tool loop lives inside the streaming generator in send_message.
    These drive it with a scripted Ollama so the round-tripping, the
    persistence, and the prompt-growth invariant are all pinned down without
    a model anywhere near the test."""

    def setUp(self):
        self.requests: list[list[dict]] = []
        self.tool_runs: list[tuple] = []
        self.saved_messages: list[tuple] = []
        self.scripted: list[list[dict]] = []

        async def fake_chat_stream(messages, tools=None):
            # Snapshot: the loop mutates the same list between rounds, so a
            # reference would show every round the final state.
            self.requests.append([dict(message) for message in messages])
            for event in self.scripted.pop(0) if self.scripted else [_token("done")]:
                yield event

        patches = [
            patch.object(main.ollama, "chat_stream", fake_chat_stream),
            patch.object(main.db, "conversation_exists", return_value=True),
            patch.object(main.db, "get_messages", return_value=[{"role": "user", "content": "hi"}]),
            patch.object(main.db, "add_message", side_effect=lambda *a: self.saved_messages.append(a)),
            patch.object(main.db, "create_tool_run", side_effect=lambda *a: self.tool_runs.append(a)),
            # Retrieval is exercised in its own test below; keep it out of the way here.
            patch.object(main.ollama, "embed", side_effect=AssertionError("unused")),
            patch.object(main.persona, "build_system_prompt", return_value="SYSTEM"),
            patch.object(main.config, "DEFAULT_MODEL", "qwen2.5"),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

        # embed() failing is a supported path (retrieval is best-effort) and
        # short-circuits the searches, keeping these tests to the tool loop.
        embed = patch.object(main.ollama, "embed", side_effect=main.httpx.HTTPError("skip"))
        embed.start()
        self.addCleanup(embed.stop)

    def _send(self):
        async def run():
            response = await main.send_message("conv-1", main.SendMessageRequest(content="hi"))
            return "".join([chunk async for chunk in response.body_iterator])

        return asyncio.run(run())

    def test_tool_free_turn_streams_once(self):
        self.scripted = [[_token("Hello"), _token(" there")]]
        self.assertEqual(self._send(), "Hello there")
        self.assertEqual(len(self.requests), 1, "a tool-free reply must not cost a second pass")
        self.assertEqual(self.tool_runs, [])

    def test_tool_call_is_executed_and_fed_back(self):
        self.scripted = [[_calls("get_current_datetime")], [_token("It's Tuesday.")]]
        with patch.object(main.tools, "dispatch_all", return_value=["it is Tuesday"]):
            body = self._send()

        self.assertEqual(body, "It's Tuesday.")
        self.assertEqual(len(self.requests), 2)

        follow_up = self.requests[1]
        self.assertEqual(follow_up[-1], {"role": "tool", "content": "it is Tuesday"})
        self.assertEqual(follow_up[-2]["role"], "assistant")
        self.assertIn("tool_calls", follow_up[-2])

    def test_tool_run_is_recorded(self):
        self.scripted = [[_calls("get_current_datetime")], [_token("ok")]]
        with patch.object(main.tools, "dispatch_all", return_value=["it is Tuesday"]):
            self._send()
        self.assertEqual(
            self.tool_runs, [("conv-1", "get_current_datetime", {}, "it is Tuesday")]
        )

    def test_follow_up_prompt_is_a_strict_prefix_extension(self):
        """What makes the extra pass cheap: because nothing before the
        appended messages changes, Ollama's KV cache covers the whole
        original prompt and only the new tokens need prefilling. A refactor
        that rewrites or reorders earlier messages would silently turn every
        tool turn into a full re-prefill -- so it fails here instead."""
        self.scripted = [[_calls("get_current_datetime")], [_token("ok")]]
        with patch.object(main.tools, "dispatch_all", return_value=["result"]):
            self._send()

        first, second = self.requests
        self.assertEqual(second[: len(first)], first)
        self.assertGreater(len(second), len(first))

    def test_two_calls_in_one_round_both_get_tool_messages(self):
        self.scripted = [[_calls("a", "b")], [_token("ok")]]
        with patch.object(main.tools, "dispatch_all", return_value=["ra", "rb"]):
            self._send()

        tool_messages = [m for m in self.requests[1] if m["role"] == "tool"]
        self.assertEqual([m["content"] for m in tool_messages], ["ra", "rb"])
        self.assertEqual(len(self.tool_runs), 2)

    def test_loop_is_bounded(self):
        """A model that keeps calling tools forever must not stream forever."""
        self.scripted = [[_calls("a")] for _ in range(config.TOOL_MAX_ITERATIONS + 5)]
        with patch.object(main.tools, "dispatch_all", return_value=["r"]):
            self._send()
        self.assertEqual(len(self.requests), config.TOOL_MAX_ITERATIONS)

    def test_text_streamed_before_a_tool_call_is_kept(self):
        self.scripted = [[_token("Let me check. "), _calls("a")], [_token("It's Tuesday.")]]
        with patch.object(main.tools, "dispatch_all", return_value=["r"]):
            body = self._send()
        self.assertEqual(body, "Let me check. It's Tuesday.")
        # The saved transcript must match what the user actually saw, tool
        # round or not -- otherwise reloading the conversation loses the
        # first half of the reply.
        assistant = [m for m in self.saved_messages if m[1] == "assistant"]
        self.assertEqual(assistant, [("conv-1", "assistant", "Let me check. It's Tuesday.")])

    def test_tools_are_not_offered_to_a_model_that_cannot_use_them(self):
        seen = {}

        async def capture(messages, tools=None):
            seen["tools"] = tools
            yield _token("hi")

        with (
            patch.object(main.config, "DEFAULT_MODEL", "llama3"),
            patch.object(main.ollama, "chat_stream", capture),
        ):
            self._send()
        self.assertEqual(seen["tools"], [])


class ParallelRetrievalTests(unittest.TestCase):
    """The three searches share one embedding and are independent, so they
    run concurrently -- in sequence they added three round trips to the
    front of every turn. One failing collection must still leave the other
    two usable, which a bare gather would not do."""

    def setUp(self):
        patches = [
            patch.object(main.db, "conversation_exists", return_value=True),
            patch.object(main.db, "get_messages", return_value=[{"role": "user", "content": "hi"}]),
            patch.object(main.db, "add_message"),
            patch.object(main.ollama, "embed", return_value=[0.1]),
            patch.object(main.persona, "build_system_prompt", return_value="SYSTEM"),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

        async def stream(messages, tools=None):
            yield {"type": "token", "text": "ok"}

        stream_patch = patch.object(main.ollama, "chat_stream", stream)
        stream_patch.start()
        self.addCleanup(stream_patch.stop)

    def _send(self):
        async def run():
            response = await main.send_message("conv-1", main.SendMessageRequest(content="hi"))
            async for _ in response.body_iterator:
                pass
            return response

        return asyncio.run(run())

    def test_searches_overlap(self):
        active = 0
        peak = 0

        async def slow(*_args, **_kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
            active -= 1
            return [{"content": "x"}]

        with (
            patch.object(main.memory, "search_memories", slow),
            patch.object(main.memory, "search_vault", slow),
            patch.object(main.memory, "search_style_examples", slow),
        ):
            self._send()

        self.assertEqual(peak, 3, "retrieval searches ran sequentially")

    def test_one_failing_collection_does_not_lose_the_others(self):
        async def ok(*_args, **_kwargs):
            return [{"content": "x"}, {"content": "y"}]

        async def boom(*_args, **_kwargs):
            raise main.httpx.HTTPError("chroma is down")

        with (
            patch.object(main.memory, "search_memories", ok),
            patch.object(main.memory, "search_vault", boom),
            patch.object(main.memory, "search_style_examples", ok),
        ):
            response = self._send()

        self.assertEqual(response.headers["X-Memory-Count"], "2")
        self.assertEqual(response.headers["X-History-Count"], "0")


class ToolRunTimestampRoundTripTests(unittest.TestCase):
    """The Talk tab filters tool runs by `?after=`, and the query compares
    ISO strings, so the value it sends has to be spelled exactly the way
    stored rows are. That's why the turn's start time is issued by the
    server (X-Turn-Started) instead of by the browser: Python writes
    "...710898+00:00" while JS toISOString() gives "...710Z", and those two
    sort against each other wrongly -- a browser-issued value silently
    filtered out the very rows it was meant to select. Real DB, real HTTP,
    because the bug lived precisely in the seam between them.
    """

    def setUp(self):
        import sqlite3
        import tempfile
        from pathlib import Path

        from fastapi.testclient import TestClient

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = patch.object(main.config, "DB_PATH", Path(tmp.name) / "test.db")
        path.start()
        self.addCleanup(path.stop)
        main.db.init_db()
        self.sqlite3 = sqlite3
        self.client = TestClient(main.app)

    def test_custom_headers_are_exposed_to_the_browser(self):
        """The frontend and sidecar are always different origins, so a
        custom response header the JS reads is invisible unless CORS exposes
        it -- and the failure is silent (the value reads back as null, no
        error anywhere). Every X- header the frontend reads must be listed.
        """
        exposed = {
            header.strip()
            for middleware in main.app.user_middleware
            if middleware.cls is main.CORSMiddleware
            for header in middleware.kwargs.get("expose_headers", [])
        }
        self.assertEqual(exposed, {"X-Memory-Count", "X-History-Count", "X-Turn-Started"})

    def test_header_timestamp_selects_the_turns_tool_runs(self):
        conversation = main.db.create_conversation()

        async def scripted(messages, tools=None):
            if any(message.get("role") == "tool" for message in messages):
                yield {"type": "token", "text": "It's Tuesday."}
            else:
                yield {
                    "type": "tool_calls",
                    "calls": [{"function": {"name": "get_current_datetime", "arguments": {}}}],
                }

        with (
            patch.object(main.ollama, "chat_stream", scripted),
            patch.object(main.ollama, "embed", side_effect=main.httpx.HTTPError("skip")),
            patch.object(main.persona, "build_system_prompt", return_value="SYS"),
            patch.object(main.config, "DEFAULT_MODEL", "qwen2.5"),
        ):
            response = self.client.post(
                f"/conversations/{conversation['id']}/messages", json={"content": "what day is it"}
            )
            self.assertEqual(response.text, "It's Tuesday.")
            started = response.headers["X-Turn-Started"]

            filtered = self.client.get(
                f"/conversations/{conversation['id']}/tool-runs", params={"after": started}
            ).json()["tool_runs"]

        self.assertEqual(len(filtered), 1, "the turn's own tool run was filtered out by its own timestamp")
        self.assertEqual(filtered[0]["tool_name"], "get_current_datetime")
        self.assertEqual(filtered[0]["arguments"], {})


if __name__ == "__main__":
    unittest.main()
