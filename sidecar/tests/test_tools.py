import asyncio
import unittest
from datetime import UTC, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from app import config, main, tools


class ToolDispatchTests(unittest.TestCase):
    """dispatch() is the boundary between a model's guess at a function call
    and code that actually runs, so it has to be total: any name, any
    arguments, never an exception. It runs inside the streaming generator --
    anything that escapes kills a reply mid-sentence."""

    def setUp(self):
        self.store: dict[str, str] = {}
        get = patch.object(tools.db, "get_setting", side_effect=lambda k, d: self.store.get(k, d))
        put = patch.object(tools.db, "set_setting", side_effect=self.store.__setitem__)
        get.start()
        put.start()
        self.addCleanup(get.stop)
        self.addCleanup(put.stop)

    def test_known_tool_runs(self):
        result = asyncio.run(tools.dispatch("get_current_datetime", {}))
        self.assertIn(str(datetime.now(UTC).year), result)

    def test_unknown_tool_returns_explanation_not_exception(self):
        result = asyncio.run(tools.dispatch("summon_demon", {}))
        self.assertIn("summon_demon", result)

    def test_failing_tool_is_reported_not_raised(self):
        async def boom(_arguments, _context):
            raise RuntimeError("network is on fire")

        with patch.dict(tools._HANDLERS, {"get_current_datetime": boom}):
            result = asyncio.run(tools.dispatch("get_current_datetime", {}))
        self.assertIn("network is on fire", result)

    def test_none_arguments_are_tolerated(self):
        # Ollama omits `arguments` entirely for a no-parameter tool.
        result = asyncio.run(tools.dispatch("get_current_datetime", None))
        self.assertIn(str(datetime.now(UTC).year), result)


class NormalizeCallsTests(unittest.TestCase):
    """Ollama returns `arguments` as an object, but the OpenAI shape it
    mirrors uses a JSON string and models sometimes emit one anyway."""

    def test_object_arguments(self):
        calls = tools.normalize_calls([{"function": {"name": "t", "arguments": {"a": 1}}}])
        self.assertEqual(calls, [("t", {"a": 1})])

    def test_json_string_arguments(self):
        calls = tools.normalize_calls([{"function": {"name": "t", "arguments": '{"a": 1}'}}])
        self.assertEqual(calls, [("t", {"a": 1})])

    def test_unparseable_arguments_degrade_to_empty(self):
        calls = tools.normalize_calls([{"function": {"name": "t", "arguments": "{not json"}}])
        self.assertEqual(calls, [("t", {})])

    def test_nameless_call_is_dropped(self):
        self.assertEqual(tools.normalize_calls([{"function": {"arguments": {}}}]), [])


class ConcurrencyTests(unittest.TestCase):
    """A round's calls must overlap. "Weather in Lisbon and Nairobi" is two
    independent network calls; running them in sequence doubles a stall the
    user is sitting and watching."""

    def test_calls_in_one_round_overlap(self):
        active = 0
        peak = 0

        async def slow(_arguments, _context):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
            active -= 1
            return "ok"

        with patch.dict(tools._HANDLERS, {"a": slow, "b": slow}, clear=True):
            results = asyncio.run(tools.dispatch_all([("a", {}), ("b", {})]))

        self.assertEqual(results, ["ok", "ok"])
        self.assertEqual(peak, 2, "tool calls in one round ran sequentially")


class ClockTests(unittest.TestCase):
    def setUp(self):
        self.store: dict[str, str] = {}
        get = patch.object(tools.db, "get_setting", side_effect=lambda k, d: self.store.get(k, d))
        get.start()
        self.addCleanup(get.stop)

    def test_configured_timezone_is_used(self):
        self.store["tools_timezone"] = "Europe/Vienna"
        self.assertEqual(tools.resolve_timezone(), ZoneInfo("Europe/Vienna"))

    def test_unknown_timezone_falls_back_to_system_local(self):
        # A typo in a setting must not leave the twin unable to tell the time.
        self.store["tools_timezone"] = "Middle/Earth"
        self.assertIsNone(tools.resolve_timezone())

    def test_format_is_minute_resolution(self):
        """The formatted stamp goes into the system prompt on every turn. At
        second resolution it would differ on every message, invalidating
        Ollama's KV cache for everything below it and forcing a full
        re-prefill each time. Two moments in the same minute must format
        identically."""
        zone = ZoneInfo("Europe/Vienna")
        first = datetime(2026, 3, 14, 9, 30, 1, tzinfo=zone)
        second = datetime(2026, 3, 14, 9, 30, 59, tzinfo=zone)
        self.assertEqual(tools.format_datetime(first), tools.format_datetime(second))


class AvailableSpecsTests(unittest.TestCase):
    def setUp(self):
        self.store: dict[str, str] = {}
        get = patch.object(tools.db, "get_setting", side_effect=lambda k, d: self.store.get(k, d))
        put = patch.object(tools.db, "set_setting", side_effect=self.store.__setitem__)
        get.start()
        put.start()
        self.addCleanup(get.stop)
        self.addCleanup(put.stop)

    def test_enabled_by_default(self):
        names = [spec["function"]["name"] for spec in tools.available_specs()]
        self.assertIn("get_current_datetime", names)

    def test_disabled_feature_is_not_offered_to_the_model(self):
        tools.set_feature_enabled(tools.FEATURE_CLOCK, False)
        names = [spec["function"]["name"] for spec in tools.available_specs()]
        self.assertNotIn("get_current_datetime", names)


class ModelCapabilityTests(unittest.TestCase):
    """Ollama silently ignores `tools` for a model that can't use them, so
    the guard has to be ours -- otherwise a wrong model reads as "the twin
    just never uses its tools" with nothing in the logs."""

    def test_tool_capable_models(self):
        for model in ("qwen2.5", "qwen2.5:7b", "llama3.1:8b-instruct-q4_0", "QWEN3"):
            with self.subTest(model=model):
                self.assertTrue(config.model_supports_tools(model))

    def test_non_tool_capable_models(self):
        for model in ("llama3", "llama3:8b", "phi3", "gemma2"):
            with self.subTest(model=model):
                self.assertFalse(config.model_supports_tools(model))

    def test_llama3_prefix_does_not_match_llama3_1(self):
        # Guards the prefix match: "llama3" must not be dragged in by the
        # "llama3.1" entry, nor vice versa.
        self.assertFalse(config.model_supports_tools("llama3"))
        self.assertTrue(config.model_supports_tools("llama3.1"))


class ToolSettingsEndpointTests(unittest.TestCase):
    def setUp(self):
        self.store: dict[str, str] = {}
        get = patch.object(main.db, "get_setting", side_effect=lambda k, d: self.store.get(k, d))
        put = patch.object(main.db, "set_setting", side_effect=self.store.__setitem__)
        get.start()
        put.start()
        self.addCleanup(get.stop)
        self.addCleanup(put.stop)

    def test_reports_whether_the_configured_model_can_use_tools(self):
        payload = main.get_tools()
        self.assertEqual(payload["model"], config.DEFAULT_MODEL)
        self.assertEqual(payload["model_supports_tools"], config.model_supports_tools(config.DEFAULT_MODEL))

    def test_toggle_and_timezone_update_independently(self):
        main.update_tools(main.ToolSettingsRequest(timezone="  Europe/Vienna  "))
        self.assertEqual(self.store["tools_timezone"], "Europe/Vienna")
        self.assertTrue(main.get_tools()["clock_enabled"])

        result = main.update_tools(main.ToolSettingsRequest(clock_enabled=False))
        self.assertFalse(result["clock_enabled"])
        self.assertEqual(result["timezone"], "Europe/Vienna")

    def test_weather_defaults_to_no_locations(self):
        # No default city ships with the app -- there's no generic choice
        # that isn't either an arbitrary opinion or a leak of wherever
        # whoever set this up actually lives. The owner adds their own.
        payload = main.get_tools()
        self.assertTrue(payload["weather_enabled"])
        self.assertEqual(payload["weather_locations"], [])

    def test_weather_locations_round_trip(self):
        result = main.update_tools(
            main.ToolSettingsRequest(
                weather_locations=[main.WeatherLocation(name="Berlin", latitude=52.52, longitude=13.405)]
            )
        )
        self.assertEqual(result["weather_locations"], [{"name": "Berlin", "latitude": 52.52, "longitude": 13.405}])
        # Unrelated settings (clock) must survive a weather-only update.
        self.assertTrue(result["clock_enabled"])

    def test_weather_enabled_toggle_does_not_touch_locations(self):
        main.update_tools(
            main.ToolSettingsRequest(
                weather_locations=[main.WeatherLocation(name="Berlin", latitude=52.52, longitude=13.405)]
            )
        )
        result = main.update_tools(main.ToolSettingsRequest(weather_enabled=False))
        self.assertFalse(result["weather_enabled"])
        self.assertEqual([loc["name"] for loc in result["weather_locations"]], ["Berlin"])

    def test_empty_weather_locations_list_is_rejected(self):
        with self.assertRaises(main.HTTPException) as caught:
            main.update_tools(main.ToolSettingsRequest(weather_locations=[]))
        self.assertEqual(caught.exception.status_code, 400)
        self.assertNotIn("tools_weather_locations", self.store)

    def test_calendar_confirm_defaults_on(self):
        self.assertTrue(main.get_tools()["calendar_confirm"])

    def test_calendar_settings_round_trip_independently(self):
        main.update_tools(main.ToolSettingsRequest(calendar_confirm=False))
        result = main.update_tools(main.ToolSettingsRequest(calendar_enabled=False))
        self.assertFalse(result["calendar_enabled"])
        self.assertFalse(result["calendar_confirm"], "toggling calendar_enabled must not reset calendar_confirm")

    def test_calendar_configured_and_connected_reflect_google_calendar_state(self):
        self.assertFalse(main.get_tools()["calendar_configured"])
        self.assertFalse(main.get_tools()["calendar_connected"])
        main.google_calendar.set_client_credentials("cid", "csecret")
        self.assertTrue(main.get_tools()["calendar_configured"])
        self.assertFalse(main.get_tools()["calendar_connected"])
        main.google_calendar._save_tokens({"access_token": "AT", "refresh_token": "RT", "expires_at": 0})
        self.assertTrue(main.get_tools()["calendar_connected"])


class CalendarHandlerTests(unittest.TestCase):
    def setUp(self):
        self.store: dict[str, str] = {}
        get = patch.object(tools.db, "get_setting", side_effect=lambda k, d: self.store.get(k, d))
        put = patch.object(tools.db, "set_setting", side_effect=self.store.__setitem__)
        get.start()
        put.start()
        self.addCleanup(get.stop)
        self.addCleanup(put.stop)

    def test_list_upcoming_events_when_not_connected(self):
        with patch.object(tools.google_calendar, "is_connected", return_value=False):
            result = asyncio.run(tools.dispatch("list_upcoming_events", {}))
        self.assertIn("isn't connected", result)

    def test_list_upcoming_events_formats_prose(self):
        events = [
            {"summary": "Standup", "start": {"dateTime": "2026-08-10T09:00:00"}},
            {"summary": "Holiday", "start": {"date": "2026-08-12"}},
        ]
        with (
            patch.object(tools.google_calendar, "is_connected", return_value=True),
            patch.object(tools.google_calendar, "list_upcoming_events", return_value=events),
        ):
            result = asyncio.run(tools.dispatch("list_upcoming_events", {}))
        self.assertIn("Standup", result)
        self.assertIn("2026-08-10T09:00:00", result)
        self.assertIn("Holiday", result)
        self.assertIn("2026-08-12", result)

    def test_list_upcoming_events_when_empty(self):
        with (
            patch.object(tools.google_calendar, "is_connected", return_value=True),
            patch.object(tools.google_calendar, "list_upcoming_events", return_value=[]),
        ):
            result = asyncio.run(tools.dispatch("list_upcoming_events", {}))
        self.assertIn("nothing coming up", result)

    def test_create_event_when_not_connected(self):
        with patch.object(tools.google_calendar, "is_connected", return_value=False):
            result = asyncio.run(
                tools.dispatch("create_calendar_event", {"title": "x", "start_at": "2026-08-10T10:00:00"})
            )
        self.assertIn("isn't connected", result)

    def test_create_event_requires_title_and_start(self):
        with patch.object(tools.google_calendar, "is_connected", return_value=True):
            result = asyncio.run(tools.dispatch("create_calendar_event", {"title": "only a title"}))
        self.assertIn("start time", result)

    def test_create_event_drafts_a_pending_row_when_confirm_is_on(self):
        # unset tools_calendar_confirm defaults to "1" (confirm-first), per main.py.
        created = {}

        def fake_create_pending(conversation_id, title, start_at, duration_minutes, description=""):
            created.update(
                conversation_id=conversation_id,
                title=title,
                start_at=start_at,
                duration_minutes=duration_minutes,
                description=description,
            )
            return created

        with (
            patch.object(tools.google_calendar, "is_connected", return_value=True),
            patch.object(tools.db, "create_pending_calendar_event", fake_create_pending),
        ):
            result = asyncio.run(
                tools.dispatch(
                    "create_calendar_event",
                    {"title": "Dentist", "start_at": "2026-08-10T14:00:00", "duration_minutes": 30},
                    context={"conversation_id": "conv-1"},
                )
            )
        self.assertIn("Drafted", result)
        self.assertIn("confirm", result.lower())
        self.assertEqual(created["conversation_id"], "conv-1")
        self.assertEqual(created["title"], "Dentist")
        self.assertEqual(created["duration_minutes"], 30)

    def test_create_event_without_conversation_context_is_refused(self):
        with patch.object(tools.google_calendar, "is_connected", return_value=True):
            result = asyncio.run(
                tools.dispatch("create_calendar_event", {"title": "x", "start_at": "2026-08-10T10:00:00"})
            )
        self.assertIn("no active conversation", result)

    def test_create_event_calls_google_directly_when_confirm_is_off(self):
        self.store["tools_calendar_confirm"] = "0"
        with (
            patch.object(tools.google_calendar, "is_connected", return_value=True),
            patch.object(tools.google_calendar, "create_event", return_value={"htmlLink": "https://cal/x"}) as create_event,
        ):
            result = asyncio.run(
                tools.dispatch(
                    "create_calendar_event",
                    {"title": "Dentist", "start_at": "2026-08-10T14:00:00"},
                    context={"conversation_id": "conv-1"},
                )
            )
        create_event.assert_awaited_once_with("Dentist", "2026-08-10T14:00:00", 60, "")
        self.assertIn("Created", result)
        self.assertIn("https://cal/x", result)

    def test_create_event_defaults_duration_when_unparseable(self):
        self.store["tools_calendar_confirm"] = "0"
        with (
            patch.object(tools.google_calendar, "is_connected", return_value=True),
            patch.object(tools.google_calendar, "create_event", return_value={}) as create_event,
        ):
            asyncio.run(
                tools.dispatch(
                    "create_calendar_event",
                    {"title": "x", "start_at": "2026-08-10T10:00:00", "duration_minutes": "not-a-number"},
                    context={"conversation_id": "conv-1"},
                )
            )
        create_event.assert_awaited_once_with("x", "2026-08-10T10:00:00", 60, "")

    def test_create_event_reports_google_calendar_errors(self):
        self.store["tools_calendar_confirm"] = "0"
        with (
            patch.object(tools.google_calendar, "is_connected", return_value=True),
            patch.object(
                tools.google_calendar,
                "create_event",
                side_effect=tools.google_calendar.GoogleCalendarUnavailable("reconnect please"),
            ),
        ):
            result = asyncio.run(
                tools.dispatch(
                    "create_calendar_event",
                    {"title": "x", "start_at": "2026-08-10T10:00:00"},
                    context={"conversation_id": "conv-1"},
                )
            )
        self.assertIn("reconnect please", result)


class AvailableSpecsCalendarTests(unittest.TestCase):
    def setUp(self):
        self.store: dict[str, str] = {}
        get = patch.object(tools.db, "get_setting", side_effect=lambda k, d: self.store.get(k, d))
        put = patch.object(tools.db, "set_setting", side_effect=self.store.__setitem__)
        get.start()
        put.start()
        self.addCleanup(get.stop)
        self.addCleanup(put.stop)

    def test_calendar_tools_are_offered_by_default(self):
        names = [spec["function"]["name"] for spec in tools.available_specs()]
        self.assertIn("list_upcoming_events", names)
        self.assertIn("create_calendar_event", names)

    def test_disabled_calendar_is_not_offered(self):
        tools.set_feature_enabled(tools.FEATURE_CALENDAR, False)
        names = [spec["function"]["name"] for spec in tools.available_specs()]
        self.assertNotIn("list_upcoming_events", names)
        self.assertNotIn("create_calendar_event", names)

    def test_calendar_spec_descriptions_use_the_configured_owner_name(self):
        self.store["owner_name"] = "Priya"
        specs = {spec["function"]["name"]: spec for spec in tools.available_specs()}
        self.assertIn("Priya's", specs["list_upcoming_events"]["function"]["description"])
        self.assertIn("Priya's", specs["create_calendar_event"]["function"]["description"])

    def test_calendar_spec_descriptions_fall_back_to_a_neutral_name(self):
        specs = {spec["function"]["name"]: spec for spec in tools.available_specs()}
        self.assertIn("the owner's", specs["list_upcoming_events"]["function"]["description"])
        self.assertIn("the owner's", specs["create_calendar_event"]["function"]["description"])
        self.assertNotIn("Alex", specs["list_upcoming_events"]["function"]["description"])
        self.assertNotIn("Alex", specs["create_calendar_event"]["function"]["description"])


class CalendarOwnerNameTests(unittest.TestCase):
    """The calendar handlers used to hardcode "Alex" in strings the model
    and user actually see, even though owner_name is configurable -- anyone
    else running the app got a twin that referred to a stranger."""

    def setUp(self):
        self.store: dict[str, str] = {}
        get = patch.object(tools.db, "get_setting", side_effect=lambda k, d: self.store.get(k, d))
        put = patch.object(tools.db, "set_setting", side_effect=self.store.__setitem__)
        get.start()
        put.start()
        self.addCleanup(get.stop)
        self.addCleanup(put.stop)

    def test_not_connected_message_uses_the_configured_owner_name(self):
        self.store["owner_name"] = "Priya"
        with patch.object(tools.google_calendar, "is_connected", return_value=False):
            result = asyncio.run(tools.dispatch("list_upcoming_events", {}))
        self.assertIn("ask Priya", result)
        self.assertNotIn("Alex", result)

    def test_drafted_event_message_uses_the_configured_owner_name(self):
        self.store["owner_name"] = "Priya"
        with (
            patch.object(tools.google_calendar, "is_connected", return_value=True),
            patch.object(tools.db, "create_pending_calendar_event", return_value=None),
        ):
            result = asyncio.run(
                tools.dispatch(
                    "create_calendar_event",
                    {"title": "Dentist", "start_at": "2026-08-10T14:00:00"},
                    context={"conversation_id": "conv-1"},
                )
            )
        self.assertIn("Priya needs to confirm", result)
        self.assertNotIn("Alex", result)

    def test_owner_name_falls_back_to_a_neutral_default_when_unset(self):
        with patch.object(tools.google_calendar, "is_connected", return_value=False):
            result = asyncio.run(tools.dispatch("list_upcoming_events", {}))
        self.assertIn("ask the owner", result)
        self.assertNotIn("Alex", result)


if __name__ == "__main__":
    unittest.main()
