import asyncio
import time
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import httpx

from app import google_calendar as gc


class CredentialsTests(unittest.TestCase):
    def setUp(self):
        self.store: dict[str, str] = {}
        get = patch.object(gc.db, "get_setting", side_effect=lambda k, d: self.store.get(k, d))
        put = patch.object(gc.db, "set_setting", side_effect=self.store.__setitem__)
        get.start()
        put.start()
        self.addCleanup(get.stop)
        self.addCleanup(put.stop)

    def test_unconfigured_by_default(self):
        self.assertIsNone(gc._resolve_client_credentials())

    def test_in_app_credentials_are_used(self):
        gc.set_client_credentials("  cid  ", "  csecret  ")
        self.assertEqual(gc._resolve_client_credentials(), ("cid", "csecret"))

    def test_in_app_credentials_take_precedence_over_env(self):
        gc.set_client_credentials("app-id", "app-secret")
        with patch.object(gc.config, "GOOGLE_CLIENT_ID", "env-id"), patch.object(
            gc.config, "GOOGLE_CLIENT_SECRET", "env-secret"
        ):
            self.assertEqual(gc._resolve_client_credentials(), ("app-id", "app-secret"))

    def test_env_credentials_are_a_fallback(self):
        with patch.object(gc.config, "GOOGLE_CLIENT_ID", "env-id"), patch.object(
            gc.config, "GOOGLE_CLIENT_SECRET", "env-secret"
        ):
            self.assertEqual(gc._resolve_client_credentials(), ("env-id", "env-secret"))

    def test_partial_credentials_are_treated_as_unconfigured(self):
        gc.set_client_credentials("only-id", "")
        self.assertIsNone(gc._resolve_client_credentials())


class TokenStorageTests(unittest.TestCase):
    def setUp(self):
        self.store: dict[str, str] = {}
        get = patch.object(gc.db, "get_setting", side_effect=lambda k, d: self.store.get(k, d))
        put = patch.object(gc.db, "set_setting", side_effect=self.store.__setitem__)
        get.start()
        put.start()
        self.addCleanup(get.stop)
        self.addCleanup(put.stop)

    def test_not_connected_with_no_tokens(self):
        self.assertFalse(gc.is_connected())

    def test_connected_once_a_refresh_token_is_stored(self):
        gc._save_tokens({"access_token": "AT", "refresh_token": "RT", "expires_at": 0})
        self.assertTrue(gc.is_connected())

    def test_corrupt_token_json_is_treated_as_disconnected(self):
        self.store["google_calendar_tokens"] = "{not json"
        self.assertFalse(gc.is_connected())

    def test_disconnect_clears_tokens(self):
        gc._save_tokens({"access_token": "AT", "refresh_token": "RT", "expires_at": 0})
        gc.disconnect()
        self.assertFalse(gc.is_connected())


class ConnectSessionTests(unittest.TestCase):
    """The loopback listener is the one genuinely new piece of machinery
    here -- these drive it with a real TCP connection standing in for the
    browser's redirect, since that's the actual interface being tested
    (parsing a raw HTTP request line, matching state, tearing the listener
    down), not something mockable at the httpx layer."""

    def setUp(self):
        self.store: dict[str, str] = {}
        get = patch.object(gc.db, "get_setting", side_effect=lambda k, d: self.store.get(k, d))
        put = patch.object(gc.db, "set_setting", side_effect=self.store.__setitem__)
        get.start()
        put.start()
        self.addCleanup(get.stop)
        self.addCleanup(put.stop)
        gc.set_client_credentials("cid", "csecret")
        gc._session = None
        self.addCleanup(lambda: setattr(gc, "_session", None))

    async def _hit_callback(self, port: int, query: str) -> bytes:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(f"GET /callback?{query} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        await writer.drain()
        response = await reader.read(500)
        writer.close()
        await writer.wait_closed()
        return response

    def test_raises_when_not_configured(self):
        self.store.clear()  # undo set_client_credentials from setUp
        with self.assertRaises(gc.GoogleCalendarUnavailable):
            asyncio.run(gc.start_connect())

    def test_auth_url_carries_the_generated_state_and_loopback_redirect(self):
        async def scenario():
            auth_url = await gc.start_connect()
            self.assertIn("client_id=cid", auth_url)
            self.assertIn("127.0.0.1", auth_url)
            self.assertIn(f"state={gc._session.state}", auth_url)
            session = gc._session
            gc.disconnect()
            await session.task  # let the background task actually close its socket

        asyncio.run(scenario())

    def test_successful_callback_exchanges_the_code(self):
        async def scenario():
            auth_url = await gc.start_connect()
            port = gc._session.server.sockets[0].getsockname()[1]
            state = gc._session.state

            async def fake_exchange(code, redirect_uri):
                self.assertEqual(code, "GOODCODE")
                self.assertIn(str(port), redirect_uri)
                gc._save_tokens({"access_token": "AT", "refresh_token": "RT", "expires_at": time.time() + 3600})

            with patch.object(gc, "_exchange_code", fake_exchange):
                response = await self._hit_callback(port, f"code=GOODCODE&state={state}")
                await gc._session.task

            self.assertIn(b"200 OK", response)
            self.assertTrue(gc._session.done)
            self.assertIsNone(gc._session.error)
            self.assertTrue(gc.is_connected())
            self.assertTrue(auth_url.startswith(gc.AUTH_URL))

        asyncio.run(scenario())

    def test_state_mismatch_is_rejected(self):
        async def scenario():
            await gc.start_connect()
            port = gc._session.server.sockets[0].getsockname()[1]

            with patch.object(gc, "_exchange_code", side_effect=AssertionError("must not exchange")):
                await self._hit_callback(port, "code=SOMECODE&state=wrong-state")
                await gc._session.task

            self.assertIsNotNone(gc._session.error)
            self.assertIn("state", gc._session.error.lower())
            self.assertFalse(gc.is_connected())

        asyncio.run(scenario())

    def test_google_declining_is_surfaced_as_an_error(self):
        async def scenario():
            await gc.start_connect()
            port = gc._session.server.sockets[0].getsockname()[1]
            state = gc._session.state

            await self._hit_callback(port, f"error=access_denied&state={state}")
            await gc._session.task

            self.assertIn("access_denied", gc._session.error)

        asyncio.run(scenario())

    def test_a_second_connect_while_one_is_in_flight_reuses_it(self):
        async def scenario():
            first_url = await gc.start_connect()
            first_session = gc._session
            second_url = await gc.start_connect()
            self.assertIs(gc._session, first_session)
            self.assertEqual(first_url, second_url)
            gc.disconnect()
            await first_session.task

        asyncio.run(scenario())

    def test_timeout_waiting_for_the_redirect(self):
        async def scenario():
            with patch.object(gc.config, "GOOGLE_OAUTH_CALLBACK_TIMEOUT", 0.05):
                await gc.start_connect()
                await gc._session.task
            self.assertIn("timed out", gc._session.error.lower())

        asyncio.run(scenario())

    def test_disconnect_while_connecting_tears_the_listener_down(self):
        async def scenario():
            await gc.start_connect()
            session = gc._session
            gc.disconnect()
            await session.task
            self.assertTrue(session.done)
            self.assertIsNone(gc._session)

        asyncio.run(scenario())


class AccessTokenTests(unittest.TestCase):
    def setUp(self):
        self.store: dict[str, str] = {}
        get = patch.object(gc.db, "get_setting", side_effect=lambda k, d: self.store.get(k, d))
        put = patch.object(gc.db, "set_setting", side_effect=self.store.__setitem__)
        get.start()
        put.start()
        self.addCleanup(get.stop)
        self.addCleanup(put.stop)
        gc.set_client_credentials("cid", "csecret")

        self.track_calls: list[str] = []

        @contextmanager
        def fake_track(label: str):
            self.track_calls.append(label)
            yield

        track_patch = patch.object(gc.network_activity, "track", fake_track)
        track_patch.start()
        self.addCleanup(track_patch.stop)

    def _install_transport(self, handler):
        real_async_client = httpx.AsyncClient

        def factory(*_args, **_kwargs):
            return real_async_client(transport=httpx.MockTransport(handler))

        patcher = patch.object(gc.httpx, "AsyncClient", factory)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_raises_when_never_connected(self):
        with self.assertRaises(gc.GoogleCalendarUnavailable):
            asyncio.run(gc._access_token())

    def test_unexpired_cached_token_is_reused_without_a_network_call(self):
        gc._save_tokens({"access_token": "CACHED", "refresh_token": "RT", "expires_at": time.time() + 3600})
        self._install_transport(lambda request: (_ for _ in ()).throw(AssertionError("must not refresh")))
        token = asyncio.run(gc._access_token())
        self.assertEqual(token, "CACHED")

    def test_expired_token_is_refreshed_and_the_refresh_token_is_preserved(self):
        gc._save_tokens({"access_token": "STALE", "refresh_token": "RT", "expires_at": time.time() - 10})

        def handler(request: httpx.Request) -> httpx.Response:
            body = request.read().decode()
            self.assertIn("grant_type=refresh_token", body)
            self.assertIn("refresh_token=RT", body)
            return httpx.Response(200, json={"access_token": "FRESH", "expires_in": 3600})

        self._install_transport(handler)
        token = asyncio.run(gc._access_token())
        self.assertEqual(token, "FRESH")
        self.assertEqual(self.track_calls, ["Google: refreshing calendar access"])

        stored = gc._load_tokens()
        self.assertEqual(stored["refresh_token"], "RT", "a refresh response never repeats it -- must be carried forward")
        self.assertEqual(stored["access_token"], "FRESH")


class ResolveCalendarIdTests(unittest.TestCase):
    """The one gate between "the model asked for an event" and "an event
    actually got created somewhere" -- a miss here has to come back None,
    never a guess, since guessing means writing to the wrong calendar."""

    def test_exact_name_match_is_case_insensitive(self):
        calendars = [{"id": "1", "summary": "Tasks"}, {"id": "2", "summary": "Work"}]
        self.assertEqual(gc._resolve_calendar_id("tasks", calendars), "1")

    def test_exact_id_match(self):
        calendars = [{"id": "abc@group.calendar.google.com", "summary": "Something"}]
        self.assertEqual(
            gc._resolve_calendar_id("ABC@group.calendar.google.com", calendars),
            "abc@group.calendar.google.com",
        )

    def test_substring_match_on_summary(self):
        calendars = [{"id": "1", "summary": "Alex's Tasks List"}]
        self.assertEqual(gc._resolve_calendar_id("tasks", calendars), "1")

    def test_no_match_returns_none_rather_than_a_guess(self):
        calendars = [{"id": "1", "summary": "Work"}]
        self.assertIsNone(gc._resolve_calendar_id("Tasks", calendars))


class WriteCalendarNameTests(unittest.TestCase):
    def setUp(self):
        self.store: dict[str, str] = {}
        get = patch.object(gc.db, "get_setting", side_effect=lambda k, d: self.store.get(k, d))
        put = patch.object(gc.db, "set_setting", side_effect=self.store.__setitem__)
        get.start()
        put.start()
        self.addCleanup(get.stop)
        self.addCleanup(put.stop)

    def test_defaults_to_tasks(self):
        self.assertEqual(gc.write_calendar_name(), "Tasks")

    def test_configurable_and_trimmed(self):
        gc.set_write_calendar_name("  Errands  ")
        self.assertEqual(gc.write_calendar_name(), "Errands")


class CalendarOperationsTests(unittest.TestCase):
    def setUp(self):
        self.store: dict[str, str] = {}
        get = patch.object(gc.db, "get_setting", side_effect=lambda k, d: self.store.get(k, d))
        put = patch.object(gc.db, "set_setting", side_effect=self.store.__setitem__)
        get.start()
        put.start()
        self.addCleanup(get.stop)
        self.addCleanup(put.stop)
        gc.set_client_credentials("cid", "csecret")
        gc._save_tokens({"access_token": "AT", "refresh_token": "RT", "expires_at": time.time() + 3600})

        self.track_calls: list[str] = []

        @contextmanager
        def fake_track(label: str):
            self.track_calls.append(label)
            yield

        track_patch = patch.object(gc.network_activity, "track", fake_track)
        track_patch.start()
        self.addCleanup(track_patch.stop)

    def _install_transport(self, handler):
        real_async_client = httpx.AsyncClient

        def factory(*_args, **_kwargs):
            return real_async_client(transport=httpx.MockTransport(handler))

        patcher = patch.object(gc.httpx, "AsyncClient", factory)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _install_calendars_and(self, calendars, events_handler):
        """Routes calendarList.list to `calendars`; everything else (the
        per-calendar events endpoints) goes to `events_handler`."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/calendar/v3/users/me/calendarList":
                return httpx.Response(200, json={"items": calendars})
            return events_handler(request)

        self._install_transport(handler)

    def test_list_calendars_returns_id_and_summary_only(self):
        async def scenario():
            def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200, json={"items": [{"id": "1", "summary": "Work", "accessRole": "owner"}]}
                )

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await gc.list_calendars(client, "AT")

        self.assertEqual(asyncio.run(scenario()), [{"id": "1", "summary": "Work"}])

    def test_create_event_resolves_the_configured_write_calendar_by_name(self):
        calendars = [
            {"id": "primary", "summary": "alex@example.com"},
            {"id": "tasks-id@group.calendar.google.com", "summary": "Tasks"},
        ]
        captured = {}

        def events_handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["auth"] = request.headers.get("authorization")
            captured["body"] = request.read()
            return httpx.Response(200, json={"htmlLink": "https://calendar.google.com/event?x"})

        self._install_calendars_and(calendars, events_handler)
        event = asyncio.run(gc.create_event("Dentist", "2026-08-10T14:00:00", 30, "bring insurance card"))

        self.assertIn("tasks-id@group.calendar.google.com", captured["path"])
        self.assertEqual(captured["auth"], "Bearer AT")
        import json as _json

        body = _json.loads(captured["body"])
        self.assertEqual(body["summary"], "Dentist")
        self.assertEqual(body["start"]["dateTime"], "2026-08-10T14:00:00")
        self.assertEqual(body["end"]["dateTime"], "2026-08-10T14:30:00")
        self.assertEqual(event["htmlLink"], "https://calendar.google.com/event?x")
        self.assertEqual(self.track_calls, ["Google Calendar: creating 'Dentist'"])

    def test_create_event_never_writes_to_primary_even_when_listed_first(self):
        """The one rule this tool has to keep: writes only ever go to the
        configured calendar, regardless of what else exists in the
        account or what order the API happens to list it in."""
        calendars = [{"id": "primary", "summary": "alex@example.com"}, {"id": "tasks-id", "summary": "Tasks"}]
        paths_hit = []

        def events_handler(request: httpx.Request) -> httpx.Response:
            paths_hit.append(request.url.path)
            return httpx.Response(200, json={})

        self._install_calendars_and(calendars, events_handler)
        asyncio.run(gc.create_event("x", "2026-08-10T10:00:00", 30))
        self.assertTrue(all("primary" not in path for path in paths_hit), paths_hit)

    def test_create_event_uses_whatever_write_target_is_configured(self):
        self.store["tools_calendar_write_target"] = "Errands"
        calendars = [{"id": "errands-id", "summary": "Errands"}, {"id": "tasks-id", "summary": "Tasks"}]
        paths_hit = []

        def events_handler(request: httpx.Request) -> httpx.Response:
            paths_hit.append(request.url.path)
            return httpx.Response(200, json={})

        self._install_calendars_and(calendars, events_handler)
        asyncio.run(gc.create_event("x", "2026-08-10T10:00:00", 30))
        self.assertTrue(any("errands-id" in path for path in paths_hit), paths_hit)

    def test_create_event_raises_clearly_when_the_write_calendar_is_missing(self):
        self._install_calendars_and(
            [{"id": "primary", "summary": "alex@example.com"}], lambda request: httpx.Response(200, json={})
        )
        with self.assertRaises(gc.GoogleCalendarUnavailable) as caught:
            asyncio.run(gc.create_event("x", "2026-08-10T10:00:00", 30))
        self.assertIn("Tasks", str(caught.exception))

    def test_list_upcoming_events_merges_and_sorts_across_calendars(self):
        calendars = [{"id": "cal-a", "summary": "Work"}, {"id": "cal-b", "summary": "Personal"}]

        def events_handler(request: httpx.Request) -> httpx.Response:
            if "cal-a" in request.url.path:
                return httpx.Response(
                    200, json={"items": [{"summary": "Later", "start": {"dateTime": "2026-08-10T15:00:00"}}]}
                )
            return httpx.Response(
                200, json={"items": [{"summary": "Sooner", "start": {"dateTime": "2026-08-10T09:00:00"}}]}
            )

        self._install_calendars_and(calendars, events_handler)
        events = asyncio.run(gc.list_upcoming_events())

        self.assertEqual([event["summary"] for event in events], ["Sooner", "Later"])
        self.assertEqual(events[0]["_calendar"], "Personal")
        self.assertEqual(events[1]["_calendar"], "Work")
        self.assertEqual(self.track_calls, ["Google Calendar: checking upcoming events"])

    def test_list_upcoming_events_truncates_to_max_results_after_merging(self):
        calendars = [{"id": "cal-a", "summary": "Work"}, {"id": "cal-b", "summary": "Personal"}]

        def events_handler(request: httpx.Request) -> httpx.Response:
            prefix = "a" if "cal-a" in request.url.path else "b"
            items = [
                {"summary": f"{prefix}{i}", "start": {"dateTime": f"2026-08-10T{10 + i:02d}:00:00"}}
                for i in range(5)
            ]
            return httpx.Response(200, json={"items": items})

        self._install_calendars_and(calendars, events_handler)
        events = asyncio.run(gc.list_upcoming_events(max_results=3))
        self.assertEqual(len(events), 3)

    def test_list_upcoming_events_skips_a_failing_calendar_without_failing_the_rest(self):
        calendars = [{"id": "cal-a", "summary": "Broken"}, {"id": "cal-b", "summary": "Fine"}]

        def events_handler(request: httpx.Request) -> httpx.Response:
            if "cal-a" in request.url.path:
                return httpx.Response(403, json={"error": "forbidden"})
            return httpx.Response(
                200, json={"items": [{"summary": "Standup", "start": {"dateTime": "2026-08-10T09:00:00"}}]}
            )

        self._install_calendars_and(calendars, events_handler)
        events = asyncio.run(gc.list_upcoming_events())
        self.assertEqual([event["summary"] for event in events], ["Standup"])

    def test_list_upcoming_events_fetches_calendars_concurrently(self):
        calendars = [{"id": f"cal-{i}", "summary": f"Cal{i}"} for i in range(3)]
        active = 0
        peak = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal active, peak
            if request.url.path == "/calendar/v3/users/me/calendarList":
                return httpx.Response(200, json={"items": calendars})
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
            active -= 1
            return httpx.Response(200, json={"items": []})

        self._install_transport(handler)
        asyncio.run(gc.list_upcoming_events())
        self.assertEqual(peak, 3, "calendars were fetched serially instead of concurrently")

    def test_list_upcoming_events_with_no_calendars(self):
        self._install_calendars_and([], lambda request: httpx.Response(200, json={"items": []}))
        self.assertEqual(asyncio.run(gc.list_upcoming_events()), [])


if __name__ == "__main__":
    unittest.main()
