import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app import google_calendar, main


class PendingCalendarEventRoutesTests(unittest.TestCase):
    """Real DB, real HTTP -- these routes exist specifically because
    tool_runs' after-filter bug (a browser timestamp compared as a string
    against a server one) showed that mocking the DB layer can hide exactly
    this kind of seam bug. The pending-events list uses the identical
    after-filter shape, so it's worth the same rigor."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = patch.object(main.config, "DB_PATH", Path(tmp.name) / "test.db")
        path.start()
        self.addCleanup(path.stop)
        main.db.init_db()
        self.client = TestClient(main.app)

    def test_pending_events_404_for_missing_conversation(self):
        response = self.client.get("/conversations/does-not-exist/calendar/pending")
        self.assertEqual(response.status_code, 404)

    def test_empty_by_default(self):
        conversation = main.db.create_conversation()
        response = self.client.get(f"/conversations/{conversation['id']}/calendar/pending")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"pending_events": []})

    def test_after_filters_to_events_from_that_turn_onward(self):
        conversation = main.db.create_conversation()
        first = main.db.create_pending_calendar_event(conversation["id"], "Old", "2026-08-10T10:00:00", 30)
        second = main.db.create_pending_calendar_event(conversation["id"], "New", "2026-08-11T10:00:00", 30)

        response = self.client.get(
            f"/conversations/{conversation['id']}/calendar/pending", params={"after": second["created_at"]}
        )
        titles = [event["title"] for event in response.json()["pending_events"]]
        self.assertEqual(titles, ["New"])
        self.assertNotIn(first["id"], [e["id"] for e in response.json()["pending_events"]])

    def test_confirm_creates_the_event_and_removes_the_draft(self):
        conversation = main.db.create_conversation()
        pending = main.db.create_pending_calendar_event(
            conversation["id"], "Dentist", "2026-08-10T14:00:00", 30, "bring card"
        )

        with patch.object(
            main.google_calendar, "create_event", return_value={"htmlLink": "https://cal/x"}
        ) as create_event:
            response = self.client.post(f"/calendar/pending/{pending['id']}/confirm")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"html_link": "https://cal/x"})
        create_event.assert_awaited_once_with("Dentist", "2026-08-10T14:00:00", 30, "bring card")
        self.assertIsNone(main.db.get_pending_calendar_event(pending["id"]))

    def test_confirm_404_for_a_gone_draft(self):
        response = self.client.post("/calendar/pending/does-not-exist/confirm")
        self.assertEqual(response.status_code, 404)

    def test_confirm_surfaces_not_connected_as_400_and_keeps_the_draft(self):
        conversation = main.db.create_conversation()
        pending = main.db.create_pending_calendar_event(conversation["id"], "x", "2026-08-10T10:00:00", 30)

        with patch.object(
            main.google_calendar,
            "create_event",
            side_effect=google_calendar.GoogleCalendarUnavailable("not connected"),
        ):
            response = self.client.post(f"/calendar/pending/{pending['id']}/confirm")

        self.assertEqual(response.status_code, 400)
        self.assertIsNotNone(main.db.get_pending_calendar_event(pending["id"]))

    def test_confirm_surfaces_a_google_failure_as_502_and_keeps_the_draft(self):
        conversation = main.db.create_conversation()
        pending = main.db.create_pending_calendar_event(conversation["id"], "x", "2026-08-10T10:00:00", 30)

        with patch.object(
            main.google_calendar, "create_event", side_effect=httpx.HTTPError("boom")
        ):
            response = self.client.post(f"/calendar/pending/{pending['id']}/confirm")

        self.assertEqual(response.status_code, 502)
        self.assertIsNotNone(main.db.get_pending_calendar_event(pending["id"]))

    def test_discard_removes_the_draft(self):
        conversation = main.db.create_conversation()
        pending = main.db.create_pending_calendar_event(conversation["id"], "x", "2026-08-10T10:00:00", 30)

        response = self.client.delete(f"/calendar/pending/{pending['id']}")

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(main.db.get_pending_calendar_event(pending["id"]))

    def test_discard_is_idempotent_for_an_already_gone_draft(self):
        response = self.client.delete("/calendar/pending/never-existed")
        self.assertEqual(response.status_code, 200)


class GoogleConnectRoutesTests(unittest.TestCase):
    def setUp(self):
        self.store: dict[str, str] = {}
        get = patch.object(main.db, "get_setting", side_effect=lambda k, d: self.store.get(k, d))
        put = patch.object(main.db, "set_setting", side_effect=self.store.__setitem__)
        get.start()
        put.start()
        self.addCleanup(get.stop)
        self.addCleanup(put.stop)
        google_calendar._session = None
        self.addCleanup(lambda: setattr(google_calendar, "_session", None))
        self.client = TestClient(main.app)

    def test_status_when_unconfigured(self):
        response = self.client.get("/tools/google/status")
        self.assertEqual(
            response.json(), {"configured": False, "connected": False, "connecting": False, "error": None}
        )

    def test_credentials_require_both_fields(self):
        response = self.client.put(
            "/tools/google/credentials", json={"client_id": "cid", "client_secret": "  "}
        )
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("google_client_id", self.store)

    def test_saving_credentials_makes_status_configured(self):
        response = self.client.put(
            "/tools/google/credentials", json={"client_id": "cid", "client_secret": "csecret"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["configured"])

    def test_connect_without_credentials_is_rejected(self):
        response = self.client.post("/tools/google/connect")
        self.assertEqual(response.status_code, 400)

    def test_connect_returns_an_auth_url_and_status_reports_connecting(self):
        # TestClient opens a fresh event loop per request by default, which
        # would cancel (and so finalize) the background listener task
        # before a follow-up call could ever observe "connecting" -- using
        # it as a context manager shares one portal across both calls here,
        # the same as the one persistent loop the real ASGI server runs.
        with TestClient(main.app) as client:
            client.put("/tools/google/credentials", json={"client_id": "cid", "client_secret": "csecret"})
            response = client.post("/tools/google/connect")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["auth_url"].startswith(google_calendar.AUTH_URL))
            self.assertTrue(client.get("/tools/google/status").json()["connecting"])
        google_calendar.disconnect()

    def test_disconnect_clears_a_connected_account(self):
        google_calendar._save_tokens({"access_token": "AT", "refresh_token": "RT", "expires_at": 0})
        response = self.client.post("/tools/google/disconnect")
        self.assertFalse(response.json()["connected"])


if __name__ == "__main__":
    unittest.main()
