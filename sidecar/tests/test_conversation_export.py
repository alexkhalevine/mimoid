import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app import db, main


class ConversationExportTests(unittest.TestCase):
    """Exercises export_conversation() against a real (temp) SQLite file --
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

    def test_exports_user_and_assistant_messages_in_order(self):
        conversation = db.create_conversation("Test chat")
        db.add_message(conversation["id"], "user", "First message")
        db.add_message(conversation["id"], "assistant", "First reply")
        db.add_message(conversation["id"], "user", "Second message")

        result = main.export_conversation(conversation["id"])

        self.assertEqual(result["format"], main.CONVERSATION_EXPORT_FORMAT)
        self.assertEqual(result["version"], main.CONVERSATION_EXPORT_VERSION)
        self.assertIn("exported_at", result)
        self.assertEqual(result["conversation"]["id"], conversation["id"])
        self.assertEqual(result["conversation"]["title"], "Test chat")
        self.assertEqual(
            [(m["role"], m["content"]) for m in result["messages"]],
            [
                ("user", "First message"),
                ("assistant", "First reply"),
                ("user", "Second message"),
            ],
        )

    def test_message_rows_carry_content_and_timestamp(self):
        """The point of the export -- confirms the shape a consumer would
        actually read isn't missing a field, not just that something came
        back."""
        conversation = db.create_conversation()
        db.add_message(conversation["id"], "user", "Hello there")

        result = main.export_conversation(conversation["id"])

        message = result["messages"][0]
        self.assertEqual(message["role"], "user")
        self.assertEqual(message["content"], "Hello there")
        self.assertIn("created_at", message)

    def test_system_rows_are_excluded(self):
        """Matches what the Talk view actually renders -- chat.ts's history
        load skips role == "system" the same way, so an export shouldn't
        surface internal bookkeeping the owner never saw."""
        conversation = db.create_conversation()
        db.add_message(conversation["id"], "system", "internal bookkeeping")
        db.add_message(conversation["id"], "user", "a real message")

        result = main.export_conversation(conversation["id"])

        roles = [m["role"] for m in result["messages"]]
        self.assertNotIn("system", roles)
        self.assertEqual(roles, ["user"])

    def test_empty_conversation_exports_an_empty_message_list(self):
        conversation = db.create_conversation()

        result = main.export_conversation(conversation["id"])

        self.assertEqual(result["messages"], [])

    def test_archived_conversation_still_exports(self):
        """A conversation left behind by "New chat" is archived, not
        deleted -- it should still be reachable by id for export."""
        conversation = db.create_conversation()
        db.add_message(conversation["id"], "user", "before the reset")
        db.archive_conversation(conversation["id"])

        result = main.export_conversation(conversation["id"])

        self.assertEqual(len(result["messages"]), 1)

    def test_nonexistent_conversation_raises_404(self):
        with self.assertRaises(HTTPException) as caught:
            main.export_conversation("does-not-exist")
        self.assertEqual(caught.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
