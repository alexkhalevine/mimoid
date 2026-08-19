import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import db, main


class ImportBatchDbTests(unittest.TestCase):
    """Exercises the import_batches table + helpers against a real (temp)
    SQLite file -- these are thin enough SQL that mocking db would just
    re-assert the implementation."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmp_dir.name) / "test.db"
        self._patcher = patch.object(db.config, "DB_PATH", self._db_path)
        self._patcher.start()
        db.init_db()

    def tearDown(self):
        self._patcher.stop()
        self._tmp_dir.cleanup()

    def test_migration_is_idempotent(self):
        # init_db() already ran once in setUp; running it again must not
        # raise even though the entry_count column already exists.
        db.init_db()
        db.init_db()
        guide = db.set_style_guide("hello", entry_count=5)
        self.assertEqual(guide["entry_count"], 5)

    def test_list_batches_orders_newest_first(self):
        with db.get_connection() as conn:
            conn.execute(
                "INSERT INTO import_batches (id, source, filename, entry_count, created_at) "
                "VALUES ('a', 'whatsapp', 'old.txt', 3, '2026-01-01T00:00:00+00:00')"
            )
            conn.execute(
                "INSERT INTO import_batches (id, source, filename, entry_count, created_at) "
                "VALUES ('b', 'signal', 'new.txt', 7, '2026-02-01T00:00:00+00:00')"
            )

        batches = db.list_import_batches()
        self.assertEqual([b["id"] for b in batches], ["b", "a"])

    def test_create_import_batch_round_trips(self):
        record = db.create_import_batch("whatsapp", "chat.txt", 12)
        self.assertEqual(record["source"], "whatsapp")
        self.assertEqual(record["filename"], "chat.txt")
        self.assertEqual(record["entry_count"], 12)

        batches = db.list_import_batches()
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["id"], record["id"])

    def test_delete_by_source_only_removes_matching_rows(self):
        db.create_import_batch("whatsapp", "a.txt", 1)
        db.create_import_batch("signal", "b.txt", 2)

        deleted = db.delete_import_batches_by_source("whatsapp")

        self.assertEqual(deleted, 1)
        remaining = db.list_import_batches()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["source"], "signal")

    def test_import_batches_source_check_constraint(self):
        with self.assertRaises(sqlite3.IntegrityError), db.get_connection() as conn:
            conn.execute(
                "INSERT INTO import_batches (id, source, filename, entry_count, created_at) "
                "VALUES ('x', 'bogus', 'f.txt', 1, '2026-01-01T00:00:00+00:00')"
            )


class _FakeUploadFile:
    def __init__(self, filename: str, content: bytes = b"content"):
        self.filename = filename
        self._content = content

    async def read(self) -> bytes:
        return self._content


class ImportBatchRecordingTests(unittest.IsolatedAsyncioTestCase):
    """Covers main.py's wiring: the filename recorded in a batch must come
    from parse time (the upload cache), not the import request body, and a
    zero-entry import must not create a history row that describes nothing."""

    async def test_whatsapp_import_records_batch_with_filename_from_parse_time(self):
        upload_id = "upload-1"
        main._whatsapp_uploads[upload_id] = {
            "filename": "family-chat.txt",
            "messages": [{"sender": "Alex", "text": "hi"}],
        }
        entries = [{"kind": "text", "content": "hi", "prompt": None}]
        try:
            with (
                patch.object(main.whatsapp, "participants", return_value=[{"name": "Alex", "message_count": 1}]),
                patch.object(main.whatsapp, "select_entries", return_value=entries),
                patch.object(main.db, "list_style_entry_contents_by_source", return_value=set()),
                patch.object(main.db, "create_style_entry", return_value={"id": "e1", "kind": "text", "content": "hi", "prompt": None}),
                patch.object(main, "_index_style_entry_best_effort", return_value=None),
                patch.object(main.db, "create_import_batch") as create_batch,
            ):
                await main.import_whatsapp(main.WhatsappImportRequest(upload_id=upload_id, me="Alex"))
        finally:
            main._whatsapp_uploads.pop(upload_id, None)

        create_batch.assert_called_once_with("whatsapp", "family-chat.txt", 1)

    async def test_whatsapp_zero_entry_import_records_nothing(self):
        upload_id = "upload-2"
        main._whatsapp_uploads[upload_id] = {
            "filename": "empty-chat.txt",
            "messages": [{"sender": "Alex", "text": "hi"}],
        }
        try:
            with (
                patch.object(main.whatsapp, "participants", return_value=[{"name": "Alex", "message_count": 1}]),
                patch.object(main.whatsapp, "select_entries", return_value=[]),
                patch.object(main.db, "list_style_entry_contents_by_source", return_value=set()),
                patch.object(main.db, "create_import_batch") as create_batch,
            ):
                await main.import_whatsapp(main.WhatsappImportRequest(upload_id=upload_id, me="Alex"))
        finally:
            main._whatsapp_uploads.pop(upload_id, None)

        create_batch.assert_not_called()

    async def test_signal_import_names_batch_from_first_file_plus_n_more(self):
        files = [_FakeUploadFile("a.txt"), _FakeUploadFile("b.txt"), _FakeUploadFile("c.txt")]
        entries = [{"kind": "text", "content": "hi", "prompt": None}]
        with (
            patch.object(main.signal, "parse_export", return_value=[{"sender": "Alex", "text": "hi"}]),
            patch.object(main.signal, "conversation_name", return_value="A chat"),
            patch.object(main.signal, "select_entries_for_files", return_value=entries),
            patch.object(main.db, "list_style_entry_contents_by_source", return_value=set()),
            patch.object(main.db, "create_style_entry", return_value={"id": "e1", "kind": "text", "content": "hi", "prompt": None}),
            patch.object(main, "_index_style_entry_best_effort", return_value=None),
            patch.object(main.db, "create_import_batch") as create_batch,
        ):
            await main.import_signal(files)

        create_batch.assert_called_once_with("signal", "a.txt +2 more", 1)

    async def test_signal_zero_entry_import_records_nothing(self):
        files = [_FakeUploadFile("a.txt")]
        with (
            patch.object(main.signal, "parse_export", return_value=[{"sender": "Alex", "text": "hi"}]),
            patch.object(main.signal, "conversation_name", return_value="A chat"),
            patch.object(main.signal, "select_entries_for_files", return_value=[]),
            patch.object(main.db, "list_style_entry_contents_by_source", return_value=set()),
            patch.object(main.db, "create_import_batch") as create_batch,
        ):
            await main.import_signal(files)

        create_batch.assert_not_called()


class ImportHistoryEndpointTests(unittest.TestCase):
    def test_get_import_history_wraps_db_list(self):
        batches = [{"id": "1", "source": "whatsapp", "filename": "a.txt", "entry_count": 3, "created_at": "now"}]
        with patch.object(main.db, "list_import_batches", return_value=batches):
            result = main.get_import_history()
        self.assertEqual(result, {"batches": batches})


class DeleteStyleEntriesTests(unittest.TestCase):
    def test_delete_by_source_also_clears_import_history(self):
        with (
            patch.object(main.db, "list_style_entries", return_value=[]),
            patch.object(main.db, "delete_style_entries_by_source", return_value=0),
            patch.object(main.db, "delete_import_batches_by_source") as delete_batches,
        ):
            main.delete_style_entries("whatsapp")
        delete_batches.assert_called_once_with("whatsapp")


if __name__ == "__main__":
    unittest.main()
