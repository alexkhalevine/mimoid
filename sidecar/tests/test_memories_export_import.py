import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from app import db, main


class _FakeUploadFile:
    def __init__(self, content: bytes):
        self.filename = "export.json"
        self._content = content

    async def read(self) -> bytes:
        return self._content


class ParseIsoDatetimeTests(unittest.TestCase):
    def test_accepts_a_real_iso_timestamp(self):
        self.assertEqual(
            main._parse_iso_datetime("2026-01-01T00:00:00+00:00"),
            "2026-01-01T00:00:00+00:00",
        )

    def test_rejects_non_string_and_malformed_values(self):
        self.assertIsNone(main._parse_iso_datetime(None))
        self.assertIsNone(main._parse_iso_datetime(12345))
        self.assertIsNone(main._parse_iso_datetime("not a date"))
        self.assertIsNone(main._parse_iso_datetime(""))


class MemoriesExportImportTests(unittest.IsolatedAsyncioTestCase):
    """Exercises export_memories/import_memories against a real (temp)
    SQLite file, the same pattern test_import_history.py uses -- this is
    thin enough SQL that mocking db would just re-assert the
    implementation. Chroma/Ollama indexing is mocked out (memory.index_memory)
    since it needs a live Ollama server neither this test nor CI has."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmp_dir.name) / "test.db"
        self._patcher = patch.object(db.config, "DB_PATH", self._db_path)
        self._patcher.start()
        db.init_db()
        self._index_patcher = patch.object(main.memory, "index_memory", new=AsyncMock(return_value=None))
        self._index_patcher.start()

    def tearDown(self):
        self._index_patcher.stop()
        self._patcher.stop()
        self._tmp_dir.cleanup()

    async def test_export_round_trips_through_import_into_an_empty_db(self):
        db.create_memory("First memory", topic="life", occurred_at="2020")
        db.create_memory("Second memory")

        exported = main.export_memories()
        self.assertEqual(exported["format"], main.MEMORIES_EXPORT_FORMAT)
        self.assertEqual(len(exported["memories"]), 2)
        # id/updated_at are this database's own bookkeeping -- an export
        # shouldn't carry them, since a re-import creates fresh ones.
        self.assertNotIn("id", exported["memories"][0])
        self.assertNotIn("updated_at", exported["memories"][0])

        # Wipe and reimport, proving the export is actually sufficient to
        # reconstruct the memories, not just a partial snapshot.
        with db.get_connection() as conn:
            conn.execute("DELETE FROM memories")
        upload = _FakeUploadFile(json.dumps(exported).encode())

        result = await main.import_memories(upload)

        self.assertEqual(result, {"created": 2, "skipped_duplicates": 0, "skipped_invalid": 0})
        contents = {m["content"] for m in db.list_memories()}
        self.assertEqual(contents, {"First memory", "Second memory"})

    async def test_import_preserves_original_created_at(self):
        upload = _FakeUploadFile(
            b'{"format": "mimoid-memories", "memories": '
            b'[{"content": "old memory", "created_at": "2020-01-01T00:00:00+00:00"}]}'
        )
        await main.import_memories(upload)

        record = db.list_memories()[0]
        self.assertEqual(record["created_at"], "2020-01-01T00:00:00+00:00")
        self.assertEqual(record["source"], "import")

    async def test_import_falls_back_to_now_for_a_bad_created_at(self):
        upload = _FakeUploadFile(
            b'{"memories": [{"content": "no real date", "created_at": "not-a-date"}]}'
        )
        await main.import_memories(upload)

        record = db.list_memories()[0]
        self.assertNotEqual(record["created_at"], "not-a-date")

    async def test_import_skips_duplicates_against_existing_content(self):
        db.create_memory("Already here")
        upload = _FakeUploadFile(
            b'{"memories": [{"content": "Already here"}, {"content": "  ALREADY here  "}, {"content": "New one"}]}'
        )

        result = await main.import_memories(upload)

        # Both the exact repeat and the whitespace/case variant count as the
        # same memory; only the genuinely new one is created.
        self.assertEqual(result, {"created": 1, "skipped_duplicates": 2, "skipped_invalid": 0})
        self.assertEqual(len(db.list_memories()), 2)

    async def test_import_skips_duplicates_within_the_same_file(self):
        upload = _FakeUploadFile(b'{"memories": [{"content": "same"}, {"content": "same"}]}')

        result = await main.import_memories(upload)

        self.assertEqual(result, {"created": 1, "skipped_duplicates": 1, "skipped_invalid": 0})

    async def test_import_skips_invalid_entries_without_failing_the_batch(self):
        upload = _FakeUploadFile(
            b'{"memories": [{"content": "good"}, {"content": ""}, {"content": 123}, "not an object", {}]}'
        )

        result = await main.import_memories(upload)

        self.assertEqual(result, {"created": 1, "skipped_duplicates": 0, "skipped_invalid": 4})

    async def test_import_rejects_a_file_with_no_memories_list(self):
        upload = _FakeUploadFile(b'{"not_memories": []}')
        with self.assertRaises(main.HTTPException) as ctx:
            await main.import_memories(upload)
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_import_rejects_malformed_json(self):
        upload = _FakeUploadFile(b"not json at all")
        with self.assertRaises(main.HTTPException) as ctx:
            await main.import_memories(upload)
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_import_rejects_a_mismatched_export_format_tag(self):
        upload = _FakeUploadFile(b'{"format": "some-other-app-export", "memories": [{"content": "x"}]}')
        with self.assertRaises(main.HTTPException) as ctx:
            await main.import_memories(upload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(len(db.list_memories()), 0)

    async def test_import_accepts_a_file_with_no_format_tag_at_all(self):
        # Permissive on the way in: a hand-written or foreign-but-compatible
        # file with no "format" key at all is still accepted, since the
        # only thing actually required is a "memories" list.
        upload = _FakeUploadFile(b'{"memories": [{"content": "x"}]}')
        result = await main.import_memories(upload)
        self.assertEqual(result["created"], 1)

    async def test_indexing_failure_does_not_abort_the_import(self):
        main.memory.index_memory.side_effect = httpx.ConnectError("no ollama")
        upload = _FakeUploadFile(b'{"memories": [{"content": "still saved"}]}')

        result = await main.import_memories(upload)

        self.assertEqual(result["created"], 1)
        self.assertEqual(db.list_memories()[0]["content"], "still saved")


class ListMemoryContentsTests(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmp_dir.name) / "test.db"
        self._patcher = patch.object(db.config, "DB_PATH", self._db_path)
        self._patcher.start()
        db.init_db()

    def tearDown(self):
        self._patcher.stop()
        self._tmp_dir.cleanup()

    def test_normalizes_case_and_whitespace(self):
        db.create_memory("  Some Memory  ")
        self.assertEqual(db.list_memory_contents(), {"some memory"})

    def test_empty_before_any_memories_exist(self):
        self.assertEqual(db.list_memory_contents(), set())


if __name__ == "__main__":
    unittest.main()
