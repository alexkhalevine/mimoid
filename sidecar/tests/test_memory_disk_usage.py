import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import memory


class DiskUsageBytesTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.chroma_dir = self.tmp_dir / "chroma"
        patcher = patch.object(memory.config, "CHROMA_DIR", self.chroma_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_zero_when_directory_does_not_exist_yet(self):
        self.assertEqual(memory.disk_usage_bytes(), 0)

    def test_sums_files_across_nested_subdirectories(self):
        self.chroma_dir.mkdir(parents=True)
        (self.chroma_dir / "chroma.sqlite3").write_bytes(b"a" * 100)
        segment_dir = self.chroma_dir / "segment-uuid"
        segment_dir.mkdir()
        (segment_dir / "data_level0.bin").write_bytes(b"b" * 250)

        self.assertEqual(memory.disk_usage_bytes(), 350)


if __name__ == "__main__":
    unittest.main()
