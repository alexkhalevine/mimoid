import unittest
from unittest.mock import patch

from app import main


class StorageEndpointTests(unittest.TestCase):
    """/storage powers the header's System details popover -- it should just
    report whatever memory.disk_usage_bytes() says, with no logic of its
    own to duplicate/drift from that function's tests."""

    def test_reports_chroma_disk_usage(self):
        with patch.object(main.memory, "disk_usage_bytes", return_value=4096):
            self.assertEqual(main.get_storage(), {"chroma_bytes": 4096})

    def test_zero_before_anything_is_indexed(self):
        with patch.object(main.memory, "disk_usage_bytes", return_value=0):
            self.assertEqual(main.get_storage(), {"chroma_bytes": 0})


if __name__ == "__main__":
    unittest.main()
