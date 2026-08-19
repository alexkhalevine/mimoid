import tempfile
import unittest
from pathlib import Path

from app.config import migrate_legacy_data_dir


class MigrateLegacyDataDirTests(unittest.TestCase):
    """config.py's one-time move of a pre-fix install's data out of the
    (in a packaged app) read-only, disposable app-bundle location into the
    real writable data directory sidecar.rs now points MIMOID_DATA_DIR at.
    Getting this wrong either loses a real user's memories or silently
    leaves them stuck in the fragile spot -- so every branch is covered."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_moves_legacy_data_to_the_new_location(self):
        legacy = self.root / "old" / "data"
        legacy.mkdir(parents=True)
        (legacy / "mimoid.db").write_text("real user data")
        (legacy / "chroma").mkdir()
        (legacy / "chroma" / "index").write_text("embeddings")
        new = self.root / "new" / "data"

        migrated = migrate_legacy_data_dir(legacy, new)

        self.assertTrue(migrated)
        self.assertFalse(legacy.exists())
        self.assertEqual((new / "mimoid.db").read_text(), "real user data")
        self.assertEqual((new / "chroma" / "index").read_text(), "embeddings")

    def test_noop_when_the_paths_are_the_same(self):
        # The dev case: MIMOID_DATA_DIR unset, so DATA_DIR *is*
        # _LEGACY_DATA_DIR -- nothing to migrate, and this must never try to
        # "move" a directory onto itself.
        same = self.root / "data"
        same.mkdir()
        (same / "mimoid.db").write_text("dev data")

        migrated = migrate_legacy_data_dir(same, same)

        self.assertFalse(migrated)
        self.assertEqual((same / "mimoid.db").read_text(), "dev data")

    def test_noop_on_a_fresh_install_with_nothing_at_the_legacy_path(self):
        legacy = self.root / "old" / "data"  # never created
        new = self.root / "new" / "data"

        migrated = migrate_legacy_data_dir(legacy, new)

        self.assertFalse(migrated)
        self.assertFalse(new.exists())

    def test_noop_once_the_new_location_already_has_data(self):
        # Covers both "migration already ran" and "somehow both exist" --
        # never overwrite a real, already-populated new location with
        # whatever's left over at the old one.
        legacy = self.root / "old" / "data"
        legacy.mkdir(parents=True)
        (legacy / "mimoid.db").write_text("stale leftover")
        new = self.root / "new" / "data"
        new.mkdir(parents=True)
        (new / "mimoid.db").write_text("already migrated, real data")

        migrated = migrate_legacy_data_dir(legacy, new)

        self.assertFalse(migrated)
        self.assertEqual((new / "mimoid.db").read_text(), "already migrated, real data")
        # And the old copy is left alone too -- not silently deleted.
        self.assertTrue(legacy.exists())

    def test_creates_missing_parent_directories_for_the_new_location(self):
        legacy = self.root / "old" / "data"
        legacy.mkdir(parents=True)
        (legacy / "mimoid.db").write_text("data")
        # sidecar-venv already exists as a sibling under this parent in a
        # real install (created by sidecar.rs before the sidecar itself
        # ever starts), but the "data" subdirectory and its parent app-data
        # root might not, on a machine that's never run the app before.
        new = self.root / "Library" / "Application Support" / "com.example.app" / "data"

        migrated = migrate_legacy_data_dir(legacy, new)

        self.assertTrue(migrated)
        self.assertEqual((new / "mimoid.db").read_text(), "data")


if __name__ == "__main__":
    unittest.main()
