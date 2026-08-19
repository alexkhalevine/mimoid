import unittest
from unittest.mock import patch

from app import main


class SettingsEndpointTests(unittest.TestCase):
    """/settings carries the language, the voice pause length, and the
    owner's name. Each field must be updatable on its own, because the
    language selector, the Talk tab's pause slider, and the name field (set
    once at first-run onboarding, editable later in System) PUT
    independently -- a required-field model would make whichever saved last
    wipe the others."""

    def setUp(self):
        self.store: dict[str, str] = {}
        get = patch.object(main.db, "get_setting", side_effect=lambda k, d: self.store.get(k, d))
        put = patch.object(main.db, "set_setting", side_effect=self.store.__setitem__)
        get.start()
        put.start()
        self.addCleanup(get.stop)
        self.addCleanup(put.stop)

    def test_defaults_when_nothing_stored(self):
        self.assertEqual(
            main.get_settings(),
            {
                "language": main.config.DEFAULT_LANGUAGE,
                "pause_ms": main.DEFAULT_PAUSE_MS,
                "owner_name": main.config.DEFAULT_OWNER_NAME,
            },
        )

    def test_pause_only_update_leaves_others_alone(self):
        self.store["language"] = "de"
        self.store["owner_name"] = "Jordan"
        result = main.update_settings(main.SettingsRequest(pause_ms=3500))
        self.assertEqual(result, {"language": "de", "pause_ms": 3500, "owner_name": "Jordan"})
        self.assertEqual(self.store["pause_ms"], "3500")

    def test_language_only_update_leaves_others_alone(self):
        self.store["pause_ms"] = "3000"
        self.store["owner_name"] = "Jordan"
        result = main.update_settings(main.SettingsRequest(language="nl"))
        self.assertEqual(result, {"language": "nl", "pause_ms": 3000, "owner_name": "Jordan"})

    def test_owner_name_only_update_leaves_others_alone(self):
        self.store["language"] = "de"
        self.store["pause_ms"] = "3000"
        result = main.update_settings(main.SettingsRequest(owner_name="Jordan"))
        self.assertEqual(result, {"language": "de", "pause_ms": 3000, "owner_name": "Jordan"})
        self.assertEqual(self.store["owner_name"], "Jordan")

    def test_owner_name_is_trimmed(self):
        main.update_settings(main.SettingsRequest(owner_name="  Jordan  "))
        self.assertEqual(self.store["owner_name"], "Jordan")

    def test_out_of_range_pause_is_rejected(self):
        for bad in (main.MIN_PAUSE_MS - 1, main.MAX_PAUSE_MS + 1):
            with self.subTest(pause_ms=bad):
                with self.assertRaises(main.HTTPException) as caught:
                    main.update_settings(main.SettingsRequest(pause_ms=bad))
                self.assertEqual(caught.exception.status_code, 400)
                self.assertNotIn("pause_ms", self.store)

    def test_unsupported_language_is_rejected(self):
        with self.assertRaises(main.HTTPException) as caught:
            main.update_settings(main.SettingsRequest(language="klingon"))
        self.assertEqual(caught.exception.status_code, 400)

    def test_blank_owner_name_is_rejected(self):
        for bad in ("", "   "):
            with self.subTest(owner_name=repr(bad)):
                with self.assertRaises(main.HTTPException) as caught:
                    main.update_settings(main.SettingsRequest(owner_name=bad))
                self.assertEqual(caught.exception.status_code, 400)
                self.assertNotIn("owner_name", self.store)

    def test_overlong_owner_name_is_rejected(self):
        with self.assertRaises(main.HTTPException) as caught:
            main.update_settings(main.SettingsRequest(owner_name="x" * (main.MAX_OWNER_NAME_LENGTH + 1)))
        self.assertEqual(caught.exception.status_code, 400)
        self.assertNotIn("owner_name", self.store)

    def test_owner_name_at_max_length_is_accepted(self):
        name = "x" * main.MAX_OWNER_NAME_LENGTH
        main.update_settings(main.SettingsRequest(owner_name=name))
        self.assertEqual(self.store["owner_name"], name)

    def test_corrupt_stored_pause_falls_back_to_default(self):
        self.store["pause_ms"] = "not-a-number"
        self.assertEqual(main.get_settings()["pause_ms"], main.DEFAULT_PAUSE_MS)


if __name__ == "__main__":
    unittest.main()
