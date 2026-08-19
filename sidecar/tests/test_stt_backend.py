import unittest
from unittest.mock import patch

from app import stt


def _spec_finder(*present: str):
    """Stands in for importlib.util.find_spec, reporting only `present` as
    installed. The real find_spec answers from this interpreter's own site-
    packages, which would make these tests depend on whichever backend the
    dev machine happens to have."""

    def find_spec(name):
        return object() if name in present else None

    return find_spec


class BackendSelectionTests(unittest.TestCase):
    """_backend() is the single decision every other function in stt.py
    routes through, and getting it wrong is what made is_available() claim
    speech-to-text worked on Linux while /transcribe 503'd."""

    def _backend_for(self, sys_platform, machine, *installed):
        with (
            patch.object(stt.sys, "platform", sys_platform),
            patch.object(stt.platform, "machine", return_value=machine),
            patch("importlib.util.find_spec", _spec_finder(*installed)),
        ):
            return stt._backend()

    def test_apple_silicon_prefers_mlx(self):
        self.assertEqual(
            self._backend_for("darwin", "arm64", "mlx_whisper", "whisper"),
            stt._MLX,
        )

    def test_apple_silicon_falls_back_to_torch_without_mlx(self):
        self.assertEqual(self._backend_for("darwin", "arm64", "whisper"), stt._TORCH)

    def test_linux_ignores_mlx_even_when_installed(self):
        # The regression this guards: pip will happily install mlx-whisper on
        # Linux (it's only the native libmlx.so that's missing), so a venv
        # created before requirements.txt grew its platform markers still has
        # the module present and importable-looking.
        self.assertEqual(
            self._backend_for("linux", "x86_64", "mlx_whisper", "whisper"),
            stt._TORCH,
        )

    def test_intel_mac_uses_torch(self):
        self.assertEqual(
            self._backend_for("darwin", "x86_64", "mlx_whisper", "whisper"),
            stt._TORCH,
        )

    def test_no_backend_installed(self):
        self.assertIsNone(self._backend_for("linux", "x86_64"))


class AvailabilityTests(unittest.TestCase):
    def test_is_available_tracks_backend(self):
        with patch.object(stt, "_backend", return_value=stt._TORCH):
            self.assertTrue(stt.is_available())
        with patch.object(stt, "_backend", return_value=None):
            self.assertFalse(stt.is_available())

    def test_active_model_matches_backend(self):
        with patch.object(stt, "_backend", return_value=stt._MLX):
            self.assertEqual(stt.active_model(), stt.config.WHISPER_MODEL)
        with patch.object(stt, "_backend", return_value=stt._TORCH):
            self.assertEqual(stt.active_model(), stt.config.WHISPER_TORCH_MODEL)

    def test_is_model_downloaded_is_false_without_a_backend(self):
        with patch.object(stt, "_backend", return_value=None):
            self.assertFalse(stt.is_model_downloaded())

    def test_transcribe_without_a_backend_raises_unavailable(self):
        # /transcribe turns this into a clean 503 rather than a 500.
        with patch.object(stt, "_backend", return_value=None), self.assertRaises(stt.SpeechToTextUnavailable):
            stt.transcribe("/tmp/nope.wav")


class TorchModelCacheTests(unittest.TestCase):
    def test_reports_cached_checkpoint(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "whisper").mkdir()
            checkpoint = Path(tmp) / "whisper" / "large-v3-turbo.pt"
            with patch.dict(
                "os.environ", {"XDG_CACHE_HOME": tmp}
            ), patch.dict(
                "sys.modules",
                {"whisper": type("m", (), {"_MODELS": {"turbo": "https://x/large-v3-turbo.pt"}})},
            ):
                self.assertFalse(stt._torch_model_downloaded("turbo"))
                checkpoint.write_bytes(b"weights")
                self.assertTrue(stt._torch_model_downloaded("turbo"))

    def test_loaded_model_is_reused_and_reloaded_on_name_change(self):
        loads = []

        def load_model(name, device=None):
            loads.append(name)
            return f"model:{name}"

        fake_whisper = type("m", (), {"load_model": staticmethod(load_model), "_MODELS": {}})
        fake_torch = type("t", (), {"cuda": type("c", (), {"is_available": staticmethod(lambda: False)})})

        with patch.dict("sys.modules", {"whisper": fake_whisper, "torch": fake_torch}), patch.object(
            stt, "_torch_model", None
        ), patch.object(stt, "_torch_model_name", None):
            self.assertEqual(stt._get_torch_model("turbo"), "model:turbo")
            # Second call for the same name must hit the cache, not reload
            # the (multi-hundred-MB) checkpoint.
            self.assertEqual(stt._get_torch_model("turbo"), "model:turbo")
            self.assertEqual(loads, ["turbo"])

            # A different name must actually reload rather than silently
            # serving the previous checkpoint.
            self.assertEqual(stt._get_torch_model("small"), "model:small")
            self.assertEqual(loads, ["turbo", "small"])

    def test_unknown_model_name_is_not_cached(self):
        with patch.dict(
            "sys.modules",
            {"whisper": type("m", (), {"_MODELS": {"turbo": "https://x/large-v3-turbo.pt"}})},
        ):
            self.assertFalse(stt._torch_model_downloaded("no-such-model"))


if __name__ == "__main__":
    unittest.main()
