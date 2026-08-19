import unittest
from unittest.mock import MagicMock, patch

from app import tts


def _fake_torch(cuda=False, mps=False, explode=False):
    torch = MagicMock()
    if explode:
        torch.cuda.is_available.side_effect = RuntimeError("driver not loaded")
    else:
        torch.cuda.is_available.return_value = cuda
    torch.backends.mps.is_available.return_value = mps
    return torch


class SynthesisDeviceTests(unittest.TestCase):
    """XTTS runs on whatever torch device it's moved to, and the cost of
    getting this wrong is silent -- synthesis still works, just many times
    slower on CPU while a perfectly good GPU idles."""

    def _device_with(self, torch):
        with patch.dict("sys.modules", {"torch": torch}):
            return tts._synthesis_device()

    def test_prefers_cuda(self):
        self.assertEqual(self._device_with(_fake_torch(cuda=True, mps=False)), "cuda")

    def test_uses_mps_without_cuda(self):
        self.assertEqual(self._device_with(_fake_torch(cuda=False, mps=True)), "mps")

    def test_falls_back_to_cpu(self):
        self.assertEqual(self._device_with(_fake_torch(cuda=False, mps=False)), "cpu")

    def test_cuda_wins_when_both_somehow_report_available(self):
        self.assertEqual(self._device_with(_fake_torch(cuda=True, mps=True)), "cuda")

    def test_broken_gpu_probe_falls_back_to_cpu(self):
        # The regression guard: an NVIDIA box whose driver isn't loaded must
        # degrade to CPU rather than taking down sidecar startup.
        self.assertEqual(self._device_with(_fake_torch(explode=True)), "cpu")


if __name__ == "__main__":
    unittest.main()
