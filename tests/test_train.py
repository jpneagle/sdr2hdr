from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.train import resolve_training_device


class TrainScriptTests(unittest.TestCase):
    def test_resolve_training_device_prefers_cuda(self) -> None:
        with patch("scripts.train.torch.cuda.is_available", return_value=True):
            device = resolve_training_device("auto")
        self.assertEqual(device.type, "cuda")

    def test_resolve_training_device_uses_mps_when_cuda_is_unavailable(self) -> None:
        with patch("scripts.train.torch.cuda.is_available", return_value=False):
            with patch("scripts.train.torch.backends.mps.is_available", return_value=True):
                device = resolve_training_device("auto")
        self.assertEqual(device.type, "mps")

    def test_resolve_training_device_falls_back_to_cpu(self) -> None:
        with patch("scripts.train.torch.cuda.is_available", return_value=False):
            with patch("scripts.train.torch.backends.mps.is_available", return_value=False):
                device = resolve_training_device("auto")
        self.assertEqual(device.type, "cpu")

    def test_resolve_training_device_respects_explicit_request(self) -> None:
        device = resolve_training_device("cpu")
        self.assertEqual(device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
