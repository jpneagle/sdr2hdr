import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from sdr2hdr.ai import TorchMapEnhancer
from sdr2hdr.dataset import HDRSDRPairDataset, derive_target_maps
from sdr2hdr.model import EnhancementUNet


@unittest.skipIf(torch is None, "torch not installed")
class ModelDatasetTests(unittest.TestCase):
    def test_model_returns_expected_shape(self) -> None:
        model = EnhancementUNet()
        output = model(torch.rand(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 3, 64, 64))
        self.assertGreaterEqual(float(output.detach().min()), -1.0)
        self.assertLessEqual(float(output.detach().max()), 1.0)

    def test_model_handles_odd_input_sizes(self) -> None:
        model = EnhancementUNet()
        output = model(torch.rand(1, 3, 611, 917))
        self.assertEqual(tuple(output.shape), (1, 3, 611, 917))

    def test_derive_target_maps_returns_bounded_maps(self) -> None:
        sdr = np.full((32, 32, 3), 0.25, dtype=np.float32)
        hdr = np.full((32, 32, 3), 0.45, dtype=np.float32)
        targets = derive_target_maps(sdr, hdr)
        self.assertEqual(targets.expansion.shape, (32, 32))
        self.assertEqual(targets.contrast.shape, (32, 32))
        self.assertEqual(targets.protection.shape, (32, 32))
        self.assertTrue(np.all(targets.expansion >= -1.0))
        self.assertTrue(np.all(targets.expansion <= 1.0))
        self.assertTrue(np.all(targets.protection >= -1.0))
        self.assertTrue(np.all(targets.protection <= 1.0))

    def test_derive_target_maps_reduces_expansion_for_vivid_memory_colors(self) -> None:
        sdr = np.full((16, 16, 3), [0.18, 0.55, 0.12], dtype=np.float32)
        hdr = np.full((16, 16, 3), [0.35, 0.75, 0.22], dtype=np.float32)
        targets = derive_target_maps(sdr, hdr)
        self.assertGreater(float(targets.protection.mean()), 0.0)
        self.assertLess(float(targets.expansion.mean()), 0.25)

    def test_dataset_loads_npz_and_returns_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path = Path(temp_dir) / "sample.npz"
            np.savez_compressed(
                sample_path,
                sdr_linear=np.full((48, 48, 3), 0.2, dtype=np.float32),
                hdr_linear=np.full((48, 48, 3), 0.4, dtype=np.float32),
            )
            dataset = HDRSDRPairDataset(temp_dir, patch_size=32, training=False)
            sample = dataset[0]
            self.assertEqual(tuple(sample["sdr_linear"].shape), (3, 32, 32))
            self.assertEqual(tuple(sample["target_maps"].shape), (3, 32, 32))
            self.assertEqual(tuple(sample["clip_mask"].shape), (1, 32, 32))

    def test_validation_dataset_center_crops_large_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path = Path(temp_dir) / "sample.npz"
            np.savez_compressed(
                sample_path,
                sdr_linear=np.full((96, 128, 3), 0.2, dtype=np.float32),
                hdr_linear=np.full((96, 128, 3), 0.4, dtype=np.float32),
            )
            dataset = HDRSDRPairDataset(temp_dir, patch_size=32, training=False)
            sample = dataset[0]
            self.assertEqual(tuple(sample["sdr_linear"].shape), (3, 32, 32))
            self.assertEqual(tuple(sample["target_maps"].shape), (3, 32, 32))

    def test_torch_map_enhancer_loads_scripted_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "enhancer.pt"
            model = EnhancementUNet().eval()
            scripted = torch.jit.script(model)
            torch.jit.save(scripted, str(model_path))
            enhancer = TorchMapEnhancer(str(model_path), device="cpu")
            frame = np.full((16, 16, 3), 0.25, dtype=np.float32)
            maps = enhancer.estimate(frame)
            self.assertEqual(maps.expansion.shape, (16, 16))
            self.assertEqual(maps.contrast.shape, (16, 16))
            self.assertEqual(maps.protection.shape, (16, 16))

    def test_torch_map_enhancer_preserves_output_shape_with_low_res_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "enhancer.pt"
            model = EnhancementUNet().eval()
            scripted = torch.jit.script(model)
            torch.jit.save(scripted, str(model_path))
            enhancer = TorchMapEnhancer(str(model_path), device="cpu", inference_scale=0.5)
            frame = np.full((73, 121, 3), 0.25, dtype=np.float32)
            maps = enhancer.estimate(frame)
            self.assertEqual(maps.expansion.shape, (73, 121))
            self.assertTrue(np.all(maps.expansion >= 0.0))
            self.assertTrue(np.all(maps.expansion <= 1.0))


if __name__ == "__main__":
    unittest.main()
