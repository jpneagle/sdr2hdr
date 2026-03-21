import unittest

import numpy as np

from sdr2hdr.ai import EnhancementMaps, HeuristicEnhancer, estimate_heuristic_maps


class HeuristicMapsTests(unittest.TestCase):
    def test_output_shapes_match_input(self) -> None:
        frame = np.random.default_rng(42).random((64, 64, 3)).astype(np.float32)
        maps = estimate_heuristic_maps(frame)
        self.assertEqual(maps.expansion.shape, (64, 64))
        self.assertEqual(maps.contrast.shape, (64, 64))
        self.assertEqual(maps.protection.shape, (64, 64))

    def test_output_bounded_zero_one(self) -> None:
        frame = np.random.default_rng(42).random((32, 32, 3)).astype(np.float32)
        maps = estimate_heuristic_maps(frame)
        self.assertGreaterEqual(float(maps.expansion.min()), 0.0)
        self.assertLessEqual(float(maps.expansion.max()), 1.0)
        self.assertGreaterEqual(float(maps.contrast.min()), 0.0)
        self.assertLessEqual(float(maps.contrast.max()), 1.0)
        self.assertGreaterEqual(float(maps.protection.min()), 0.0)
        self.assertLessEqual(float(maps.protection.max()), 1.0)

    def test_dark_frame_low_expansion(self) -> None:
        frame = np.full((32, 32, 3), 0.1, dtype=np.float32)
        maps = estimate_heuristic_maps(frame)
        self.assertAlmostEqual(float(maps.expansion.mean()), 0.0, places=2)

    def test_bright_frame_high_expansion(self) -> None:
        frame = np.full((32, 32, 3), 0.9, dtype=np.float32)
        maps = estimate_heuristic_maps(frame)
        self.assertGreater(float(maps.expansion.mean()), 0.5)

    def test_neutral_frame_high_protection(self) -> None:
        frame = np.full((32, 32, 3), 0.5, dtype=np.float32)
        maps = estimate_heuristic_maps(frame)
        self.assertAlmostEqual(float(maps.protection.mean()), 1.0, places=2)

    def test_saturated_frame_low_protection(self) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.float32)
        frame[..., 0] = 0.8
        maps = estimate_heuristic_maps(frame)
        self.assertLess(float(maps.protection.mean()), 0.2)


class HeuristicEnhancerTests(unittest.TestCase):
    def test_returns_enhancement_maps(self) -> None:
        enhancer = HeuristicEnhancer()
        frame = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
        result = enhancer.estimate(frame)
        self.assertIsInstance(result, EnhancementMaps)

    def test_estimate_matches_standalone(self) -> None:
        enhancer = HeuristicEnhancer()
        frame = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
        maps_a = enhancer.estimate(frame)
        maps_b = estimate_heuristic_maps(frame)
        np.testing.assert_array_equal(maps_a.expansion, maps_b.expansion)
        np.testing.assert_array_equal(maps_a.contrast, maps_b.contrast)
        np.testing.assert_array_equal(maps_a.protection, maps_b.protection)


if __name__ == "__main__":
    unittest.main()
