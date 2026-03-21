import unittest

import numpy as np

from sdr2hdr.masks import (
    apply_near_white_rolloff,
    build_ai_gate,
    compute_adaptive_highlight_boost,
    compute_chroma,
    compute_luma,
    estimate_clipped_white_mask,
    estimate_high_chroma_mask,
    estimate_memory_color_mask,
    estimate_noise_mask,
    estimate_skin_mask,
    estimate_sky_mask,
    estimate_specular_mask,
    estimate_subtitle_mask_fast,
    limit_ai_highlight_expansion,
)
from sdr2hdr.constants import LUMA_R, LUMA_G, LUMA_B


class ConstantsTests(unittest.TestCase):
    def test_luma_coefficients_sum_to_one(self) -> None:
        self.assertAlmostEqual(LUMA_R + LUMA_G + LUMA_B, 1.0, places=4)

    def test_compute_luma_uses_constants(self) -> None:
        frame = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        self.assertAlmostEqual(compute_luma(frame).item(), LUMA_R, places=4)

    def test_compute_chroma_neutral(self) -> None:
        frame = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        self.assertAlmostEqual(compute_chroma(frame).item(), 0.0, places=6)

    def test_compute_chroma_saturated(self) -> None:
        frame = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        self.assertAlmostEqual(compute_chroma(frame).item(), 1.0, places=6)


class MaskImportTests(unittest.TestCase):
    """Verify masks are accessible from both masks and core modules."""

    def test_import_from_core_backward_compat(self) -> None:
        from sdr2hdr.core import estimate_skin_mask as skin_from_core
        from sdr2hdr.masks import estimate_skin_mask as skin_from_masks

        self.assertIs(skin_from_core, skin_from_masks)

    def test_import_build_ai_gate_from_core(self) -> None:
        from sdr2hdr.core import build_ai_gate as gate_from_core
        from sdr2hdr.masks import build_ai_gate as gate_from_masks

        self.assertIs(gate_from_core, gate_from_masks)


class NearWhiteRolloffTests(unittest.TestCase):
    def test_zero_strength_returns_ones(self) -> None:
        luma = np.array([0.5, 0.8, 0.95], dtype=np.float32)
        result = apply_near_white_rolloff(luma, 0.78, 0.0)
        np.testing.assert_array_equal(result, np.ones_like(luma))

    def test_rolloff_increases_with_luma(self) -> None:
        luma = np.array([0.7, 0.85, 0.95], dtype=np.float32)
        result = apply_near_white_rolloff(luma, 0.78, 0.6)
        self.assertGreater(result[0], result[1])
        self.assertGreater(result[1], result[2])


class AdaptiveHighlightTests(unittest.TestCase):
    def test_clipped_white_reduces_boost(self) -> None:
        base = compute_adaptive_highlight_boost(1.2, 0.0, 0.0, 0.0, 0.0, 0.5, 1.5)
        with_clip = compute_adaptive_highlight_boost(1.2, 0.5, 0.0, 0.0, 0.0, 0.5, 1.5)
        self.assertLess(with_clip, base)

    def test_specular_increases_boost(self) -> None:
        base = compute_adaptive_highlight_boost(1.2, 0.0, 0.0, 0.0, 0.0, 0.5, 1.5)
        with_spec = compute_adaptive_highlight_boost(1.2, 0.0, 0.0, 0.5, 0.0, 0.5, 1.5)
        self.assertGreater(with_spec, base)


if __name__ == "__main__":
    unittest.main()
