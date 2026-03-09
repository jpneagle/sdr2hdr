import unittest
from unittest import mock

import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from sdr2hdr.core import (
    ProcessorConfig,
    SDRToHDRProcessor,
    TemporalState,
    apply_near_white_rolloff,
    build_ai_gate,
    compute_adaptive_highlight_boost,
    estimate_clipped_white_mask,
    estimate_high_chroma_mask,
    estimate_memory_color_mask,
    estimate_noise_mask,
    estimate_skin_mask,
    estimate_specular_mask,
    estimate_subtitle_mask,
    limit_ai_highlight_expansion,
    linear_to_pq,
    srgb_to_linear,
)
from sdr2hdr.review import default_sample_times, parse_times, pq_to_relative_linear, tone_map_hdr_preview


class CoreTests(unittest.TestCase):
    def test_parse_times(self) -> None:
        self.assertEqual(parse_times("0.5, 1.25,2"), [0.5, 1.25, 2.0])

    def test_default_sample_times_with_duration(self) -> None:
        out = default_sample_times(10.0, 4)
        self.assertEqual(out, [1.0, 3.667, 6.333, 9.0])

    def test_default_sample_times_without_duration(self) -> None:
        out = default_sample_times(None, 3)
        self.assertEqual(out, [0.0, 1.0, 2.0])

    def test_tone_map_hdr_preview_returns_uint8_image(self) -> None:
        frame = np.full((4, 4, 3), 0.75, dtype=np.float32)
        out = tone_map_hdr_preview(frame)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, frame.shape)

    def test_pq_to_relative_linear_is_monotonic(self) -> None:
        values = np.array([0.1, 0.2, 0.4, 0.8], dtype=np.float32)
        out = pq_to_relative_linear(values, peak_nits=1000.0)
        self.assertTrue(np.all(np.diff(out) > 0.0))

    def test_srgb_to_linear_identity_bounds(self) -> None:
        sample = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        out = srgb_to_linear(sample)
        self.assertAlmostEqual(float(out[0]), 0.0)
        self.assertAlmostEqual(float(out[-1]), 1.0)
        self.assertGreater(float(out[1]), 0.0)
        self.assertLess(float(out[1]), 0.5)

    def test_linear_to_pq_is_monotonic(self) -> None:
        values = np.array([0.0, 0.18, 0.5, 1.0], dtype=np.float32)
        out = linear_to_pq(values, peak_nits=1000.0)
        self.assertTrue(np.all(np.diff(out) >= 0.0))
        self.assertLessEqual(float(out[-1]), 1.0)

    def test_skin_mask_detects_typical_skin_tone(self) -> None:
        frame = np.array([[[0.55, 0.34, 0.22]]], dtype=np.float32)
        mask = estimate_skin_mask(frame)
        self.assertEqual(float(mask[0, 0]), 1.0)

    def test_subtitle_mask_finds_bright_text_in_lower_band(self) -> None:
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        frame[26:30, 18:46] = 255
        luma = np.full((32, 64), 0.2, dtype=np.float32)
        mask = estimate_subtitle_mask(frame, luma)
        self.assertGreater(float(mask[27:29, 20:44].mean()), 0.2)

    def test_noise_mask_stronger_in_dark_noisy_region(self) -> None:
        rng = np.random.default_rng(0)
        base = np.full((16, 16, 3), 0.03, dtype=np.float32)
        noisy = np.clip(base + rng.normal(scale=0.02, size=base.shape).astype(np.float32), 0.0, 1.0)
        luma = np.full((16, 16), 0.03, dtype=np.float32)
        mask = estimate_noise_mask(noisy, luma, 0.08)
        self.assertGreater(float(mask.mean()), 0.05)

    def test_specular_mask_prefers_bright_neutral_pixels(self) -> None:
        frame = np.array([[[0.9, 0.88, 0.86], [0.9, 0.3, 0.1]]], dtype=np.float32)
        luma = np.array([[0.88, 0.45]], dtype=np.float32)
        mask = estimate_specular_mask(frame, luma)
        self.assertGreater(float(mask[0, 0]), float(mask[0, 1]))

    def test_clipped_white_mask_prefers_flat_bright_white(self) -> None:
        frame = np.full((8, 8, 3), [0.96, 0.95, 0.95], dtype=np.float32)
        frame[:, 4:] = [0.96, 0.65, 0.30]
        luma = np.full((8, 8), 0.95, dtype=np.float32)
        luma[:, 4:] = 0.70
        mask = estimate_clipped_white_mask(frame, luma)
        self.assertGreater(float(mask[:, :4].mean()), float(mask[:, 4:].mean()))

    def test_high_chroma_mask_prefers_vivid_regions(self) -> None:
        frame = np.array([[[0.9, 0.1, 0.1], [0.4, 0.38, 0.36]]], dtype=np.float32)
        luma = np.array([[0.4, 0.4]], dtype=np.float32)
        mask = estimate_high_chroma_mask(frame, luma)
        self.assertGreater(float(mask[0, 0]), float(mask[0, 1]))

    def test_memory_color_mask_catches_foliage_like_green(self) -> None:
        frame = np.array([[[0.18, 0.55, 0.12], [0.35, 0.35, 0.35]]], dtype=np.float32)
        luma = np.array([[0.35, 0.35]], dtype=np.float32)
        mask = estimate_memory_color_mask(frame, luma)
        self.assertGreater(float(mask[0, 0]), float(mask[0, 1]))

    def test_ai_gate_suppresses_protected_color_regions(self) -> None:
        gate = build_ai_gate(
            skin_mask=np.array([[0.0, 0.0]], dtype=np.float32),
            subtitle_mask=np.array([[0.0, 0.0]], dtype=np.float32),
            noise_mask=np.array([[0.0, 0.0]], dtype=np.float32),
            clipped_white_mask=np.array([[0.0, 0.0]], dtype=np.float32),
            high_chroma_mask=np.array([[0.9, 0.0]], dtype=np.float32),
            memory_color_mask=np.array([[0.0, 0.0]], dtype=np.float32),
            learned_protection=np.array([[0.0, 0.0]], dtype=np.float32),
        )
        self.assertLess(float(gate[0, 0]), 0.5)
        self.assertAlmostEqual(float(gate[0, 1]), 1.0)

    def test_near_white_rolloff_reduces_upper_luma_gain(self) -> None:
        luma = np.array([[0.4, 0.8, 0.95]], dtype=np.float32)
        rolloff = apply_near_white_rolloff(luma, 0.78, 0.6)
        self.assertAlmostEqual(float(rolloff[0, 0]), 1.0)
        self.assertGreater(float(rolloff[0, 1]), float(rolloff[0, 2]))

    def test_ai_highlight_limiter_suppresses_clipped_and_near_white_regions(self) -> None:
        expansion = np.ones((1, 3), dtype=np.float32)
        luma = np.array([[0.55, 0.86, 0.98]], dtype=np.float32)
        clipped_white = np.array([[0.0, 0.15, 1.0]], dtype=np.float32)
        rolloff = apply_near_white_rolloff(luma, 0.74, 0.72)
        limited = limit_ai_highlight_expansion(expansion, luma, clipped_white, rolloff)
        self.assertAlmostEqual(float(limited[0, 0]), 1.0)
        self.assertLess(float(limited[0, 1]), 0.8)
        self.assertLess(float(limited[0, 2]), 0.2)

    def test_adaptive_highlight_boost_penalizes_flat_white_scenes(self) -> None:
        flat_white = compute_adaptive_highlight_boost(0.8, 0.8, 0.1, 0.05, 0.1, 0.55, 1.05)
        specular_scene = compute_adaptive_highlight_boost(0.8, 0.1, 0.05, 0.5, 0.0, 0.55, 1.05)
        self.assertLess(flat_white, specular_scene)

    def test_temporal_state_detects_scene_cut(self) -> None:
        state = TemporalState()
        first = np.zeros((8, 8), dtype=np.float32)
        chroma_first = np.zeros((8, 8), dtype=np.float32)
        state.update(0.2, 0.9, first, chroma_first, 0.1)
        second = np.ones((8, 8), dtype=np.float32)
        chroma_second = np.ones((8, 8), dtype=np.float32) * 0.5
        _, scene_cut = state.update(0.8, 0.9, second, chroma_second, 0.1)
        self.assertTrue(scene_cut)

    def test_processor_returns_uint16_rgb(self) -> None:
        frame = np.full((8, 8, 3), 180, dtype=np.uint8)
        processor = SDRToHDRProcessor(ProcessorConfig())
        out = processor.process_frame(frame)
        self.assertEqual(out.dtype, np.uint16)
        self.assertEqual(out.shape, frame.shape)
        self.assertLessEqual(int(out.max()), 65535)

    def test_fast_mode_with_processing_scale_preserves_output_shape(self) -> None:
        frame = np.full((24, 40, 3), 150, dtype=np.uint8)
        config = ProcessorConfig(fast_mode=True, processing_scale=0.5)
        processor = SDRToHDRProcessor(config)
        out = processor.process_frame(frame)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, np.uint16)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_torch_cpu_backend_preserves_output_shape(self) -> None:
        frame = np.full((24, 40, 3), 170, dtype=np.uint8)
        config = ProcessorConfig(fast_mode=True, processing_scale=0.75, backend="torch-cpu")
        processor = SDRToHDRProcessor(config)
        out = processor.process_frame(frame)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, np.uint16)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_windows_auto_backend_prefers_cuda(self) -> None:
        with (
            mock.patch("sdr2hdr.core.platform.system", return_value="Windows"),
            mock.patch("sdr2hdr.core.torch.cuda.is_available", return_value=True),
            mock.patch("sdr2hdr.core.torch.backends.mps.is_available", return_value=False),
        ):
            processor = SDRToHDRProcessor.__new__(SDRToHDRProcessor)
            processor.config = ProcessorConfig(backend="auto")
            self.assertEqual(processor._resolve_torch_device(), "cuda")

    @unittest.skipIf(torch is None, "torch not installed")
    def test_cuda_backend_requires_cuda(self) -> None:
        with mock.patch("sdr2hdr.core.torch.cuda.is_available", return_value=False):
            processor = SDRToHDRProcessor.__new__(SDRToHDRProcessor)
            processor.config = ProcessorConfig(backend="cuda")
            self.assertIsNone(processor._resolve_torch_device())


if __name__ == "__main__":
    unittest.main()
