import argparse
import unittest

from sdr2hdr.cli import build_parser, parse_resolution


class CLIParserTests(unittest.TestCase):
    def test_required_model_path(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["input.mp4"])

    def test_default_values(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["input.mp4", "--model-path", "model.pt"])
        self.assertEqual(args.preset, "portrait")
        self.assertEqual(args.encoder, "libx265")
        self.assertEqual(args.x265_mode, "balanced")
        self.assertEqual(args.backend, "auto")
        self.assertEqual(args.ai_strength, 0.25)
        self.assertEqual(args.upscale_engine, "ffmpeg")
        self.assertEqual(args.output_scale, 1.0)
        self.assertIsNone(args.target_resolution)
        self.assertEqual(args.scaler, "lanczos")
        self.assertEqual(args.rtx_video_quality, "high")
        self.assertFalse(args.no_fallback_to_x265_on_hardware_error)
        self.assertFalse(args.discard_partial_output_on_cancel)

    def test_output_path_optional(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["input.mp4", "--model-path", "model.pt"])
        self.assertIsNone(args.output_path)

    def test_output_path_explicit(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["input.mp4", "output.mp4", "--model-path", "model.pt"])
        self.assertEqual(args.output_path, "output.mp4")

    def test_preset_choices(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["input.mp4", "--model-path", "m.pt", "--preset", "nonexistent"])

    def test_backend_choices(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["input.mp4", "--model-path", "m.pt", "--backend", "xpu"])
        self.assertEqual(args.backend, "xpu")
        with self.assertRaises(SystemExit):
            parser.parse_args(["input.mp4", "--model-path", "m.pt", "--backend", "invalid"])

    def test_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "input.mp4",
            "--model-path", "model.pt",
            "--no-fallback-to-x265-on-hardware-error",
            "--discard-partial-output-on-cancel",
        ])
        self.assertTrue(args.no_fallback_to_x265_on_hardware_error)
        self.assertTrue(args.discard_partial_output_on_cancel)

    def test_ai_strength_float(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["input.mp4", "--model-path", "m.pt", "--ai-strength", "0.6"])
        self.assertAlmostEqual(args.ai_strength, 0.6)

    def test_high_resolution_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "input.mp4",
            "--model-path", "m.pt",
            "--upscale-engine", "rtx-video",
            "--output-scale", "2.0",
            "--scaler", "bicubic",
            "--rtx-video-quality", "ultra",
        ])
        self.assertEqual(args.upscale_engine, "rtx-video")
        self.assertEqual(args.output_scale, 2.0)
        self.assertEqual(args.scaler, "bicubic")
        self.assertEqual(args.rtx_video_quality, "ultra")

    def test_target_resolution_parser(self) -> None:
        self.assertEqual(parse_resolution("3840x2160"), (3840, 2160))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_resolution("3840")


if __name__ == "__main__":
    unittest.main()
