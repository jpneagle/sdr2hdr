import unittest

from sdr2hdr.cli import build_parser


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


if __name__ == "__main__":
    unittest.main()
