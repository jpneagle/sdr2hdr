import unittest

from sdr2hdr.gui import build_backend_options, build_encoder_options, describe_mode_hint, format_ai_strength


class GuiTests(unittest.TestCase):
    def test_build_encoder_options_for_windows(self) -> None:
        options = build_encoder_options("Windows")
        self.assertIn("hevc_nvenc", options)
        self.assertNotIn("hevc_videotoolbox", options)
        self.assertEqual(options["hevc_nvenc"], "NVENC (Fast on NVIDIA)")

    def test_build_backend_options_for_windows(self) -> None:
        options = build_backend_options("Windows")
        self.assertIn("cuda", options)
        self.assertNotIn("mps", options)
        self.assertEqual(options["cuda"], "CUDA (NVIDIA GPU)")

    def test_build_encoder_options_for_macos(self) -> None:
        options = build_encoder_options("Darwin")
        self.assertIn("hevc_videotoolbox", options)
        self.assertNotIn("hevc_nvenc", options)

    def test_describe_mode_hint_mentions_learned_mode_when_model_is_set(self) -> None:
        hint = describe_mode_hint("libx265", "balanced", "cuda", "portrait", "/tmp/model.pt")
        self.assertIn("Learned-map mode", hint)

    def test_format_ai_strength(self) -> None:
        self.assertEqual(format_ai_strength(0.2), "0.20")


if __name__ == "__main__":
    unittest.main()
