import unittest

from sdr2hdr.gui import build_backend_options, build_encoder_options, describe_mode_hint


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

    def test_describe_mode_hint_warns_when_portrait_ml_has_no_model(self) -> None:
        hint = describe_mode_hint("libx265", "balanced", "mps", "portrait-ml", "")
        self.assertIn("requires a learned model path", hint)

    def test_describe_mode_hint_mentions_cuda_for_portrait_ml(self) -> None:
        hint = describe_mode_hint("libx265", "balanced", "cuda", "portrait-ml", "/tmp/model.pt")
        self.assertIn("NVIDIA GPU", hint)


if __name__ == "__main__":
    unittest.main()
