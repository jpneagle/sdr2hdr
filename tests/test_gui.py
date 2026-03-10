import tempfile
import unittest
from pathlib import Path

from sdr2hdr.gui import (
    build_backend_options,
    build_encoder_options,
    describe_mode_hint,
    filter_models_for_backend,
    format_ai_strength,
    list_available_models,
)


class GuiTests(unittest.TestCase):
    def test_build_encoder_options_for_windows(self) -> None:
        options = build_encoder_options("Windows")
        self.assertIn("hevc_nvenc", options)
        self.assertNotIn("hevc_videotoolbox", options)
        self.assertEqual(options["hevc_nvenc"], "NVENC (Fast on NVIDIA)")

    def test_build_backend_options_for_windows(self) -> None:
        options = build_backend_options("Windows")
        self.assertIn("cuda", options)
        self.assertIn("directml", options)
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

    def test_list_available_models_returns_supported_model_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            (models_dir / "a.pt").write_bytes(b"")
            (models_dir / "b.onnx").write_bytes(b"")
            (models_dir / "b.txt").write_bytes(b"")
            models = list_available_models(models_dir)
            self.assertEqual([path.name for path in models], ["a.pt", "b.onnx"])

    def test_filter_models_for_directml_returns_only_onnx(self) -> None:
        models = [Path("a.pt"), Path("b.onnx")]
        filtered = filter_models_for_backend(models, "directml", "Windows")
        self.assertEqual(filtered, [Path("b.onnx")])

    def test_filter_models_for_auto_returns_both_on_windows(self) -> None:
        models = [Path("a.pt"), Path("b.onnx")]
        filtered = filter_models_for_backend(models, "auto", "Windows")
        self.assertEqual(filtered, models)


if __name__ == "__main__":
    unittest.main()
