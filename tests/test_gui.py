from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sdr2hdr.gui import (
    build_backend_options,
    build_encoder_options,
    filter_models_for_backend,
    format_ai_strength,
    list_available_models,
)


class GuiTests(unittest.TestCase):
    def test_build_encoder_options_for_windows(self) -> None:
        options = build_encoder_options("Windows")
        self.assertIn("libx265", options)
        self.assertIn("hevc_nvenc", options)
        self.assertNotIn("hevc_videotoolbox", options)

    def test_build_encoder_options_for_macos(self) -> None:
        options = build_encoder_options("Darwin")
        self.assertIn("libx265", options)
        self.assertIn("hevc_videotoolbox", options)
        self.assertNotIn("hevc_nvenc", options)

    def test_build_backend_options_for_windows(self) -> None:
        options = build_backend_options("Windows")
        self.assertIn("auto", options)
        self.assertIn("cuda", options)
        self.assertIn("numpy", options)
        self.assertNotIn("directml", options)

    def test_build_backend_options_for_macos(self) -> None:
        options = build_backend_options("Darwin")
        self.assertIn("auto", options)
        self.assertIn("mps", options)
        self.assertIn("numpy", options)
        self.assertNotIn("cuda", options)

    def test_format_ai_strength(self) -> None:
        self.assertEqual(format_ai_strength(0.25), "0.25")
        self.assertEqual(format_ai_strength(0.2), "0.20")

    def test_list_available_models_returns_pt_only(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.pt").write_bytes(b"pt")
            (root / "b.onnx").write_bytes(b"onnx")
            (root / "notes.txt").write_text("x", encoding="utf-8")
            models = list_available_models(root)
        self.assertEqual([path.name for path in models], ["a.pt"])

    def test_filter_models_for_backend_returns_pt_only(self) -> None:
        models = [Path("a.pt"), Path("b.onnx")]
        filtered = filter_models_for_backend(models, "auto", "Windows")
        self.assertEqual(filtered, [Path("a.pt")])


if __name__ == "__main__":
    unittest.main()
