from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.export_model import export_torchscript
from sdr2hdr.model import EnhancementUNet


class ExportModelTests(unittest.TestCase):
    def test_export_torchscript_stays_backend_neutral(self) -> None:
        model = EnhancementUNet().eval()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.pt"
            export_torchscript(model, output_path)
            loaded = torch.jit.load(str(output_path), map_location="cpu")
            graph_text = str(loaded.inlined_graph)
            self.assertNotIn("cudnn_", graph_text)


if __name__ == "__main__":
    unittest.main()
