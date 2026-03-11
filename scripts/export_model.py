from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sdr2hdr.model import EnhancementUNet


def export_torchscript(model: EnhancementUNet, output_path: Path) -> None:
    # Export TorchScript from a CPU copy so the serialized graph stays portable
    # across CUDA, MPS, and CPU runtimes.
    cpu_model = EnhancementUNet()
    cpu_state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    cpu_model.load_state_dict(cpu_state_dict)
    cpu_model.eval()
    scripted = torch.jit.script(cpu_model)
    torch.jit.save(scripted, str(output_path))


def export_onnx(model: EnhancementUNet, output_path: Path, device: torch.device) -> None:
    dummy = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
        opset_version=17,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export enhancement model checkpoint to TorchScript and/or ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--format", choices=["torchscript", "onnx", "both"], default="torchscript")
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model = EnhancementUNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    output_path = Path(args.output)
    if args.format == "torchscript":
        export_torchscript(model, output_path)
    elif args.format == "onnx":
        export_onnx(model, output_path, device)
    else:
        export_torchscript(model, output_path.with_suffix(".pt"))
        export_onnx(model, output_path.with_suffix(".onnx"), device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
