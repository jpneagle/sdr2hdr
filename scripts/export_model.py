from __future__ import annotations

import argparse

import torch

from sdr2hdr.model import EnhancementUNet


def main() -> int:
    parser = argparse.ArgumentParser(description="Export enhancement model checkpoint to TorchScript.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model = EnhancementUNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    scripted = torch.jit.script(model)
    scripted = torch.jit.optimize_for_inference(scripted)
    torch.jit.save(scripted, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
