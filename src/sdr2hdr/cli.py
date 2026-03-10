from __future__ import annotations

import argparse
from pathlib import Path

from .app import (
    PRESETS,
    X265_PROFILE_DEFAULTS,
    ConversionCallbacks,
    ConversionRequest,
    build_output_path,
    run_conversion,
    validate_request,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert SDR video to HDR10.")
    parser.add_argument("input_path", help="Input SDR video path")
    parser.add_argument("output_path", nargs="?", help="Output HDR video path")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="portrait")
    parser.add_argument("--encoder", default="libx265")
    parser.add_argument("--x265-mode", choices=sorted(X265_PROFILE_DEFAULTS), default="balanced")
    parser.add_argument("--backend", choices=["auto", "numpy", "cuda", "mps"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-path", required=True, help="Path to a TorchScript .pt enhancement model")
    parser.add_argument("--ai-strength", type=float, default=0.25)
    parser.add_argument(
        "--no-fallback-to-x265-on-hardware-error",
        action="store_true",
        help="Disable automatic fallback to libx265 when hardware encoding fails",
    )
    parser.add_argument(
        "--discard-partial-output-on-cancel",
        action="store_true",
        help="Remove partial output instead of keeping it when cancellation happens",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = args.output_path or build_output_path(args.input_path)
    model_path = str(Path(args.model_path))
    if Path(model_path).suffix.lower() != ".pt":
        parser.error("--model-path must point to a .pt TorchScript model.")

    request = ConversionRequest(
        input_path=str(Path(args.input_path)),
        output_path=output_path,
        preset=args.preset,
        encoder=args.encoder,
        x265_mode=args.x265_mode,
        backend=args.backend,
        device=args.device,
        model_path=model_path,
        ai_strength=args.ai_strength,
        fallback_to_x265_on_hardware_error=not args.no_fallback_to_x265_on_hardware_error,
        keep_partial_output_on_cancel=not args.discard_partial_output_on_cancel,
    )
    validate_request(request)

    callbacks = ConversionCallbacks(
        on_status=lambda message: print(message, flush=True),
        on_progress=lambda processed, total, fps: print(
            f"{processed}/{total or '?'} frames ({fps:.1f} fps)" if fps else f"{processed}/{total or '?'} frames",
            flush=True,
        ),
        on_complete=lambda result: print(
            f"cancelled after {result.processed_frames} frames"
            if result.cancelled
            else f"completed: {result.output_path}",
            flush=True,
        ),
        on_error=lambda message: print(message, flush=True),
    )
    run_conversion(request, callbacks=callbacks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
