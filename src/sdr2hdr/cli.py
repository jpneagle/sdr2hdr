from __future__ import annotations

import argparse
import platform
import sys

from sdr2hdr.app import (
    ConversionCallbacks,
    ConversionRequest,
    PRESETS,
    X265_PROFILE_DEFAULTS,
    default_encoder_for_platform,
    run_conversion,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert SDR video into HDR10 HEVC output.")
    parser.add_argument("input", help="Input SDR video path")
    parser.add_argument("output", help="Output HDR10 video path")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="high")
    parser.add_argument("--peak-nits", type=float, default=None, help="Target HDR10 peak luminance")
    parser.add_argument("--ai-strength", type=float, default=None, help="Blend factor for enhancement maps")
    parser.add_argument("--highlight-boost", type=float, default=None, help="Base highlight expansion strength")
    parser.add_argument("--detail-boost", type=float, default=None, help="Local contrast enhancement amount")
    parser.add_argument(
        "--encoder",
        choices=["hevc_videotoolbox", "hevc_nvenc", "libx265"],
        default=default_encoder_for_platform(platform.system()),
    )
    parser.add_argument("--x265-mode", choices=sorted(X265_PROFILE_DEFAULTS), default="balanced")
    parser.add_argument("--x265-preset", help="Override libx265 preset, e.g. veryfast, medium, slow")
    parser.add_argument("--x265-crf", type=int, help="Override libx265 CRF value")
    parser.add_argument("--processing-scale", type=float, default=None, help="Internal processing scale, 0.5-1.0")
    parser.add_argument("--fast-mode", action="store_true", help="Use faster approximation filters")
    parser.add_argument("--backend", choices=["auto", "numpy", "torch-cpu", "mps", "cuda"], default="auto")
    parser.add_argument(
        "--scene-cut-mode",
        choices=["auto", "fixed"],
        default="auto",
        help="Reserved for future scene-cut detection strategy",
    )
    parser.add_argument("--model-path", help="Optional TorchScript map-estimation model path")
    parser.add_argument("--device", default="cpu", help="Torch device when --model-path is supplied")
    parser.add_argument("--max-frames", type=int, help="Limit the number of processed frames for debugging")
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        from sdr2hdr.gui import main as gui_main

        return gui_main()
    parser = build_parser()
    args = parser.parse_args(argv)
    request = ConversionRequest(
        input_path=args.input,
        output_path=args.output,
        preset=args.preset,
        encoder=args.encoder,
        x265_mode=args.x265_mode,
        x265_preset=args.x265_preset,
        x265_crf=args.x265_crf,
        peak_nits=args.peak_nits,
        ai_strength=args.ai_strength,
        highlight_boost=args.highlight_boost,
        detail_boost=args.detail_boost,
        processing_scale=args.processing_scale,
        fast_mode=args.fast_mode,
        backend=args.backend,
        model_path=args.model_path,
        device=args.device,
        max_frames=args.max_frames,
        fallback_to_x265_on_hardware_error=args.encoder in {"hevc_videotoolbox", "hevc_nvenc"},
    )
    callbacks = ConversionCallbacks(on_status=None, on_progress=None, on_complete=None, on_error=None)
    result = run_conversion(request, callbacks=callbacks)
    print(f"Converted {result.processed_frames} frames to {result.output_path}")
    return 0
