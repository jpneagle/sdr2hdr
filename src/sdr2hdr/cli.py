from __future__ import annotations

import argparse
from pathlib import Path

from .app import (
    HDR_STYLE_DEFAULTS,
    INTEL_SR_DEVICE_OPTIONS,
    INTEL_SR_MODEL_OPTIONS,
    INPUT_EOTF_OPTIONS,
    PRESETS,
    RTX_VIDEO_QUALITY_OPTIONS,
    SCALER_OPTIONS,
    TONE_DIFFUSE_WHITE,
    UPSCALE_ENGINE_OPTIONS,
    X265_PROFILE_DEFAULTS,
    ConversionCallbacks,
    ConversionRequest,
    build_output_path,
    run_conversion,
    validate_request,
)


def parse_resolution(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Resolution must be WIDTHxHEIGHT, for example 3840x2160.") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution dimensions must be positive.")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert SDR video to HDR10.")
    parser.add_argument("input_path", help="Input SDR video path")
    parser.add_argument("output_path", nargs="?", help="Output HDR video path")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="portrait")
    parser.add_argument("--encoder", default="libx265")
    parser.add_argument("--x265-mode", choices=sorted(X265_PROFILE_DEFAULTS), default="balanced")
    parser.add_argument("--backend", choices=["auto", "numpy", "cuda", "xpu", "mps"], default="auto")
    parser.add_argument("--hdr-style", choices=sorted(HDR_STYLE_DEFAULTS), default="natural")
    parser.add_argument(
        "--tone",
        choices=sorted(TONE_DIFFUSE_WHITE),
        default="vivid",
        help="Brightness anchoring: vivid maps SDR white to peak nits, reference anchors it at 203 nits (BT.2408)",
    )
    parser.add_argument(
        "--input-eotf",
        choices=sorted(INPUT_EOTF_OPTIONS),
        default="srgb",
        help="Transfer function used to decode the SDR input (bt1886 for broadcast/BT.709 video)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-path", required=True, help="Path to a TorchScript .pt enhancement model")
    parser.add_argument("--ai-strength", type=float, default=0.25)
    parser.add_argument(
        "--upscale-engine",
        choices=UPSCALE_ENGINE_OPTIONS,
        default="ffmpeg",
        help="High-resolution engine: ffmpeg uses final encode scaling; rtx-video uses RTX Video SDK; intel-vino uses Intel OpenVINO super resolution.",
    )
    parser.add_argument(
        "--output-scale",
        type=float,
        default=1.0,
        help="Scale the HDR output resolution after SDR-to-HDR processing, for example 2.0 for 1080p to 4K.",
    )
    parser.add_argument(
        "--target-resolution",
        type=parse_resolution,
        help="Exact even output resolution as WIDTHxHEIGHT. Cannot be combined with --output-scale other than 1.0.",
    )
    parser.add_argument(
        "--scaler",
        choices=SCALER_OPTIONS,
        default="lanczos",
        help="FFmpeg scaler used for high-resolution output.",
    )
    parser.add_argument(
        "--rtx-video-quality",
        choices=RTX_VIDEO_QUALITY_OPTIONS,
        default="high",
        help="RTX Video SDK super resolution quality when --upscale-engine rtx-video is used.",
    )
    parser.add_argument(
        "--intel-sr-model",
        choices=INTEL_SR_MODEL_OPTIONS,
        default="sr-1032",
        help="Intel OpenVINO super resolution model: sr-1032 (4x) or sr-1033 (3x).",
    )
    parser.add_argument(
        "--intel-sr-device",
        choices=INTEL_SR_DEVICE_OPTIONS,
        default="AUTO",
        help="Intel OpenVINO inference device: AUTO, CPU, or GPU (Intel iGPU/Arc).",
    )
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
    target_width = None
    target_height = None
    if args.target_resolution is not None:
        target_width, target_height = args.target_resolution

    request = ConversionRequest(
        input_path=str(Path(args.input_path)),
        output_path=output_path,
        preset=args.preset,
        encoder=args.encoder,
        x265_mode=args.x265_mode,
        backend=args.backend,
        hdr_style=args.hdr_style,
        tone=args.tone,
        input_eotf=args.input_eotf,
        upscale_engine=args.upscale_engine,
        output_scale=args.output_scale,
        target_width=target_width,
        target_height=target_height,
        scaler=args.scaler,
        rtx_video_quality=args.rtx_video_quality,
        intel_sr_model=args.intel_sr_model,
        intel_sr_device=args.intel_sr_device,
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
