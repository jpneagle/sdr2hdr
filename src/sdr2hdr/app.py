from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from sdr2hdr.ai import HeuristicEnhancer, TorchMapEnhancer
from sdr2hdr.core import ProcessorConfig, SDRToHDRProcessor
from sdr2hdr.io import (
    ffprobe_video,
    finalize_process,
    has_expected_hdr_metadata,
    open_decoder,
    open_encoder,
    read_frame,
    restamp_hdr_metadata,
)

PRESETS = {
    "poc": ProcessorConfig(
        peak_nits=600.0,
        ai_strength=0.15,
        detail_boost=0.12,
        scene_smoothing=0.82,
        processing_scale=0.75,
        fast_mode=True,
    ),
    "balanced": ProcessorConfig(
        peak_nits=1000.0,
        ai_strength=0.30,
        detail_boost=0.20,
        scene_smoothing=0.88,
        fast_mode=True,
    ),
    "high": ProcessorConfig(peak_nits=1000.0, ai_strength=0.42, detail_boost=0.28, scene_smoothing=0.92),
    "portrait": ProcessorConfig(
        peak_nits=800.0,
        ai_strength=0.18,
        detail_boost=0.12,
        scene_smoothing=0.93,
        scene_cut_threshold=0.14,
        highlight_boost=0.72,
        subtitle_protection=0.90,
        shadow_noise_floor=0.10,
        skin_protection=0.82,
        shadow_rolloff=0.62,
        processing_scale=0.85,
        fast_mode=True,
        clipped_white_protection=0.78,
        near_white_rolloff_start=0.74,
        near_white_rolloff_strength=0.72,
    ),
}

X265_PROFILE_DEFAULTS = {
    "preview": {"preset": "veryfast", "crf": 20},
    "balanced": {"preset": "medium", "crf": 16},
    "final": {"preset": "slow", "crf": 14},
}


@dataclass
class ConversionRequest:
    input_path: str
    output_path: str
    preset: str = "portrait"
    encoder: str = "hevc_videotoolbox"
    x265_mode: str = "balanced"
    x265_preset: str | None = None
    x265_crf: int | None = None
    peak_nits: float | None = None
    ai_strength: float | None = None
    highlight_boost: float | None = None
    detail_boost: float | None = None
    processing_scale: float | None = None
    fast_mode: bool = False
    backend: str = "auto"
    model_path: str | None = None
    device: str = "cpu"
    max_frames: int | None = None
    fallback_to_x265_on_hardware_error: bool = False
    keep_partial_output_on_cancel: bool = True
    verify_hdr_metadata: bool = True


@dataclass
class ConversionResult:
    output_path: str
    processed_frames: int
    total_frames: int | None
    cancelled: bool = False


@dataclass
class ConversionCallbacks:
    on_status: Callable[[str], None] | None = None
    on_progress: Callable[[int, int | None, float | None], None] | None = None
    on_complete: Callable[[ConversionResult], None] | None = None
    on_error: Callable[[str], None] | None = None


class CancelToken:
    def __init__(self) -> None:
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True


def build_output_path(input_path: str) -> str:
    path = Path(input_path)
    if not path.suffix:
        return str(path.with_name(f"{path.name}_hdr"))
    suffix = path.suffix.lower()
    output_suffix = ".mp4" if suffix in {".m2ts", ".mts", ".m2t", ".ts"} else path.suffix
    return str(path.with_name(f"{path.stem}_hdr{output_suffix}"))


def build_request_config(request: ConversionRequest) -> tuple[ProcessorConfig, str, int]:
    config = replace(PRESETS[request.preset])
    if request.peak_nits is not None:
        config.peak_nits = request.peak_nits
    if request.ai_strength is not None:
        config.ai_strength = request.ai_strength
    if request.highlight_boost is not None:
        config.highlight_boost = request.highlight_boost
    if request.detail_boost is not None:
        config.detail_boost = request.detail_boost
    if request.processing_scale is not None:
        config.processing_scale = request.processing_scale
    if request.fast_mode:
        config.fast_mode = True
    config.backend = request.backend
    profile = X265_PROFILE_DEFAULTS[request.x265_mode]
    x265_preset = request.x265_preset or profile["preset"]
    x265_crf = request.x265_crf if request.x265_crf is not None else profile["crf"]
    return config, x265_preset, x265_crf


def validate_request(request: ConversionRequest) -> None:
    input_path = Path(request.input_path)
    output_path = Path(request.output_path)
    if not input_path.exists():
        raise ValueError(f"Input file does not exist: {input_path}")
    if not request.output_path.strip():
        raise ValueError("Output path is required.")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")
    if request.preset not in PRESETS:
        raise ValueError(f"Unknown preset: {request.preset}")
    if request.x265_mode not in X265_PROFILE_DEFAULTS:
        raise ValueError(f"Unknown x265 mode: {request.x265_mode}")


def _emit_status(callbacks: ConversionCallbacks | None, message: str) -> None:
    if callbacks and callbacks.on_status:
        callbacks.on_status(message)


def _emit_progress(callbacks: ConversionCallbacks | None, processed: int, total: int | None, fps: float | None) -> None:
    if callbacks and callbacks.on_progress:
        callbacks.on_progress(processed, total, fps)


def _emit_complete(callbacks: ConversionCallbacks | None, result: ConversionResult) -> None:
    if callbacks and callbacks.on_complete:
        callbacks.on_complete(result)


def _emit_error(callbacks: ConversionCallbacks | None, message: str) -> None:
    if callbacks and callbacks.on_error:
        callbacks.on_error(message)


def is_hardware_encoder_failure(message: str) -> bool:
    lowered = message.lower()
    return (
        "videotoolbox" in lowered
        or "compression session" in lowered
        or "nvenc" in lowered
        or "nvidia" in lowered
        or "no capable devices found" in lowered
        or "cannot load nvcuda" in lowered
        or "unsupported device" in lowered
    )


def is_videotoolbox_failure(message: str) -> bool:
    lowered = message.lower()
    return "videotoolbox" in lowered or "compression session" in lowered


def default_encoder_for_platform(system_name: str | None = None) -> str:
    system_name = system_name or platform.system()
    if system_name == "Darwin":
        return "hevc_videotoolbox"
    if system_name == "Windows":
        return "hevc_nvenc"
    return "libx265"


def _terminate_process(process: object | None) -> None:
    if process is None:
        return
    try:
        process.terminate()
    except Exception:
        return


def _wait_terminated_process(process: object | None) -> None:
    if process is None:
        return
    for handle_name in ("stdin", "stdout", "stderr"):
        handle = getattr(process, handle_name, None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
    try:
        process.wait(timeout=5)
    except Exception:
        pass


def run_conversion(
    request: ConversionRequest,
    callbacks: ConversionCallbacks | None = None,
    cancel_token: CancelToken | None = None,
) -> ConversionResult:
    current_request = request
    attempted_fallback = False
    while True:
        try:
            return _run_conversion_once(current_request, callbacks=callbacks, cancel_token=cancel_token)
        except RuntimeError as exc:
            if (
                current_request.fallback_to_x265_on_hardware_error
                and current_request.encoder in {"hevc_videotoolbox", "hevc_nvenc"}
                and not attempted_fallback
                and is_hardware_encoder_failure(str(exc))
            ):
                attempted_fallback = True
                _emit_status(callbacks, f"{current_request.encoder} failed; falling back to libx265")
                try:
                    os.remove(current_request.output_path)
                except FileNotFoundError:
                    pass
                current_request = replace(current_request, encoder="libx265")
                continue
            raise


def _run_conversion_once(
    request: ConversionRequest,
    callbacks: ConversionCallbacks | None = None,
    cancel_token: CancelToken | None = None,
) -> ConversionResult:
    validate_request(request)
    config, x265_preset, x265_crf = build_request_config(request)
    info = ffprobe_video(request.input_path)
    total_frames = request.max_frames if request.max_frames is not None else info.frames
    processor = SDRToHDRProcessor(
        config,
        enhancer=TorchMapEnhancer(request.model_path, device=request.device) if request.model_path else HeuristicEnhancer(),
    )
    decoder = open_decoder(request.input_path, info)
    encoder = open_encoder(
        request.output_path,
        request.input_path,
        info,
        config.peak_nits,
        encoder=request.encoder,
        x265_preset=x265_preset,
        x265_crf=x265_crf,
    )
    processed = 0
    cancelled = False
    encoder_broken_pipe = False
    start = time.monotonic()
    _emit_status(callbacks, "Preparing conversion")
    try:
        while True:
            if cancel_token and cancel_token.cancel_requested:
                cancelled = True
                _emit_status(callbacks, "Cancelling")
                break
            if request.max_frames is not None and processed >= request.max_frames:
                break
            frame = read_frame(decoder, info.width, info.height)
            if frame is None:
                break
            hdr_frame = processor.process_frame(frame)
            assert encoder.stdin is not None
            try:
                encoder.stdin.write(hdr_frame.tobytes())
            except BrokenPipeError:
                encoder_broken_pipe = True
                break
            processed += 1
            if processed == 1:
                _emit_status(callbacks, "Converting")
            elapsed = max(time.monotonic() - start, 1e-6)
            fps = processed / elapsed
            _emit_progress(callbacks, processed, total_frames, fps)
    except Exception as exc:
        _emit_error(callbacks, str(exc))
        raise
    finally:
        if cancelled:
            _terminate_process(decoder)
            _wait_terminated_process(decoder)
            finalize_process(encoder, "encoder", allow_broken_pipe=True)
        else:
            encoder_error: RuntimeError | None = None
            try:
                finalize_process(
                    encoder,
                    "encoder",
                    allow_broken_pipe=bool(request.max_frames) or encoder_broken_pipe,
                )
            except RuntimeError as exc:
                encoder_error = exc
            try:
                finalize_process(
                    decoder,
                    "decoder",
                    allow_broken_pipe=bool(request.max_frames) or encoder_broken_pipe or encoder_error is not None,
                )
            except RuntimeError:
                if encoder_error is None:
                    raise
            if encoder_error is not None:
                raise encoder_error
    if cancelled:
        if request.keep_partial_output_on_cancel and processed > 0:
            restamp_hdr_metadata(request.output_path)
        if not request.keep_partial_output_on_cancel or processed == 0:
            try:
                os.remove(request.output_path)
            except FileNotFoundError:
                pass
        result = ConversionResult(
            output_path=request.output_path,
            processed_frames=processed,
            total_frames=total_frames,
            cancelled=True,
        )
        _emit_complete(callbacks, result)
        return result
    if processed == 0:
        raise RuntimeError("No frames were processed. Check the input path and video stream.")
    if request.verify_hdr_metadata and not has_expected_hdr_metadata(request.output_path):
        _emit_status(callbacks, "HDR metadata missing; repairing output tags")
        restamp_hdr_metadata(request.output_path)
    result = ConversionResult(output_path=request.output_path, processed_frames=processed, total_frames=total_frames)
    _emit_status(callbacks, "Completed")
    _emit_complete(callbacks, result)
    return result
