from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int | None
    pix_fmt: str | None
    duration: float | None
    field_order: str | None


def ffprobe_first_audio_codec(path: str) -> str | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "json",
        path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        return None
    return streams[0].get("codec_name")


def ffprobe_video(path: str) -> VideoInfo:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,pix_fmt,duration,field_order:format=duration",
        "-of",
        "json",
        path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    fmt = payload.get("format", {})
    num, den = stream.get("avg_frame_rate", "0/1").split("/")
    fps = float(num) / max(float(den), 1.0)
    frames = stream.get("nb_frames")
    duration = stream.get("duration") or fmt.get("duration")
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        frames=int(frames) if frames and frames != "N/A" else None,
        pix_fmt=stream.get("pix_fmt"),
        duration=float(duration) if duration and duration != "N/A" else None,
        field_order=stream.get("field_order"),
    )


def is_interlaced_video(info: VideoInfo) -> bool:
    field_order = (info.field_order or "").lower()
    return field_order not in {"", "unknown", "progressive"}


def open_decoder(path: str, info: VideoInfo) -> subprocess.Popen[bytes]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
    ]
    if is_interlaced_video(info):
        cmd += [
            "-vf",
            "bwdif=mode=send_frame:parity=auto:deint=all",
        ]
    cmd += [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
    ]
    if info.fps > 0:
        # Normalize to a constant frame rate: the encoder muxes at a fixed -r,
        # so passthrough decode of VFR sources would drift out of A/V sync.
        cmd += ["-fps_mode", "cfr", "-r", f"{info.fps:.06f}"]
    else:
        cmd += ["-fps_mode", "passthrough"]
    cmd += ["-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def build_audio_output_args(output_path: str, source_path: str) -> list[str]:
    output_suffix = Path(output_path).suffix.lower()
    audio_codec = ffprobe_first_audio_codec(source_path)
    if audio_codec is None:
        return []
    if output_suffix == ".mp4":
        copy_safe_codecs = {"aac", "mp3", "ac3", "eac3", "alac"}
        if audio_codec not in copy_safe_codecs:
            return ["-c:a", "aac", "-b:a", "192k"]
    return ["-c:a", "copy"]


def build_hdr_scale_filter(
    source_width: int,
    source_height: int,
    output_width: int | None = None,
    output_height: int | None = None,
    scaler: str = "lanczos",
    include_color_matrix: bool = True,
) -> str | None:
    if output_width is None or output_height is None:
        if not include_color_matrix:
            return None
        return "scale=in_color_matrix=bt2020:out_color_matrix=bt2020"
    if output_width == source_width and output_height == source_height:
        if not include_color_matrix:
            return None
        return "scale=in_color_matrix=bt2020:out_color_matrix=bt2020"
    options = [
        str(output_width),
        str(output_height),
        f"flags={scaler}",
    ]
    if include_color_matrix:
        options += ["in_color_matrix=bt2020", "out_color_matrix=bt2020"]
    return "scale=" + ":".join(options)


def open_encoder(
    output_path: str,
    source_path: str,
    info: VideoInfo,
    peak_nits: float,
    encoder: str = "hevc_videotoolbox",
    x265_preset: str = "medium",
    x265_crf: int = 16,
    max_cll_override: tuple[int, int] | None = None,
    output_width: int | None = None,
    output_height: int | None = None,
    scaler: str = "lanczos",
) -> subprocess.Popen[bytes]:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mastering = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
    if max_cll_override is not None:
        max_cll = f"{max_cll_override[0]},{max_cll_override[1]}"
    else:
        max_cll = f"{int(peak_nits)},{max(int(peak_nits * 0.4), 1)}"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb48le",
        "-s",
        f"{info.width}x{info.height}",
        "-r",
        f"{info.fps:.06f}",
        "-i",
        "-",
        "-i",
        source_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
    ]
    audio_args = build_audio_output_args(output_path, source_path)
    hardware_filter = build_hdr_scale_filter(
        info.width,
        info.height,
        output_width,
        output_height,
        scaler=scaler,
        include_color_matrix=True,
    )
    software_filter = build_hdr_scale_filter(
        info.width,
        info.height,
        output_width,
        output_height,
        scaler=scaler,
        include_color_matrix=True,
    )
    if encoder == "hevc_videotoolbox":
        cmd += [
            "-pix_fmt",
            "p010le",
            "-c:v",
            "hevc_videotoolbox",
            "-profile:v",
            "main10",
            "-tag:v",
            "hvc1",
            "-bsf:v",
            "hevc_metadata=colour_primaries=9:transfer_characteristics=16:matrix_coefficients=9",
            "-allow_sw",
            "1",
            "-prio_speed",
            "true",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "smpte2084",
            "-colorspace",
            "bt2020nc",
        ]
        if hardware_filter is not None:
            cmd += ["-vf", hardware_filter]
        cmd += audio_args + [output_path]
    elif encoder == "hevc_nvenc":
        cmd += [
            "-pix_fmt",
            "p010le",
            "-c:v",
            "hevc_nvenc",
            "-profile:v",
            "main10",
            "-preset",
            "p5",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            "18",
            "-b:v",
            "0",
            "-tag:v",
            "hvc1",
            "-bsf:v",
            "hevc_metadata=colour_primaries=9:transfer_characteristics=16:matrix_coefficients=9",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "smpte2084",
            "-colorspace",
            "bt2020nc",
        ]
        if hardware_filter is not None:
            cmd += ["-vf", hardware_filter]
        cmd += audio_args + [output_path]
    elif encoder == "hevc_qsv":
        cmd += [
            "-pix_fmt",
            "p010le",
            "-c:v",
            "hevc_qsv",
            "-profile:v",
            "main10",
            "-preset",
            "slow",
            "-global_quality",
            "18",
            "-tag:v",
            "hvc1",
            "-bsf:v",
            "hevc_metadata=colour_primaries=9:transfer_characteristics=16:matrix_coefficients=9",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "smpte2084",
            "-colorspace",
            "bt2020nc",
        ]
        if hardware_filter is not None:
            cmd += ["-vf", hardware_filter]
        cmd += audio_args + [output_path]
    else:
        if (
            software_filter is not None
            and output_width is not None
            and output_height is not None
            and (output_width, output_height) != (info.width, info.height)
        ):
            cmd += ["-vf", software_filter]
        cmd += [
            "-c:v",
            "libx265",
            "-pix_fmt",
            "yuv420p10le",
            "-tag:v",
            "hvc1",
            "-preset",
            x265_preset,
            "-crf",
            str(x265_crf),
            "-bsf:v",
            "hevc_metadata=colour_primaries=9:transfer_characteristics=16:matrix_coefficients=9",
            "-x265-params",
            f"hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display={mastering}:max-cll={max_cll}",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "smpte2084",
            "-colorspace",
            "bt2020nc",
        ]
        cmd += audio_args + [output_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def read_frame(process: subprocess.Popen[bytes], width: int, height: int) -> np.ndarray | None:
    frame_size = width * height * 3
    assert process.stdout is not None
    buffer = process.stdout.read(frame_size)
    if len(buffer) != frame_size:
        return None
    return np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)


def start_stderr_drain(process: subprocess.Popen[bytes]) -> None:
    """Continuously read a process's stderr so the OS pipe buffer never fills up.

    Without this, ffmpeg can block mid-stream once it has written enough to
    stderr, deadlocking the whole pipeline. The collected output is picked up
    by finalize_process for error reporting.
    """
    stream = process.stderr
    if stream is None or getattr(process, "_stderr_drain_thread", None) is not None:
        return
    chunks: list[bytes] = []

    def _drain() -> None:
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        except (OSError, ValueError):
            pass

    thread = threading.Thread(target=_drain, daemon=True)
    thread.start()
    process._stderr_drain_chunks = chunks  # type: ignore[attr-defined]
    process._stderr_drain_thread = thread  # type: ignore[attr-defined]


def finalize_process(process: subprocess.Popen[bytes], name: str, allow_broken_pipe: bool = False) -> None:
    stderr = b""
    if process.stdin is not None:
        try:
            process.stdin.close()
        except BrokenPipeError:
            if not allow_broken_pipe:
                raise
    if process.stdout is not None:
        process.stdout.close()
    drain_thread = getattr(process, "_stderr_drain_thread", None)
    if drain_thread is not None:
        return_code = process.wait()
        drain_thread.join(timeout=5)
        stderr = b"".join(getattr(process, "_stderr_drain_chunks", []))
        if process.stderr is not None:
            process.stderr.close()
        rendered = stderr.decode("utf-8", errors="replace").strip()
        if allow_broken_pipe and return_code != 0 and "Broken pipe" in rendered:
            return
        if return_code != 0:
            raise RuntimeError(f"{name} failed with code {return_code}: {rendered}")
        return
    if process.stderr is not None:
        stderr = process.stderr.read()
        process.stderr.close()
    return_code = process.wait()
    rendered = stderr.decode("utf-8", errors="replace").strip()
    if allow_broken_pipe and return_code != 0 and "Broken pipe" in rendered:
        return
    if return_code != 0:
        raise RuntimeError(f"{name} failed with code {return_code}: {rendered}")


def quote_command(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def restamp_hdr_metadata(path: str, max_cll: tuple[int, int] | None = None) -> None:
    source = Path(path)
    if not source.exists():
        return
    with tempfile.NamedTemporaryFile(suffix=source.suffix, dir=source.parent, delete=False) as handle:
        temp_path = Path(handle.name)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v?",
        "-map",
        "0:a?",
        "-c",
        "copy",
        "-tag:v",
        "hvc1",
        "-bsf:v",
        "hevc_metadata=colour_primaries=9:transfer_characteristics=16:matrix_coefficients=9",
        "-movflags",
        "+faststart",
        "-color_primaries",
        "bt2020",
        "-color_trc",
        "smpte2084",
        "-colorspace",
        "bt2020nc",
        str(temp_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        temp_path.replace(source)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def has_expected_hdr_metadata(path: str) -> bool:
    source = Path(path)
    if not source.exists():
        return False
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=color_space,color_transfer,color_primaries",
        "-of",
        "json",
        str(source),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        return False
    stream = streams[0]
    return (
        stream.get("color_space") == "bt2020nc"
        and stream.get("color_transfer") == "smpte2084"
        and stream.get("color_primaries") == "bt2020"
    )
