from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
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
        "-vsync",
        "0",
        "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def open_encoder(
    output_path: str,
    source_path: str,
    info: VideoInfo,
    peak_nits: float,
    encoder: str = "hevc_videotoolbox",
    x265_preset: str = "medium",
    x265_crf: int = 16,
) -> subprocess.Popen[bytes]:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mastering = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
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
    if encoder == "hevc_videotoolbox":
        cmd += [
            "-pix_fmt",
            "p010le",
            "-vf",
            "scale=in_color_matrix=bt2020:out_color_matrix=bt2020",
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
            "-c:a",
            "copy",
            output_path,
        ]
    elif encoder == "hevc_nvenc":
        cmd += [
            "-pix_fmt",
            "p010le",
            "-vf",
            "scale=in_color_matrix=bt2020:out_color_matrix=bt2020",
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
            "-c:a",
            "copy",
            output_path,
        ]
    else:
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
            "-c:a",
            "copy",
            output_path,
        ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def read_frame(process: subprocess.Popen[bytes], width: int, height: int) -> np.ndarray | None:
    frame_size = width * height * 3
    assert process.stdout is not None
    buffer = process.stdout.read(frame_size)
    if len(buffer) != frame_size:
        return None
    return np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)


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


def restamp_hdr_metadata(path: str) -> None:
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
