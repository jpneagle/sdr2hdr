from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from sdr2hdr.io import ffprobe_video

REC2020_TO_REC709 = np.array(
    [
        [1.6605, -0.5876, -0.0728],
        [-0.1246, 1.1329, -0.0083],
        [-0.0182, -0.1006, 1.1187],
    ],
    dtype=np.float32,
)


def parse_times(raw: str | None) -> list[float]:
    if not raw:
        return []
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def default_sample_times(duration: float | None, count: int) -> list[float]:
    if count <= 0:
        return []
    if duration is None or duration <= 0.0:
        return [float(index) for index in range(count)]
    edge = min(duration * 0.1, 3.0)
    start = edge
    end = max(duration - edge, start)
    if count == 1:
        return [min(duration * 0.5, end)]
    return np.linspace(start, end, count).round(3).tolist()


def pq_to_relative_linear(frame: np.ndarray, peak_nits: float) -> np.ndarray:
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 32.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 128.0
    c3 = 2392.0 / 128.0
    frame = np.clip(frame, 0.0, 1.0)
    power = np.power(frame, 1.0 / m2)
    normalized = np.power(np.maximum(power - c1, 0.0) / np.maximum(c2 - c3 * power, 1e-6), 1.0 / m1)
    return np.clip(normalized * (10000.0 / peak_nits), 0.0, 16.0)


def linear_to_srgb(frame: np.ndarray) -> np.ndarray:
    frame = np.clip(frame, 0.0, 1.0)
    return np.where(frame <= 0.0031308, frame * 12.92, 1.055 * np.power(frame, 1.0 / 2.4) - 0.055)


def tone_map_hdr_preview(frame_2020_pq: np.ndarray, peak_nits: float = 1000.0) -> np.ndarray:
    frame_2020_linear = pq_to_relative_linear(frame_2020_pq, peak_nits)
    return tone_map_linear_preview(frame_2020_linear)


def tone_map_linear_preview(frame_2020_linear: np.ndarray) -> np.ndarray:
    frame_709_linear = np.clip(np.tensordot(frame_2020_linear, REC2020_TO_REC709.T, axes=1), 0.0, None)
    luma = np.clip(
        0.2126 * frame_709_linear[..., 0]
        + 0.7152 * frame_709_linear[..., 1]
        + 0.0722 * frame_709_linear[..., 2],
        0.0,
        None,
    )
    white = max(float(np.percentile(luma, 99.7)), 0.5)
    exposed = frame_709_linear / white
    mapped = exposed / (1.0 + exposed)
    srgb = linear_to_srgb(np.clip(mapped, 0.0, 1.0))
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


def extract_frame(input_path: str, time_sec: float, output_path: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{time_sec:.3f}",
        "-i",
        input_path,
        "-frames:v",
        "1",
    ]
    cmd += [output_path]
    subprocess.run(cmd, check=True)


def extract_raw_frame(input_path: str, time_sec: float, width: int, height: int, pix_fmt: str) -> np.ndarray:
    bytes_per_channel = 2 if pix_fmt == "rgb48le" else 1
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{time_sec:.3f}",
        "-i",
        input_path,
        "-frames:v",
        "1",
        "-pix_fmt",
        pix_fmt,
        "-f",
        "rawvideo",
        "-",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True)
    expected = width * height * 3 * bytes_per_channel
    if len(result.stdout) != expected:
        raise RuntimeError(f"unexpected raw frame size: got {len(result.stdout)}, expected {expected}")
    dtype = np.uint16 if bytes_per_channel == 2 else np.uint8
    return np.frombuffer(result.stdout, dtype=dtype).reshape(height, width, 3)


def save_hdr_tiff(input_path: str, time_sec: float, output_path: str, width: int, height: int) -> None:
    frame_rgb16 = extract_raw_frame(input_path, time_sec, width, height, "rgb48le")
    if not cv2.imwrite(output_path, frame_rgb16[..., ::-1]):
        raise RuntimeError(f"failed to write HDR TIFF: {output_path}")


def save_hdr_preview_png(input_path: str, time_sec: float, output_path: str, width: int, height: int) -> None:
    frame_rgb16 = extract_raw_frame(input_path, time_sec, width, height, "rgb48le")
    preview_rgb8 = tone_map_hdr_preview(frame_rgb16.astype(np.float32) / 65535.0)
    if not cv2.imwrite(output_path, preview_rgb8[..., ::-1]):
        raise RuntimeError(f"failed to write HDR preview PNG: {output_path}")


def save_hdr_exr(input_path: str, time_sec: float, output_path: str, width: int, height: int) -> None:
    frame_rgb16 = extract_raw_frame(input_path, time_sec, width, height, "rgb48le").astype(np.float32) / 65535.0
    frame_2020_linear = pq_to_relative_linear(frame_rgb16, peak_nits=1000.0)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gbrpf32le",
        "-s",
        f"{width}x{height}",
        "-i",
        "-",
        "-frames:v",
        "1",
        "-c:v",
        "exr",
        "-compression",
        "zip1",
        "-format",
        "half",
        output_path,
    ]
    process = subprocess.run(cmd, input=frame_2020_linear.astype(np.float32).tobytes(), check=True)
    if process.returncode != 0:
        raise RuntimeError(f"failed to write HDR EXR: {output_path}")


def add_label_band(image: np.ndarray, label: str, band_color: tuple[int, int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    band_height = max(36, height // 12)
    band = np.full((band_height, width, 3), band_color, dtype=np.uint8)
    cv2.putText(
        band,
        label,
        (16, band_height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([band, image])


def save_side_by_side(left_path: str, right_path: str, output_path: str, title: str) -> None:
    left = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right = cv2.imread(right_path, cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise RuntimeError("failed to load extracted preview frames")
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
    left_labeled = add_label_band(left, "SDR", (48, 74, 96))
    right_labeled = add_label_band(right, "HDR Preview", (88, 92, 44))
    gap = np.full((left_labeled.shape[0], 24, 3), 18, dtype=np.uint8)
    combined = np.hstack([left_labeled, gap, right_labeled])
    title_height = max(44, combined.shape[0] // 14)
    header = np.full((title_height, combined.shape[1], 3), 12, dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (16, title_height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(output_path, np.vstack([header, combined]))


def save_contact_sheet(image_paths: list[Path], output_path: Path, columns: int = 2) -> None:
    if not image_paths:
        return
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in image_paths]
    if any(image is None for image in images):
        raise RuntimeError("failed to load one or more comparison images")
    assert all(image is not None for image in images)
    cell_h = max(image.shape[0] for image in images if image is not None)
    cell_w = max(image.shape[1] for image in images if image is not None)
    rows = (len(images) + columns - 1) // columns
    canvas = np.full((rows * cell_h, columns * cell_w, 3), 8, dtype=np.uint8)
    for index, image in enumerate(images):
        assert image is not None
        row = index // columns
        col = index % columns
        y = row * cell_h
        x = col * cell_w
        if image.shape[:2] != (cell_h, cell_w):
            image = cv2.resize(image, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        canvas[y : y + cell_h, x : x + cell_w] = image
    cv2.imwrite(str(output_path), canvas)


def build_frames_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract preview frames from a video.")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output_dir", help="Directory to write PNG frames")
    parser.add_argument("--times", help="Comma-separated times in seconds")
    parser.add_argument("--count", type=int, default=4, help="Number of evenly spaced frames when --times is omitted")
    parser.add_argument("--prefix", default="frame", help="Output file prefix")
    parser.add_argument("--hdr-preview", action="store_true", help="Apply a simple HDR-to-SDR preview tone map")
    parser.add_argument("--hdr-tiff", action="store_true", help="Write 16-bit TIFF frames for HDR sources")
    parser.add_argument("--hdr-exr", action="store_true", help="Write linear OpenEXR frames for HDR sources")
    return parser


def build_compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate side-by-side SDR/HDR comparison frames.")
    parser.add_argument("sdr_input", help="Original SDR video path")
    parser.add_argument("hdr_input", help="Converted HDR10 video path")
    parser.add_argument("output_dir", help="Directory to write comparison PNG files")
    parser.add_argument("--times", help="Comma-separated times in seconds")
    parser.add_argument("--count", type=int, default=4, help="Number of evenly spaced frames when --times is omitted")
    return parser


def frames_main(argv: list[str] | None = None) -> int:
    parser = build_frames_parser()
    args = parser.parse_args(argv)
    info = ffprobe_video(args.input)
    times = parse_times(args.times) or default_sample_times(info.duration, args.count)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, time_sec in enumerate(times):
        if args.hdr_exr:
            output_path = output_dir / f"{args.prefix}_{index:02d}_{time_sec:06.3f}s.exr"
            save_hdr_exr(args.input, time_sec, str(output_path), info.width, info.height)
        elif args.hdr_tiff:
            output_path = output_dir / f"{args.prefix}_{index:02d}_{time_sec:06.3f}s.tiff"
            save_hdr_tiff(args.input, time_sec, str(output_path), info.width, info.height)
        else:
            output_path = output_dir / f"{args.prefix}_{index:02d}_{time_sec:06.3f}s.png"
        if args.hdr_preview and not args.hdr_tiff and not args.hdr_exr:
            save_hdr_preview_png(args.input, time_sec, str(output_path), info.width, info.height)
        elif not args.hdr_tiff and not args.hdr_exr:
            extract_frame(args.input, time_sec, str(output_path))
        print(output_path)
    return 0


def compare_main(argv: list[str] | None = None) -> int:
    parser = build_compare_parser()
    args = parser.parse_args(argv)
    sdr_info = ffprobe_video(args.sdr_input)
    hdr_info = ffprobe_video(args.hdr_input)
    if args.times:
        times = parse_times(args.times)
    else:
        duration_candidates = [value for value in [sdr_info.duration, hdr_info.duration] if value is not None]
        compare_duration = min(duration_candidates) if duration_candidates else None
        times = default_sample_times(compare_duration, args.count)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="sdr2hdr-review-") as temp_dir:
        temp_path = Path(temp_dir)
        for index, time_sec in enumerate(times):
            sdr_path = temp_path / f"sdr_{index:02d}.png"
            hdr_preview_path = temp_path / f"hdr_{index:02d}.png"
            comparison_path = output_dir / f"compare_{index:02d}_{time_sec:06.3f}s.png"
            hdr_exr_path = output_dir / f"hdr_{index:02d}_{time_sec:06.3f}s.exr"
            extract_frame(args.sdr_input, time_sec, str(sdr_path))
            save_hdr_preview_png(args.hdr_input, time_sec, str(hdr_preview_path), hdr_info.width, hdr_info.height)
            save_hdr_exr(args.hdr_input, time_sec, str(hdr_exr_path), hdr_info.width, hdr_info.height)
            save_side_by_side(str(sdr_path), str(hdr_preview_path), str(comparison_path), f"t={time_sec:.3f}s")
            comparison_paths.append(comparison_path)
            print(comparison_path)
            print(hdr_exr_path)
    save_contact_sheet(comparison_paths, output_dir / "contact_sheet.png")
    print(output_dir / "contact_sheet.png")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review helpers for sdr2hdr outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    frames_parser = subparsers.add_parser("frames", help="Extract preview frames")
    for action in build_frames_parser()._actions[1:]:
        frames_parser._add_action(action)
    compare_parser = subparsers.add_parser("compare", help="Generate side-by-side comparisons")
    for action in build_compare_parser()._actions[1:]:
        compare_parser._add_action(action)

    args = parser.parse_args(argv)
    if args.command == "frames":
        forwarded = []
        if args.times:
            forwarded.extend(["--times", args.times])
        forwarded.extend([args.input, args.output_dir, "--count", str(args.count), "--prefix", args.prefix])
        if args.hdr_preview:
            forwarded.append("--hdr-preview")
        if args.hdr_tiff:
            forwarded.append("--hdr-tiff")
        if args.hdr_exr:
            forwarded.append("--hdr-exr")
        return frames_main(forwarded)

    forwarded = [args.sdr_input, args.hdr_input, args.output_dir, "--count", str(args.count)]
    if args.times:
        forwarded.extend(["--times", args.times])
    return compare_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
