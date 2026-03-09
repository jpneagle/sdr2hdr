from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from sdr2hdr.review import REC2020_TO_REC709, linear_to_srgb, pq_to_relative_linear


def extract_raw_frames(input_path: str, output_dir: Path, pix_fmt: str, sample_every: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        input_path,
        "-vf",
        f"select='not(mod(n\\,{sample_every}))'",
        "-vsync",
        "0",
        "-pix_fmt",
        pix_fmt,
        str(output_dir / "frame_%06d.png"),
    ]
    subprocess.run(cmd, check=True)


def srgb_to_linear(frame: np.ndarray) -> np.ndarray:
    frame = np.clip(frame, 0.0, 1.0)
    return np.where(frame <= 0.04045, frame / 12.92, ((frame + 0.055) / 1.055) ** 2.4)


def tone_map_hdr_linear_to_sdr_linear(frame_709_linear: np.ndarray) -> np.ndarray:
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
    return srgb_to_linear(srgb).astype(np.float32)


def convert_frame_to_npz(hdr_frame_path: Path, output_path: Path, peak_nits: float) -> None:
    hdr_bgr16 = cv2.imread(str(hdr_frame_path), cv2.IMREAD_UNCHANGED)
    if hdr_bgr16 is None:
        raise RuntimeError(f"failed to read HDR frame: {hdr_frame_path}")
    hdr_rgb16 = cv2.cvtColor(hdr_bgr16, cv2.COLOR_BGR2RGB)
    hdr_2020_linear = pq_to_relative_linear(hdr_rgb16.astype(np.float32) / 65535.0, peak_nits=peak_nits)
    hdr_709_linear = np.clip(np.tensordot(hdr_2020_linear, REC2020_TO_REC709.T, axes=1), 0.0, 1.5)
    sdr_linear = tone_map_hdr_linear_to_sdr_linear(hdr_709_linear)
    np.savez_compressed(
        output_path,
        sdr_linear=sdr_linear.astype(np.float16),
        hdr_linear=hdr_709_linear.astype(np.float16),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare SDR/HDR training pairs from HDR videos.")
    parser.add_argument("--input-dir", required=True, help="Directory containing HDR videos")
    parser.add_argument("--out-dir", required=True, help="Directory to save .npz training samples")
    parser.add_argument("--sample-every", type=int, default=24, help="Sample every N frames")
    parser.add_argument("--peak-nits", type=float, default=1000.0, help="Peak nits used for relative linear decode")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sdr2hdr-train-") as temp_dir:
        temp_root = Path(temp_dir)
        for video_path in sorted(input_dir.iterdir()):
            if not video_path.is_file():
                continue
            hdr_frames_dir = temp_root / f"{video_path.stem}_hdr_frames"
            extract_raw_frames(str(video_path), hdr_frames_dir, "rgb48le", args.sample_every)
            for hdr_frame_path in sorted(hdr_frames_dir.glob("frame_*.png")):
                output_path = out_dir / f"{video_path.stem}_{hdr_frame_path.stem}.npz"
                convert_frame_to_npz(hdr_frame_path, output_path, peak_nits=args.peak_nits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
