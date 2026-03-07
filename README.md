# sdr2hdr

`sdr2hdr` is an offline SDR-to-HDR10 converter for real-world video.

The project is aimed at practical up-conversion on macOS and Windows, with a GUI-first workflow and a CLI for batch or scripted use. The current pipeline focuses on making SDR footage look natural on HDR displays rather than trying to hallucinate "true HDR capture."

## What It Does

- Converts SDR `BT.709` video to HDR10-compatible `HEVC Main10`
- Preserves midtones and expands highlights instead of simply stretching the whole image
- Protects skin, subtitles, dark noisy areas, and clipped white regions
- Applies scene-aware highlight control to reduce white blowout on difficult footage
- Supports Apple Silicon acceleration with `MPS`
- Supports NVIDIA acceleration with `CUDA`
- Supports fast macOS encoding with `VideoToolbox`
- Supports fast Windows encoding with `HEVC NVENC`
- Falls back to `libx265` when hardware encoding is unavailable
- Verifies HDR metadata after export and repairs tags automatically if needed

## Current Focus

This project is currently tuned for:

- live-action footage
- offline conversion
- macOS, especially M-series Macs
- Windows systems with NVIDIA RTX GPUs

It is not designed around:

- animation-specific grading
- RAW / Log workflows
- Dolby Vision or HLG output

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## GUI

Launch the app with:

```bash
sdr2hdr
```

Current behavior:

- `sdr2hdr` with no arguments launches the GUI
- on macOS, the GUI defaults to `VideoToolbox` as the encoder
- on Windows, the GUI defaults to `NVENC` as the encoder
- if hardware encoding fails, the GUI falls back to `libx265`
- pressing `Stop` keeps a playable partial output when frames have already been written

The GUI exposes:

- input / output path selection
- preset selection
- encoder selection
- speed / quality selection
- backend selection
- progress, status, and logs
- open output / open output folder actions

## CLI

CLI conversion still works as before:

```bash
sdr2hdr input.mp4 output_hdr.mp4 --preset portrait
```

Common examples:

```bash
# Fast preview on Apple Silicon
sdr2hdr input.mp4 output_hdr.mp4 --preset portrait --encoder hevc_videotoolbox

# Fast preview on Windows + RTX
sdr2hdr input.mp4 output_hdr.mp4 --preset portrait --encoder hevc_nvenc --backend cuda

# MPS processing + x265 preview export
sdr2hdr input.mp4 output_hdr.mp4 --preset portrait --encoder libx265 --backend mps --x265-mode preview

# Balanced export
sdr2hdr input.mp4 output_hdr.mp4 --preset portrait --encoder libx265 --backend mps --x265-mode balanced

# Final export
sdr2hdr input.mp4 output_hdr.mp4 --preset portrait --encoder libx265 --backend mps --x265-mode final
```

Useful options:

- `--preset poc|balanced|high|portrait`
- `--encoder hevc_videotoolbox|hevc_nvenc|libx265`
- `--backend auto|mps|cuda|torch-cpu|numpy`
- `--x265-mode preview|balanced|final`
- `--max-frames N`

## Presets

- `portrait`
  - recommended default for live-action footage
  - stronger skin protection
  - more conservative highlight expansion
  - stronger white protection
  - fast subtitle mask path

- `balanced`
  - general-purpose preset
  - now uses the faster detail path for better CPU-side performance

- `high`
  - more aggressive quality-oriented settings

- `poc`
  - lightweight preset for fast tests

## Recommended Settings

For Apple Silicon live-action footage:

- quickest turnaround
  - `preset=portrait`
  - `encoder=hevc_videotoolbox`

- daily use
  - `preset=portrait`
  - `encoder=libx265`
  - `backend=mps`
  - `x265-mode=balanced`

- final export
  - `preset=portrait`
  - `encoder=libx265`
  - `backend=mps`
  - `x265-mode=final`

For Windows + RTX 4090 live-action footage:

- quickest turnaround
  - `preset=portrait`
  - `encoder=hevc_nvenc`
  - `backend=auto`

- daily use
  - `preset=portrait`
  - `encoder=hevc_nvenc`
  - `backend=cuda`

- final export
  - `preset=portrait`
  - `encoder=libx265`
  - `backend=cuda`
  - `x265-mode=final`

For 4K material, slower throughput is expected. At that point, encoding and full-frame processing dominate runtime.

## Quality Pipeline

The converter currently includes:

- SDR linearization
- temporal exposure smoothing
- scene cut detection
- adaptive scene-level highlight boost
- skin protection
- subtitle / burned-in text protection
- dark noise suppression
- specular / sky / clipped-white analysis
- near-white rolloff
- HDR10 PQ encoding

The processing path has been optimized for Apple Silicon:

- Torch / MPS path with reduced CPU synchronization points
- cached PQ constants and Torch kernels
- cached Rec.2020 matrix and sky gradient
- reused blur operations
- lighter subtitle mask in `fast_mode`

The processing path also supports NVIDIA-backed processing on Windows when `torch` has CUDA support.

## HDR Metadata Handling

Completed exports are checked after writing.

If expected HDR tags are missing, the tool automatically restamps the file so that the output keeps:

- `color_space=bt2020nc`
- `color_transfer=smpte2084`
- `color_primaries=bt2020`

Partial outputs saved after `Stop` are also restamped.

To inspect a file manually:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,pix_fmt,color_space,color_transfer,color_primaries \
  -of json output_hdr.mp4
```

## Review Tools

Extract SDR frames:

```bash
sdr2hdr-frames input.mp4 review/frames --times 0.5,2.0,4.0
```

Extract tonemapped HDR preview PNGs:

```bash
sdr2hdr-frames output_hdr.mp4 review/hdr_frames --hdr-preview --count 4
```

Extract linear HDR EXR frames:

```bash
sdr2hdr-frames output_hdr.mp4 review/hdr_exr --hdr-exr --count 4
```

Extract HDR TIFF frames:

```bash
sdr2hdr-frames output_hdr.mp4 review/hdr_tiff --hdr-tiff --count 4
```

Generate side-by-side comparisons:

```bash
sdr2hdr-compare input.mp4 output_hdr.mp4 review/compare --count 4
```

This writes:

- comparison PNGs
- HDR EXR frames
- a `contact_sheet.png`

## Notes

- Input is assumed to be SDR `BT.709`
- Output is HDR10-compatible `HEVC Main10`
- `torch` is optional, but recommended for Apple Silicon or CUDA acceleration
- Windows `hevc_nvenc` requires an FFmpeg build with NVENC support
- The project is optimized for practical HDR display output, not mastering-grade finishing
