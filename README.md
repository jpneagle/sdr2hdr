# sdr2hdr

- [English](#english)
- [日本語](#日本語)
- [中文](#中文)

## English

`sdr2hdr` is an offline SDR-to-HDR10 converter for real-world video.

The project is aimed at practical up-conversion on macOS and Windows, with a GUI-first workflow and a CLI for batch or scripted use. The current pipeline focuses on making SDR footage look natural on HDR displays rather than trying to hallucinate "true HDR capture."

## What It Does

- Converts SDR `BT.709` video to HDR10-compatible `HEVC Main10`
- Automatically deinterlaces interlaced inputs such as broadcast-style `m2ts` before HDR processing
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

### Prerequisites

Required:

- `Python 3.11+`
- `ffmpeg`
- `ffprobe`

Recommended:

- `torch`
  - for `MPS` on Apple Silicon
  - for `CUDA` on Windows + NVIDIA

Hardware-accelerated encoding also depends on your FFmpeg build:

- `hevc_videotoolbox` requires a macOS FFmpeg build with VideoToolbox support
- `hevc_nvenc` requires a Windows FFmpeg build with NVENC support

Quick checks:

```bash
python3 --version
ffmpeg -version
ffprobe -version
```

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### GPU acceleration (strongly recommended)

`torch` is an optional dependency, but without it the pipeline runs on CPU (numpy path) and will be very slow.

**Windows + NVIDIA GPU (CUDA)**

`pip install torch` from PyPI installs a CPU-only build. You must install the CUDA-enabled build explicitly:

```bash
# CUDA 12.6 wheels work with driver-reported CUDA 12.x and 13.x
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Verify that CUDA is available after installation:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Both lines should print `True` and your GPU name. Once confirmed, set `--backend cuda` (or leave `--backend auto`) and restart the app.

**macOS (Apple Silicon)**

```bash
pip install torch
```

The default PyPI build includes MPS support. No extra index URL is needed.

**CPU-only fallback**

If you skip `torch` entirely, the numpy path is used automatically. Expect roughly 0.3 FPS on full-HD frames — suitable only for quick tests.

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
- multiple jobs can be queued and processed one by one
- pressing `Stop` keeps a playable partial output when frames have already been written
- transport-stream inputs such as `m2ts` default to `*_hdr.mp4` outputs

The GUI exposes:

- input / output path selection
- add current job to queue
- add multiple input files to queue
- remove / clear queued jobs
- sequential queue execution
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

## 日本語

`sdr2hdr` は、実写向けのオフライン SDR→HDR10 変換ツールです。macOS と Windows を対象にしており、GUI から複数ジョブをキューに積んで順番に処理できます。目的は「本物の HDR 撮影を再現すること」ではなく、「SDR 動画を HDR ディスプレイで自然に見える形へ安定して変換すること」です。

### 主な特徴

- SDR `BT.709` 動画を HDR10 互換 `HEVC Main10` へ変換
- `m2ts` などのインターレース入力は自動でデインターレース
- 肌、字幕、暗部ノイズ、白飛びしやすい領域を保護
- シーン単位でハイライト強調量を調整
- Apple Silicon では `MPS`、Windows + NVIDIA では `CUDA` を利用可能
- macOS では `VideoToolbox`、Windows では `HEVC NVENC` を高速経路として利用
- ハードウェアエンコード失敗時は `libx265` にフォールバック
- 出力後に HDR メタデータを検証し、不足時は自動修復

### 使い方

前提条件:

- `Python 3.11 以上`
- `ffmpeg`
- `ffprobe`
- GPU 加速を使うなら `torch`

補足:

- macOS で `hevc_videotoolbox` を使うには VideoToolbox 対応の FFmpeg が必要です
- Windows で `hevc_nvenc` を使うには NVENC 対応の FFmpeg が必要です

インストール:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

GUI 起動:

```bash
sdr2hdr
```

CLI 変換:

```bash
sdr2hdr input.mp4 output_hdr.mp4 --preset portrait
```

Windows + RTX 向けの例:

```bash
sdr2hdr input.mp4 output_hdr.mp4 --preset portrait --encoder hevc_nvenc --backend cuda
```

macOS 向けの例:

```bash
sdr2hdr input.mp4 output_hdr.mp4 --preset portrait --encoder hevc_videotoolbox
```

### 推奨設定

- 実写素材の標準: `preset=portrait`
- Apple Silicon:
  - 高速重視: `hevc_videotoolbox`
  - 品質重視: `libx265 + mps`
- Windows + RTX 4090:
  - 高速重視: `hevc_nvenc + auto`
  - 品質重視: `libx265 + cuda`

### 補足

- `m2ts / mts / ts` 入力でも既定出力は `*_hdr.mp4`
- MP4 にそのまま入らない音声は自動で `AAC` に変換
- GUI では複数ファイルをキューに積んで順番に処理可能

## 中文

`sdr2hdr` 是一个面向实拍素材的离线 SDR→HDR10 转换工具，支持 macOS 和 Windows。它的目标不是“伪造真正的 HDR 拍摄素材”，而是把普通 SDR 视频稳定地转换成在 HDR 显示器上看起来更自然的 HDR10 文件。

### 主要特性

- 将 SDR `BT.709` 视频转换为 HDR10 兼容的 `HEVC Main10`
- 对 `m2ts` 等隔行扫描输入自动做反交错
- 保护肤色、字幕、暗部噪点和容易过曝的高亮区域
- 按场景动态调整高亮增强强度
- Apple Silicon 可使用 `MPS`
- Windows + NVIDIA 可使用 `CUDA`
- macOS 可使用 `VideoToolbox` 快速编码
- Windows 可使用 `HEVC NVENC` 快速编码
- 硬件编码失败时自动回退到 `libx265`
- 输出完成后自动检查 HDR 元数据，必要时自动修复

### 用法

前提条件:

- `Python 3.11+`
- `ffmpeg`
- `ffprobe`
- 如果要用 GPU 加速，建议安装 `torch`

补充:

- macOS 使用 `hevc_videotoolbox` 需要 FFmpeg 带有 VideoToolbox 支持
- Windows 使用 `hevc_nvenc` 需要 FFmpeg 带有 NVENC 支持

安装:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

启动 GUI:

```bash
sdr2hdr
```

命令行转换:

```bash
sdr2hdr input.mp4 output_hdr.mp4 --preset portrait
```

Windows + RTX 示例:

```bash
sdr2hdr input.mp4 output_hdr.mp4 --preset portrait --encoder hevc_nvenc --backend cuda
```

macOS 示例:

```bash
sdr2hdr input.mp4 output_hdr.mp4 --preset portrait --encoder hevc_videotoolbox
```

### 推荐设置

- 实拍素材默认建议: `preset=portrait`
- Apple Silicon:
  - 速度优先: `hevc_videotoolbox`
  - 质量优先: `libx265 + mps`
- Windows + RTX 4090:
  - 速度优先: `hevc_nvenc + auto`
  - 质量优先: `libx265 + cuda`

### 备注

- 即使输入是 `m2ts / mts / ts`，默认输出也会使用 `*_hdr.mp4`
- 不能直接封装进 MP4 的音频会自动转成 `AAC`
- GUI 支持把多个视频加入队列后依次处理

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
- Interlaced inputs are automatically deinterlaced with `bwdif` during decode
- Output is HDR10-compatible `HEVC Main10`
- `torch` is optional, but recommended for Apple Silicon or CUDA acceleration
- Windows `hevc_nvenc` requires an FFmpeg build with NVENC support
- The project is optimized for practical HDR display output, not mastering-grade finishing
