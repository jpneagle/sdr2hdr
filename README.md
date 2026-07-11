> [!IMPORTANT]
> Due to the weakening Japanese yen, it has become increasingly difficult to afford not only the equipment needed to continue development, but even basic daily necessities. If you find this project useful, please consider supporting its development. Your help would mean a great deal.
>
> [Support this project on Buy Me a Coffee](https://buymeacoffee.com/eaglejp2b)

# sdr2hdr

## 日本語

SDR 動画を HDR10 向けに変換する Python ツールです。GUI と CLI の両方を備えています。

現行版は `AI model 前提` の運用です。変換時には学習済み TorchScript モデル (`.pt`) を指定するか、GUI の `models/` プルダウンから選択する必要があります。

### Overview

- 入力は通常の SDR 動画です。
- 出力は HDR10 メタデータ付きの動画です。
- GUI は queue 実行に対応しています。
- AI モデルは `models/` フォルダ内の `.pt` ファイルを使用します。

### Requirements

- Python
- `ffmpeg` と `ffprobe` が実行可能であること(`ffmpeg` 5.1 以上)
- PyTorch を含む依存関係
- RTX Video SDK 超解像を使う場合は、RTX GPU と `nvidia-vfx`(`pip install -e ".[rtx]"`)
- Intel QSV エンコードを使う場合は、対応Intel GPU、ドライバー、および `hevc_qsv` 対応FFmpeg

OS ごとの backend は次の通りです。

- Windows: `Auto`, `CUDA`, `XPU`, `CPU / NumPy`
- macOS: `Auto`, `MPS`, `CPU / NumPy`
- Linux: `Auto`, `XPU`, `CPU / NumPy`
- その他: `Auto`, `CPU / NumPy`

`Auto` は使える環境で GPU backend を優先し、使えない場合は CPU 側へ寄せます。Intel GPU の XPU backend には XPU 対応の PyTorch ビルドが必要です。

Windows のGUIでは `QSV (Fast on Intel)` を選ぶとIntel Quick Sync VideoでHEVC Main10をエンコードします。QSVの初期化に失敗した場合は `libx265` へフォールバックします。

### Setup

依存関係をインストールし、学習済みモデルを `models/` に置きます。

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[ai]"
```

モデル配置例:

```text
models/
  enhancement_model_reuse_v1.pt
```

### Quick Start

#### GUI

```powershell
python -m sdr2hdr.gui
```

GUI の基本動作:

- `Input` と `Output` を指定
- `Preset` は既定で `portrait`
- `Tone` は既定で `vivid`、`Input EOTF` は既定で `bt1886`
- `Output Size` は既定で `Source`。高解像度出力が必要な場合は `2x`、`4x`、または `Custom`
- `Custom` では `Target Size` に `3840 x 2160` のような偶数解像度を指定
- `Upscale Engine` は既定で `FFmpeg Scaler`。RTX Video SDK を使う場合は `RTX Video SDK` を選択し、`RTX Quality` で品質レベルを指定
- `AI Model` で `models/` 内の `.pt` を選択
- `AI Strength` は既定で `0.25`
- `Add To Queue` または `Add Files` で queue へ追加
- `Start Queue` で順次変換

#### CLI

```powershell
python -m sdr2hdr.cli input.mp4 output_hdr.mp4 --model-path models\enhancement_model_reuse_v1.pt
```

`output_path` を省略した場合は、入力ファイル名の末尾に `_hdr` を付けた名前が自動生成されます。

```powershell
python -m sdr2hdr.cli input.mp4 --model-path models\enhancement_model_reuse_v1.pt
```

### Sample Video

YouTube comparison sample:

[![SDR to HDR comparison sample](https://img.youtube.com/vi/2OFI6urxHxI/maxresdefault.jpg)](https://youtu.be/2OFI6urxHxI)

The sample video compares:

- Left: original SDR footage
- Right: HDR output converted with `sdr2hdr`

### GUI

#### Main Controls

- `Preset`
  - 既定値は `portrait`
- `HDR Style`
  - `natural`(既定)、`cinematic`、`night` からハイライト/シャドウの傾向を選択
- `Tone`
  - 既定値は `vivid`(SDR の白を peak nits に配置する、効果の分かりやすい絵)。`reference` は BT.2408 準拠で白を 203 nits に固定する控えめな絵
- `Input EOTF`
  - 既定値は `bt1886`(放送/BT.709 系動画向け)。PC 由来のソースは `srgb`
- `Upscale Engine`
  - `FFmpeg Scaler`(既定) または `RTX Video SDK`
  - `RTX Video SDK` はフレームごとに NGX の超解像セッションを 1 回だけロードして使い回し、SDR→HDR 処理の直前で各フレームを超解像
- `Output Size`
  - `Source`(既定)、`2x`、`4x`、`Custom` から出力解像度を選択
- `Target Size`
  - `Output Size` が `Custom` の場合に使用する明示的な偶数解像度
- `Scaler`
  - 既定値は `Lanczos`。`FFmpeg Scaler` 使用時の ffmpeg スケーラを選択
- `RTX Quality`
  - `RTX Video SDK` 使用時の超解像品質(`Low`/`Medium`/`High`(既定)/`Ultra`)
- `Encoder`
  - 環境に応じて `libx265`、`NVENC`、`VideoToolbox` を選択
- `Speed/Quality`
  - `Preview`, `Balanced`, `Final`
- `Backend`
  - OS ごとの対応 backend から選択
- `AI Model`
  - `models/` 直下の `.pt` をプルダウン表示
- `Refresh`
  - `models/` を再スキャン
- `AI Strength`
  - 既定値は `0.25`

#### Preset / HDR Style / Tone の使い分け

3つとも絵に影響しますが、役割が異なります。

- `Preset`: **何を変換するか**で選ぶ(人物中心なら `portrait`)
- `HDR Style`: **仕上がりの雰囲気**で選ぶ(迷ったら `natural`)
- `Tone`: **明るさの基準**。既定の `vivid` は HDR の効果が分かりやすい派手な絵。放送グレーディングに近い控えめな絵にしたい場合は `reference`
- `Input EOTF`: 好みではなく**ソースの種類**で決める(TV/カメラ動画 = `bt1886`、PC/Web 由来 = `srgb`)

GUI では各項目にマウスを乗せると説明がツールチップで表示されます。

#### Queue

GUI は複数ジョブの queue 実行に対応しています。

- `Add To Queue`
  - 現在の入力設定を queue に追加
- `Add Files`
  - 複数ファイルをまとめて queue に追加
- `Remove Selected`
  - 選択中の queue 項目を削除
- `Clear Queue`
  - queue を全削除
- `Start Queue`
  - queue を順次処理
- `Stop Current`
  - 実行中ジョブの停止を要求

#### Queue Status

Queue の status 表示は現在次の 7 種類です。

- `QUEUED`
- `STARTING`
- `RUNNING`
- `CANCELLING`
- `OK`
- `FAILED`
- `CANCELLED`

`Stop Current` を押した場合は、まず `CANCELLING` になり、終了時に `CANCELLED` へ確定します。

ジョブが `FAILED` になった場合はログに記録され、queue の残りはそのまま継続実行されます。

#### Cancel Behavior

- キャンセル時は partial output を保持する前提です。
- GUI の進捗欄には `partial output saved` と表示されます。

### CLI

現行 CLI の基本仕様:

- `input_path` は必須
- `output_path` は省略可能
- `--model-path` は必須
- `--model-path` は `.pt` モデルを指定
- `--preset` の既定値は `portrait`
- `--hdr-style` は `natural`(既定), `cinematic`, `night`
- `--backend` は `auto`, `numpy`, `cuda`, `mps`
- `--ai-strength` の既定値は `0.25`
- `--tone` は `vivid`(既定。SDR の白を peak nits に配置)または `reference`(BT.2408 準拠で白を 203 nits に固定し、それ以上をハイライト用に確保)
- `--input-eotf` は `srgb`(既定)または `bt1886`(放送/BT.709 系の動画ソース向け)
- `--upscale-engine` は `ffmpeg`(既定)または `rtx-video`
- `--output-scale` は HDR 変換後の出力解像度倍率(既定 `1.0`)。例: `2.0` で 1080p 入力を 4K 出力
- `--target-resolution` は `3840x2160` のような明示的な偶数解像度。`--output-scale` とは併用不可
- `--scaler` は `lanczos`(既定), `bicubic`, `bilinear`
- `--rtx-video-quality` は RTX Video SDK 使用時の超解像品質。`low`, `medium`, `high`(既定), `ultra`

例:

```powershell
python -m sdr2hdr.cli input.mp4 output_hdr.mp4 `
  --preset portrait `
  --backend auto `
  --encoder libx265 `
  --x265-mode balanced `
  --model-path models\enhancement_model_reuse_v1.pt `
  --ai-strength 0.25 `
  --output-scale 2.0 `
  --scaler lanczos
```

RTX Video SDK 超解像を使う例:

```powershell
python -m sdr2hdr.cli input.mp4 output_hdr_4k.mp4 `
  --model-path models\enhancement_model_reuse_v1.pt `
  --upscale-engine rtx-video `
  --output-scale 2.0 `
  --rtx-video-quality high
```

RTX Video SDK の超解像は `nvidia-vfx`(`pip install -e ".[rtx]"`)経由で NVIDIA NGX を利用します。ジョブ全体で 1 つの超解像セッションをロードして使い回すため、フレームごとや外部プロセスごとに NGX を再初期化することはありません。

### Models

- GUI は `models/` フォルダを参照します。
- 参照先は環境変数 `SDR2HDR_MODELS_DIR` で上書きできます。
- 読み込むのは `.pt` ファイルのみです。
- モデル未配置時は GUI のプルダウンに有効候補が出ません。
- CLI では `--model-path` に明示指定します。

推奨:

- 配布用・運用用モデルは `models/` にまとめる
- ファイル名で日付やバージョンを区別する

### Notes

- 現行 README は `利用者向け` の内容に絞っています。
- `peak nits` などの内部パラメータは GUI からは直接設定できません。
- VFR(可変フレームレート)入力は平均フレームレートの CFR に正規化して処理します。
- PQ 量子化時にはバンディング低減のためのディザリングを適用します。
- `.onnx` や DirectML は現行の利用手順には含めていません。
- AI モデルなしでの運用は前提にしていません。

### License

This project is licensed under the MIT License.

## English

`sdr2hdr` is a Python tool for converting SDR video to HDR10-style output. It provides both a GUI and a CLI.

The current workflow assumes `AI model usage`. You must provide a trained TorchScript model (`.pt`) for conversion, either from the GUI dropdown or with `--model-path` in the CLI.

### Overview

- Input: regular SDR video
- Output: HDR10-tagged video
- The GUI supports queued batch processing
- AI models are loaded from `.pt` files in the `models/` folder

### Requirements

- Python
- `ffmpeg` and `ffprobe` available in `PATH` (`ffmpeg` 5.1 or newer)
- Project dependencies including PyTorch
- An RTX GPU and `nvidia-vfx` (`pip install -e ".[rtx]"`) when RTX Video SDK super resolution is used

Backend options by OS:

- Windows: `Auto`, `CUDA`, `CPU / NumPy`
- macOS: `Auto`, `MPS`, `CPU / NumPy`
- Other platforms: `Auto`, `CPU / NumPy`

`Auto` prefers a GPU backend when available and falls back toward CPU processing otherwise.

### Setup

Create a virtual environment, install the package, and place a trained model in `models/`.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[ai]"
```

Example model layout:

```text
models/
  enhancement_model_reuse_v1.pt
```

### Quick Start

#### GUI

```powershell
python -m sdr2hdr.gui
```

Basic GUI workflow:

- Set `Input` and `Output`
- `Preset` defaults to `portrait`
- `Tone` defaults to `vivid` and `Input EOTF` defaults to `bt1886`
- `Output Size` defaults to `Source`; choose `2x`, `4x`, or `Custom` for higher-resolution output
- `Custom` uses `Target Size`, such as `3840 x 2160`, and requires even dimensions
- `Upscale Engine` defaults to `FFmpeg Scaler`; choose `RTX Video SDK` and set `RTX Quality` to use RTX Video SDK super resolution
- Select a `.pt` model from `AI Model`
- `AI Strength` defaults to `0.25`
- Add jobs with `Add To Queue` or `Add Files`
- Run them with `Start Queue`

#### CLI

```powershell
python -m sdr2hdr.cli input.mp4 output_hdr.mp4 --model-path models\enhancement_model_reuse_v1.pt
```

If `output_path` is omitted, the tool automatically creates a name with `_hdr` appended.

```powershell
python -m sdr2hdr.cli input.mp4 --model-path models\enhancement_model_reuse_v1.pt
```

### Sample Video

YouTube 比較サンプル:

[![SDR→HDR 比較サンプル](https://img.youtube.com/vi/2OFI6urxHxI/maxresdefault.jpg)](https://youtu.be/2OFI6urxHxI)

このサンプル動画では次を比較しています。

- 左: 元の SDR 映像
- 右: `sdr2hdr` で変換した HDR 映像

### GUI

#### Main Controls

- `Preset`
  - Default: `portrait`
- `HDR Style`
  - `natural` (default), `cinematic`, or `night` highlight/shadow behavior
- `Tone`
  - Default: `vivid` (SDR white mapped to peak nits for a clearly visible HDR effect). `reference` follows BT.2408 with SDR white anchored at 203 nits for a subtler picture
- `Input EOTF`
  - Default: `bt1886` (for broadcast/BT.709 video). Use `srgb` for PC-origin sources
- `Upscale Engine`
  - `FFmpeg Scaler` (default) or `RTX Video SDK`
  - `RTX Video SDK` loads a single NGX super-resolution session once and reuses it for every frame, running just before the SDR-to-HDR pass
- `Output Size`
  - `Source` (default), `2x`, `4x`, or `Custom`
- `Target Size`
  - Exact even output resolution used when `Output Size` is `Custom`
- `Scaler`
  - Default: `Lanczos`; selects the ffmpeg scaler used with `FFmpeg Scaler`
- `RTX Quality`
  - Super-resolution quality when `RTX Video SDK` is used (`Low`/`Medium`/`High` (default)/`Ultra`)
- `Encoder`
  - `libx265`, `NVENC`, or `VideoToolbox` depending on platform
- `Speed/Quality`
  - `Preview`, `Balanced`, `Final`
- `Backend`
  - Available backends depend on the current OS
- `AI Model`
  - Dropdown populated from `.pt` files in `models/`
- `Refresh`
  - Rescans `models/`
- `AI Strength`
  - Default: `0.25`

#### Choosing between Preset, HDR Style, and Tone

All three affect the picture, but they answer different questions.

- `Preset`: pick by **what you are converting** (`portrait` for people-centric footage)
- `HDR Style`: pick by **the mood you want** (`natural` if unsure)
- `Tone`: the **brightness standard**. The default `vivid` gives a punchy, clearly-HDR picture; switch to `reference` for a subtler, broadcast-style grade
- `Input EOTF`: determined by **the source, not by taste** (TV/camera video = `bt1886`, PC/web content = `srgb`)

In the GUI, hovering over each control shows a tooltip with the same guidance.

#### Queue

The GUI supports multi-job queue execution.

- `Add To Queue`
  - Adds the current form values to the queue
- `Add Files`
  - Adds multiple files to the queue
- `Remove Selected`
  - Removes selected queue items
- `Clear Queue`
  - Clears the queue
- `Start Queue`
  - Starts sequential processing
- `Stop Current`
  - Requests cancellation for the current job

#### Queue Status

The current queue status labels are:

- `QUEUED`
- `STARTING`
- `RUNNING`
- `CANCELLING`
- `OK`
- `FAILED`
- `CANCELLED`

If you press `Stop Current`, the job first moves to `CANCELLING` and then settles on `CANCELLED`.

When a job becomes `FAILED`, the error is written to the log and the rest of the queue keeps running.

#### Cancel Behavior

- Partial output is kept on cancellation
- The GUI progress text reports `partial output saved`

### CLI

Current CLI behavior:

- `input_path` is required
- `output_path` is optional
- `--model-path` is required
- `--model-path` must point to a `.pt` model
- `--preset` defaults to `portrait`
- `--hdr-style` is `natural` (default), `cinematic`, or `night`
- `--backend` supports `auto`, `numpy`, `cuda`, `mps`
- `--ai-strength` defaults to `0.25`
- `--tone` is `vivid` (default, maps SDR white to peak nits) or `reference` (BT.2408: anchors SDR white at 203 nits, reserving the range above for highlights)
- `--input-eotf` is `srgb` (default) or `bt1886` (for broadcast/BT.709 video sources)
- `--upscale-engine` is `ffmpeg` (default) or `rtx-video`
- `--output-scale` scales the HDR output resolution after conversion (default `1.0`), for example `2.0` turns 1080p into 4K output
- `--target-resolution` sets an exact even output size such as `3840x2160`; do not combine it with `--output-scale` other than `1.0`
- `--scaler` is `lanczos` (default), `bicubic`, or `bilinear`
- `--rtx-video-quality` is the RTX Video SDK super-resolution quality: `low`, `medium`, `high` (default), or `ultra`

Example:

```powershell
python -m sdr2hdr.cli input.mp4 output_hdr.mp4 `
  --preset portrait `
  --backend auto `
  --encoder libx265 `
  --x265-mode balanced `
  --model-path models\enhancement_model_reuse_v1.pt `
  --ai-strength 0.25 `
  --output-scale 2.0 `
  --scaler lanczos
```

Example using RTX Video SDK super resolution:

```powershell
python -m sdr2hdr.cli input.mp4 output_hdr_4k.mp4 `
  --model-path models\enhancement_model_reuse_v1.pt `
  --upscale-engine rtx-video `
  --output-scale 2.0 `
  --rtx-video-quality high
```

RTX Video SDK super resolution runs through `nvidia-vfx` (`pip install -e ".[rtx]"`), which wraps NVIDIA NGX. A single super-resolution session is loaded once and reused for the whole job, so NGX is never re-initialized per frame or per external process.

### Models

- The GUI scans the `models/` folder
- The location can be overridden with the `SDR2HDR_MODELS_DIR` environment variable
- Only `.pt` files are shown
- If no model is present, the GUI dropdown has no usable candidate
- The CLI requires an explicit `--model-path`

Recommended practice:

- Keep deployment models in `models/`
- Use filenames with dates or versions

### Notes

- This README is intentionally user-focused
- Internal parameters such as `peak nits` are not directly exposed in the GUI
- VFR (variable frame rate) input is normalized to CFR at the average frame rate
- Dithering is applied at PQ quantization to reduce banding
- `.onnx` and DirectML are not part of the current usage flow
- Running without an AI model is not the intended workflow

### License

This project is licensed under the MIT License.
