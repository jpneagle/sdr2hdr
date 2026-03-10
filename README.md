# sdr2hdr

SDR 動画を HDR10 向けに変換する Python ツールです。GUI と CLI の両方を備えています。

現行版は `AI model 前提` の運用です。変換時には学習済み TorchScript モデル (`.pt`) を指定するか、GUI の `models/` プルダウンから選択する必要があります。

## Overview

- 入力は通常の SDR 動画です。
- 出力は HDR10 メタデータ付きの動画です。
- GUI は queue 実行に対応しています。
- AI モデルは `models/` フォルダ内の `.pt` ファイルを使用します。

## Requirements

- Python
- `ffmpeg` と `ffprobe` が実行可能であること
- PyTorch を含む依存関係

OS ごとの backend は次の通りです。

- Windows: `Auto`, `CUDA`, `CPU / NumPy`
- macOS: `Auto`, `MPS`, `CPU / NumPy`
- その他: `Auto`, `CPU / NumPy`

`Auto` は使える環境で GPU backend を優先し、使えない場合は CPU 側へ寄せます。

## Setup

依存関係をインストールし、学習済みモデルを `models/` に置きます。

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

モデル配置例:

```text
models/
  enhancement_model_20260310.pt
```

## Quick Start

### GUI

```powershell
python -m sdr2hdr.gui
```

GUI の基本動作:

- `Input` と `Output` を指定
- `Preset` は既定で `portrait`
- `AI Model` で `models/` 内の `.pt` を選択
- `AI Strength` は既定で `0.25`
- `Add To Queue` または `Add Files` で queue へ追加
- `Start Queue` で順次変換

### CLI

```powershell
python -m sdr2hdr.cli input.mp4 output_hdr.mp4 --model-path models\enhancement_model_20260310.pt
```

`output_path` を省略した場合は、入力ファイル名の末尾に `_hdr` を付けた名前が自動生成されます。

```powershell
python -m sdr2hdr.cli input.mp4 --model-path models\enhancement_model_20260310.pt
```

## GUI

### Main Controls

- `Preset`
  - 既定値は `portrait`
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

### Queue

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

### Queue Status

Queue の status 表示は現在次の 7 種類です。

- `QUEUED`
- `STARTING`
- `RUNNING`
- `CANCELLING`
- `OK`
- `FAILED`
- `CANCELLED`

`Stop Current` を押した場合は、まず `CANCELLING` になり、終了時に `CANCELLED` へ確定します。

### Cancel Behavior

- キャンセル時は partial output を保持する前提です。
- GUI の進捗欄には `partial output saved` と表示されます。

## CLI

現行 CLI の基本仕様:

- `input_path` は必須
- `output_path` は省略可能
- `--model-path` は必須
- `--model-path` は `.pt` モデルを指定
- `--preset` の既定値は `portrait`
- `--backend` は `auto`, `numpy`, `cuda`, `mps`
- `--ai-strength` の既定値は `0.25`

例:

```powershell
python -m sdr2hdr.cli input.mp4 output_hdr.mp4 `
  --preset portrait `
  --backend auto `
  --encoder libx265 `
  --x265-mode balanced `
  --model-path models\enhancement_model_20260310.pt `
  --ai-strength 0.25
```

## Models

- GUI は `models/` フォルダを参照します。
- 読み込むのは `.pt` ファイルのみです。
- モデル未配置時は GUI のプルダウンに有効候補が出ません。
- CLI では `--model-path` に明示指定します。

推奨:

- 配布用・運用用モデルは `models/` にまとめる
- ファイル名で日付やバージョンを区別する

## Notes

- 現行 README は `利用者向け` の内容に絞っています。
- `peak nits` などの内部パラメータは GUI からは直接設定できません。
- `.onnx` や DirectML は現行の利用手順には含めていません。
- AI モデルなしでの運用は前提にしていません。

## License

This project is licensed under the MIT License.
