# HDR Training Plan

## Goal

Retrain the enhancement-map model to reduce breakage in protected color regions while keeping practical HDR lift for live-action SDR-to-HDR conversion.

## Source Selection

Use for training:

- `E:\hdr_sample\HDR10-LG-Cymatic-Jazz-(www.demolandia.net).mp4`
- `E:\hdr_sample\HDR10-Sony-Bravia-OLED-(www.demolandia.net).mp4`
- `E:\hdr_sample\videoplayback (1).webm`
- `E:\hdr_sample\videoplayback.webm`

Hold out for later evaluation:

- `E:\hdr_sample\videoplayback (2).webm`

Exclude for the first pass:

- `E:\hdr_sample\HDR10-Life-Untouched-(www.demolandia.net).mp4`
- `E:\hdr_sample\videoplayback (3).webm`

Reason:

- `Life Untouched` shows a persistent logo in sampled frames.
- `videoplayback (3)` shows a persistent source watermark in sampled frames.
- `videoplayback (2)` is useful as a validation video outside the training mix.

## Storage Strategy

- Start with `sample_every=120`
- Use trimmed clips instead of full-video extraction for the long webm sources
- Write generated `.npz` files to `E:\hdr_sample\sdr2hdr_train\train_npz`

This keeps the first pass practical on the current free space on `E:`.

## Training Steps

1. Trim and stage selected source clips.
2. Generate SDR/HDR pair `.npz` files with `scripts/prepare_data.py`.
3. Train with `scripts/train.py`.
4. Export TorchScript with `scripts/export_model.py`.
5. Compare the new model against the current one on the held-out video.

## First-Pass Settings

- `sample_every=120`
- `epochs=50`
- `batch_size=8`
- `patch_size=256`
- `device=cuda`

## Follow-Up

If the first model is too conservative:

- keep the same source set
- lower `sample_every` only for `LG Cymatic Jazz` and one clean person video
- do not add the watermark sources unless they are cropped or filtered first
