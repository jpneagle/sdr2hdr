from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import cv2
import numpy as np

from sdr2hdr.ai import estimate_heuristic_maps
from sdr2hdr.core import (
    estimate_clipped_white_mask,
    estimate_noise_mask,
    estimate_skin_mask,
    estimate_subtitle_mask,
)

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    Dataset = object  # type: ignore[assignment]


@dataclass
class TargetMaps:
    expansion: np.ndarray
    contrast: np.ndarray
    protection: np.ndarray
    clip_mask: np.ndarray


def _compute_luma(frame: np.ndarray) -> np.ndarray:
    return 0.2627 * frame[..., 0] + 0.6780 * frame[..., 1] + 0.0593 * frame[..., 2]


def _compute_chroma(frame: np.ndarray) -> np.ndarray:
    return np.max(frame, axis=2) - np.min(frame, axis=2)


def linear_to_srgb(frame: np.ndarray) -> np.ndarray:
    frame = np.clip(frame, 0.0, 1.0)
    return np.where(frame <= 0.0031308, frame * 12.92, 1.055 * np.power(frame, 1.0 / 2.4) - 0.055)


def derive_target_maps(sdr_linear: np.ndarray, hdr_linear: np.ndarray) -> TargetMaps:
    base_maps = estimate_heuristic_maps(sdr_linear)
    luma_sdr = np.clip(_compute_luma(sdr_linear), 0.0, 1.0)
    luma_hdr = np.clip(_compute_luma(hdr_linear), 0.0, 1.5)
    expansion_abs = np.clip((luma_hdr - luma_sdr) / np.maximum(1.0 - luma_sdr, 1e-4), 0.0, 1.0)

    hdr_luma_unit = np.clip(luma_hdr, 0.0, 1.0).astype(np.float32)
    contrast_abs = np.clip(np.abs(hdr_luma_unit - cv2.GaussianBlur(hdr_luma_unit, (0, 0), 5.0)) * 6.0, 0.0, 1.0)

    clip_mask = (luma_sdr > 0.95).astype(np.float32)
    chroma_spread_hdr = _compute_chroma(hdr_linear)
    neutral_protection = np.clip(1.0 - chroma_spread_hdr * 1.2, 0.0, 1.0)
    skin_target = estimate_skin_mask(hdr_linear)
    clipped_white = estimate_clipped_white_mask(hdr_linear, np.clip(luma_hdr, 0.0, 1.0))
    noise_target = estimate_noise_mask(np.clip(luma_sdr, 0.0, 1.0), 0.08)
    sdr_rgb8 = np.clip(np.round(linear_to_srgb(sdr_linear) * 255.0), 0, 255).astype(np.uint8)
    subtitle_target = estimate_subtitle_mask(sdr_rgb8[..., ::-1], np.clip(luma_sdr, 0.0, 1.0))
    protection_abs = np.maximum.reduce(
        [
            neutral_protection,
            skin_target,
            subtitle_target,
            noise_target,
            clipped_white * 0.9,
        ]
    ).astype(np.float32)
    expansion = np.clip(expansion_abs - base_maps.expansion, -1.0, 1.0)
    contrast = np.clip(contrast_abs - base_maps.contrast, -1.0, 1.0)
    protection = np.clip(protection_abs - base_maps.protection, -1.0, 1.0)
    return TargetMaps(expansion=expansion, contrast=contrast, protection=protection, clip_mask=clip_mask)


def random_crop_pair(
    sdr_linear: np.ndarray,
    hdr_linear: np.ndarray,
    crop_size: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = sdr_linear.shape[:2]
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    top = 0 if height == crop_h else rng.randint(0, height - crop_h)
    left = 0 if width == crop_w else rng.randint(0, width - crop_w)
    return (
        sdr_linear[top : top + crop_h, left : left + crop_w],
        hdr_linear[top : top + crop_h, left : left + crop_w],
    )


def center_crop_pair(
    sdr_linear: np.ndarray,
    hdr_linear: np.ndarray,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = sdr_linear.shape[:2]
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    top = max((height - crop_h) // 2, 0)
    left = max((width - crop_w) // 2, 0)
    return (
        sdr_linear[top : top + crop_h, left : left + crop_w],
        hdr_linear[top : top + crop_h, left : left + crop_w],
    )


def augment_sdr(sdr_linear: np.ndarray, rng: random.Random) -> np.ndarray:
    augmented = np.clip(sdr_linear, 0.0, 1.35).astype(np.float32, copy=True)
    if rng.random() < 0.3:
        threshold = rng.uniform(0.75, 1.0)
        luma = _compute_luma(augmented)
        clip_region = luma > threshold
        augmented[clip_region] *= threshold / np.maximum(luma[clip_region, None], 1e-4)
    gamma = rng.uniform(0.9, 1.1)
    augmented = np.power(np.clip(augmented, 0.0, 1.0), gamma)
    if rng.random() < 0.3:
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0.0, 0.01, size=augmented.shape).astype(np.float32)
        augmented = np.clip(augmented + noise, 0.0, 1.0)
    return augmented


class HDRSDRPairDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        patch_size: int = 256,
        training: bool = True,
        seed: int = 0,
    ) -> None:
        if torch is None:  # pragma: no cover - optional dependency
            raise RuntimeError("torch is required to use HDRSDRPairDataset")
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.training = training
        self.paths = sorted(self.data_dir.glob("*.npz"))
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = np.load(self.paths[index])
        sdr_linear = sample["sdr_linear"].astype(np.float32)
        hdr_linear = sample["hdr_linear"].astype(np.float32)
        if self.training:
            sdr_linear, hdr_linear = random_crop_pair(sdr_linear, hdr_linear, self.patch_size, self.rng)
            if self.rng.random() < 0.5:
                sdr_linear = np.ascontiguousarray(sdr_linear[:, ::-1])
                hdr_linear = np.ascontiguousarray(hdr_linear[:, ::-1])
            sdr_linear = augment_sdr(sdr_linear, self.rng)
        elif self.patch_size > 0:
            sdr_linear, hdr_linear = center_crop_pair(sdr_linear, hdr_linear, self.patch_size)
        targets = derive_target_maps(sdr_linear, hdr_linear)
        return {
            "sdr_linear": torch.from_numpy(sdr_linear.transpose(2, 0, 1)).to(torch.float32),
            "hdr_linear": torch.from_numpy(hdr_linear.transpose(2, 0, 1)).to(torch.float32),
            "target_maps": torch.from_numpy(
                np.stack([targets.expansion, targets.contrast, targets.protection], axis=0)
            ).to(torch.float32),
            "clip_mask": torch.from_numpy(targets.clip_mask[None]).to(torch.float32),
        }
