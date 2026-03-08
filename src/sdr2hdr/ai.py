from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


@dataclass
class EnhancementMaps:
    expansion: np.ndarray
    contrast: np.ndarray
    protection: np.ndarray


def estimate_heuristic_maps(frame_linear: np.ndarray) -> EnhancementMaps:
    luminance = np.clip(
        0.2627 * frame_linear[..., 0]
        + 0.6780 * frame_linear[..., 1]
        + 0.0593 * frame_linear[..., 2],
        0.0,
        1.0,
    )
    smooth = cv2.GaussianBlur(luminance, (0, 0), 5.0)
    detail = np.clip(luminance - smooth, -0.2, 0.2)
    expansion = np.clip((luminance - 0.55) / 0.45, 0.0, 1.0)
    local_contrast = np.clip(np.abs(detail) * 6.0, 0.0, 1.0)
    chroma_spread = np.max(frame_linear, axis=2) - np.min(frame_linear, axis=2)
    protection = np.clip(1.0 - chroma_spread * 1.2, 0.0, 1.0)
    return EnhancementMaps(expansion=expansion, contrast=local_contrast, protection=protection)


class BaseEnhancer:
    def estimate(self, frame_linear: np.ndarray) -> EnhancementMaps:
        raise NotImplementedError


class HeuristicEnhancer(BaseEnhancer):
    """Fallback enhancer that approximates highlight/detail recovery maps."""

    def estimate(self, frame_linear: np.ndarray) -> EnhancementMaps:
        return estimate_heuristic_maps(frame_linear)


class TorchMapEnhancer(BaseEnhancer):
    """Optional torch-backed enhancer placeholder.

    The model is expected to return 3 channels: expansion, contrast, protection.
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        if torch is None:  # pragma: no cover - optional dependency
            raise RuntimeError("torch is not installed; install sdr2hdr[ai] to use model inference")
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def estimate(self, frame_linear: np.ndarray) -> EnhancementMaps:
        base = estimate_heuristic_maps(frame_linear)
        tensor = (
            torch.from_numpy(frame_linear.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device, dtype=torch.float32)
        )
        with torch.no_grad():
            output = self.model(tensor).squeeze(0).cpu().numpy()
        expansion = np.clip(base.expansion + output[0], 0.0, 1.0)
        contrast = np.clip(base.contrast + output[1], 0.0, 1.0)
        protection = np.clip(base.protection + output[2], 0.0, 1.0)
        return EnhancementMaps(
            expansion=expansion,
            contrast=contrast,
            protection=protection,
        )
