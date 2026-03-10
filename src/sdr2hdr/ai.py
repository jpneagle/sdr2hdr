from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - optional dependency
    ort = None


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


def directml_is_available() -> bool:
    if ort is None:
        return False
    try:
        return "DmlExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


class HeuristicEnhancer(BaseEnhancer):
    """Fallback enhancer that approximates highlight/detail recovery maps."""

    def estimate(self, frame_linear: np.ndarray) -> EnhancementMaps:
        return estimate_heuristic_maps(frame_linear)


class TorchMapEnhancer(BaseEnhancer):
    """Optional torch-backed enhancer placeholder.

    The model is expected to return 3 channels: expansion, contrast, protection.
    """

    def __init__(self, model_path: str, device: str = "cpu", inference_scale: float = 0.5) -> None:
        if torch is None:  # pragma: no cover - optional dependency
            raise RuntimeError("torch is not installed; install sdr2hdr[ai] to use model inference")
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.inference_scale = float(np.clip(inference_scale, 0.1, 1.0))

    def _target_size(self, height: int, width: int) -> tuple[int, int]:
        if self.inference_scale >= 0.999:
            return height, width
        return (
            max(64, int(round(height * self.inference_scale))),
            max(64, int(round(width * self.inference_scale))),
        )

    def _heuristic_maps_torch(self, frame_linear_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        luminance = torch.clamp(
            frame_linear_t[..., 0] * 0.2627 + frame_linear_t[..., 1] * 0.6780 + frame_linear_t[..., 2] * 0.0593,
            0.0,
            1.0,
        )
        smooth = F.avg_pool2d(luminance[None, None], 11, stride=1, padding=5).squeeze(0).squeeze(0)
        detail = torch.clamp(luminance - smooth, -0.2, 0.2)
        expansion = torch.clamp((luminance - 0.55) / 0.45, 0.0, 1.0)
        local_contrast = torch.clamp(torch.abs(detail) * 6.0, 0.0, 1.0)
        chroma_spread = torch.amax(frame_linear_t, dim=2) - torch.amin(frame_linear_t, dim=2)
        protection = torch.clamp(1.0 - chroma_spread * 1.2, 0.0, 1.0)
        return expansion, local_contrast, protection

    def _run_model(self, tensor: torch.Tensor) -> torch.Tensor:
        assert F is not None
        _, _, height, width = tensor.shape
        target_height, target_width = self._target_size(height, width)
        if (target_height, target_width) != (height, width):
            tensor = F.interpolate(
                tensor,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
        with torch.inference_mode():
            output = self.model(tensor)
        if tuple(output.shape[-2:]) != (height, width):
            output = F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
        return output

    def estimate_torch(self, frame_linear_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tensor = frame_linear_t.permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)
        output = self._run_model(tensor).squeeze(0)
        base_expansion, base_contrast, base_protection = self._heuristic_maps_torch(
            frame_linear_t.to(self.device, dtype=torch.float32)
        )
        expansion = torch.clamp(base_expansion + output[0], 0.0, 1.0)
        contrast = torch.clamp(base_contrast + output[1], 0.0, 1.0)
        protection = torch.clamp(base_protection + output[2], 0.0, 1.0)
        return expansion, contrast, protection

    def estimate(self, frame_linear: np.ndarray) -> EnhancementMaps:
        tensor = torch.from_numpy(frame_linear).to(self.device, dtype=torch.float32)
        expansion_t, contrast_t, protection_t = self.estimate_torch(tensor)
        return EnhancementMaps(
            expansion=expansion_t.cpu().numpy(),
            contrast=contrast_t.cpu().numpy(),
            protection=protection_t.cpu().numpy(),
        )


class OnnxDmlMapEnhancer(BaseEnhancer):
    """ONNX Runtime DirectML-backed enhancer for Windows fallback inference."""

    def __init__(self, model_path: str, inference_scale: float = 0.5) -> None:
        if ort is None:  # pragma: no cover - optional dependency
            raise RuntimeError("onnxruntime-directml is not installed; install sdr2hdr[directml] to use DirectML")
        if not directml_is_available():
            raise RuntimeError("DirectML execution provider is unavailable in this environment")
        self.session = ort.InferenceSession(
            model_path,
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.inference_scale = float(np.clip(inference_scale, 0.1, 1.0))

    def _target_size(self, height: int, width: int) -> tuple[int, int]:
        if self.inference_scale >= 0.999:
            return height, width
        return (
            max(64, int(round(height * self.inference_scale))),
            max(64, int(round(width * self.inference_scale))),
        )

    def _run_model(self, frame_linear: np.ndarray) -> np.ndarray:
        height, width = frame_linear.shape[:2]
        target_height, target_width = self._target_size(height, width)
        model_input = frame_linear
        if (target_height, target_width) != (height, width):
            model_input = cv2.resize(frame_linear, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        tensor = model_input.transpose(2, 0, 1)[None].astype(np.float32, copy=False)
        output = self.session.run(None, {self.input_name: tensor})[0].squeeze(0)
        if tuple(output.shape[-2:]) != (height, width):
            output = np.stack(
                [
                    cv2.resize(output[channel], (width, height), interpolation=cv2.INTER_LINEAR)
                    for channel in range(output.shape[0])
                ],
                axis=0,
            )
        return output

    def estimate(self, frame_linear: np.ndarray) -> EnhancementMaps:
        output = self._run_model(frame_linear)
        base_maps = estimate_heuristic_maps(frame_linear)
        expansion = np.clip(base_maps.expansion + output[0], 0.0, 1.0)
        contrast = np.clip(base_maps.contrast + output[1], 0.0, 1.0)
        protection = np.clip(base_maps.protection + output[2], 0.0, 1.0)
        return EnhancementMaps(expansion=expansion, contrast=contrast, protection=protection)
