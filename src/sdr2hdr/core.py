from __future__ import annotations

from dataclasses import dataclass
import platform

import cv2
import numpy as np

from sdr2hdr.ai import BaseEnhancer, HeuristicEnhancer

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None

REC709_TO_REC2020 = np.array(
    [
        [0.6274040, 0.3292820, 0.0433136],
        [0.0690970, 0.9195400, 0.0113612],
        [0.0163916, 0.0880132, 0.8955950],
    ],
    dtype=np.float32,
)

_PQ_M1 = 2610.0 / 16384.0
_PQ_M2 = 2523.0 / 32.0
_PQ_C1 = 3424.0 / 4096.0
_PQ_C2 = 2413.0 / 128.0
_PQ_C3 = 2392.0 / 128.0


def srgb_to_linear(frame: np.ndarray) -> np.ndarray:
    frame = np.clip(frame, 0.0, 1.0)
    return np.where(frame <= 0.04045, frame / 12.92, ((frame + 0.055) / 1.055) ** 2.4)


def linear_to_pq(frame: np.ndarray, peak_nits: float) -> np.ndarray:
    normalized = np.clip(frame * peak_nits / 10000.0, 0.0, 1.0)
    numerator = _PQ_C1 + _PQ_C2 * np.power(normalized, _PQ_M1)
    denominator = 1.0 + _PQ_C3 * np.power(normalized, _PQ_M1)
    return np.power(numerator / denominator, _PQ_M2)


def apply_matrix(frame: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.clip(np.tensordot(frame, matrix.T, axes=1), 0.0, None)


def estimate_skin_mask(frame_linear: np.ndarray) -> np.ndarray:
    r = frame_linear[..., 0]
    g = frame_linear[..., 1]
    b = frame_linear[..., 2]
    sum_rgb = np.maximum(r + g + b, 1e-6)
    rn = r / sum_rgb
    gn = g / sum_rgb
    skin = (
        (rn > 0.36)
        & (rn < 0.55)
        & (gn > 0.25)
        & (gn < 0.40)
        & (r > b)
        & (g > b * 0.9)
    )
    return skin.astype(np.float32)


def bilateral_detail_boost(luma: np.ndarray, amount: float) -> np.ndarray:
    base = cv2.bilateralFilter(luma.astype(np.float32), d=0, sigmaColor=0.08, sigmaSpace=5.0)
    detail = luma - base
    return np.clip(luma + detail * amount, 0.0, 2.0)


def fast_detail_boost(luma: np.ndarray, amount: float) -> np.ndarray:
    base = cv2.GaussianBlur(luma.astype(np.float32), (0, 0), 1.2)
    detail = luma - base
    return np.clip(luma + detail * amount, 0.0, 2.0)


@dataclass
class ProcessorConfig:
    peak_nits: float = 1000.0
    ai_strength: float = 0.35
    detail_boost: float = 0.25
    scene_smoothing: float = 0.88
    scene_cut_threshold: float = 0.16
    highlight_boost: float = 1.2
    subtitle_protection: float = 0.85
    shadow_noise_floor: float = 0.075
    skin_protection: float = 0.55
    shadow_rolloff: float = 0.45
    processing_scale: float = 1.0
    fast_mode: bool = False
    backend: str = "auto"
    clipped_white_protection: float = 0.65
    near_white_rolloff_start: float = 0.78
    near_white_rolloff_strength: float = 0.6
    adaptive_highlight: bool = True
    adaptive_highlight_min: float = 0.55
    adaptive_highlight_max: float = 1.05


class TemporalState:
    def __init__(self) -> None:
        self.prev_luma_mean: float | None = None
        self.exposure_gain: float = 1.0
        self.prev_luma_small: np.ndarray | None = None
        self.prev_chroma_small: np.ndarray | None = None
        self.scene_highlight_boost: float | None = None

    def reset(self, luma_mean: float, luma_small: np.ndarray, chroma_small: np.ndarray) -> float:
        self.prev_luma_mean = luma_mean
        self.prev_luma_small = luma_small
        self.prev_chroma_small = chroma_small
        self.exposure_gain = np.clip(0.22 / max(luma_mean, 1e-4), 0.85, 1.35)
        self.scene_highlight_boost = None
        return self.exposure_gain

    def scene_change_score(self, luma_small: np.ndarray, chroma_small: np.ndarray) -> float:
        if self.prev_luma_small is None or self.prev_chroma_small is None:
            return 0.0
        luma_delta = float(np.mean(np.abs(luma_small - self.prev_luma_small)))
        chroma_delta = float(np.mean(np.abs(chroma_small - self.prev_chroma_small)))
        return luma_delta * 0.7 + chroma_delta * 0.3

    def update(
        self,
        luma_mean: float,
        smoothing: float,
        luma_small: np.ndarray,
        chroma_small: np.ndarray,
        scene_cut_threshold: float,
    ) -> tuple[float, bool]:
        target = np.clip(0.22 / max(luma_mean, 1e-4), 0.85, 1.35)
        scene_cut = self.scene_change_score(luma_small, chroma_small) >= scene_cut_threshold
        if self.prev_luma_mean is None or scene_cut:
            self.exposure_gain = target
        else:
            alpha = np.clip(smoothing, 0.0, 0.99)
            self.exposure_gain = self.exposure_gain * alpha + target * (1.0 - alpha)
        self.prev_luma_mean = luma_mean
        self.prev_luma_small = luma_small
        self.prev_chroma_small = chroma_small
        return self.exposure_gain, scene_cut


def compute_luma(frame_linear: np.ndarray) -> np.ndarray:
    return 0.2627 * frame_linear[..., 0] + 0.6780 * frame_linear[..., 1] + 0.0593 * frame_linear[..., 2]


def compute_chroma(frame_linear: np.ndarray) -> np.ndarray:
    max_rgb = np.max(frame_linear, axis=2)
    min_rgb = np.min(frame_linear, axis=2)
    return max_rgb - min_rgb


def downsample_map(image: np.ndarray, size: int = 32) -> np.ndarray:
    return cv2.resize(image.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def estimate_subtitle_mask(frame_bgr8: np.ndarray, luma: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    height, width = gray.shape
    lower_band = np.zeros_like(gray, dtype=np.float32)
    lower_band[int(height * 0.72) :, :] = 1.0
    bright = (gray > 0.82).astype(np.float32)
    low_chroma = (np.std(frame_bgr8.astype(np.float32) / 255.0, axis=2) < 0.08).astype(np.float32)
    local_contrast = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edge = (np.abs(local_contrast) > 0.12).astype(np.float32)
    mask = bright * low_chroma * edge * lower_band * (luma < 0.92).astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1)
    return dilated.astype(np.float32)


def estimate_subtitle_mask_fast(frame_bgr8: np.ndarray, luma: np.ndarray) -> np.ndarray:
    height, width = luma.shape
    band_start = int(height * 0.72)
    gray = cv2.cvtColor(frame_bgr8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_band = gray[band_start:, :]
    luma_band = luma[band_start:, :]
    rgb_band = frame_bgr8[band_start:, :, :].astype(np.float32) / 255.0
    small_width = max(64, width // 2)
    small_height = max(16, gray_band.shape[0] // 2)
    gray_small = cv2.resize(gray_band, (small_width, small_height), interpolation=cv2.INTER_AREA)
    luma_small = cv2.resize(luma_band, (small_width, small_height), interpolation=cv2.INTER_AREA)
    rgb_small = cv2.resize(rgb_band, (small_width, small_height), interpolation=cv2.INTER_AREA)
    bright = (gray_small > 0.82).astype(np.float32)
    low_chroma = (np.std(rgb_small, axis=2) < 0.08).astype(np.float32)
    edge = (np.abs(gray_small - cv2.GaussianBlur(gray_small, (0, 0), 0.8)) > 0.035).astype(np.float32)
    mask_small = bright * low_chroma * edge * (luma_small < 0.92).astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate((mask_small > 0).astype(np.uint8), kernel, iterations=1).astype(np.float32)
    mask_band = cv2.resize(dilated, (width, height - band_start), interpolation=cv2.INTER_LINEAR)
    mask = np.zeros_like(gray, dtype=np.float32)
    mask[band_start:, :] = np.clip(mask_band, 0.0, 1.0)
    return mask


def estimate_noise_mask(
    frame_or_luma: np.ndarray,
    luma_or_threshold: np.ndarray | float,
    threshold: float | None = None,
    blur: np.ndarray | None = None,
) -> np.ndarray:
    if threshold is None:
        luma = frame_or_luma
        threshold = float(luma_or_threshold)
    else:
        luma = luma_or_threshold
        if blur is None:
            blur = cv2.GaussianBlur(np.clip(compute_luma(frame_or_luma), 0.0, 1.0).astype(np.float32), (0, 0), 1.2)
    gray = np.clip(luma, 0.0, 1.0).astype(np.float32)
    blur = blur if blur is not None else cv2.GaussianBlur(gray, (0, 0), 1.2)
    residual = np.abs(gray - blur)
    noise = residual / np.maximum(gray + 0.05, 0.05)
    shadow = np.clip((threshold - luma) / max(threshold, 1e-4), 0.0, 1.0)
    return np.clip(noise * 6.0, 0.0, 1.0) * shadow


def estimate_specular_mask(frame_linear: np.ndarray, luma: np.ndarray) -> np.ndarray:
    chroma = compute_chroma(frame_linear)
    neutral = 1.0 - np.clip(chroma / np.maximum(luma, 1e-4), 0.0, 1.0)
    bright = np.clip((luma - 0.58) / 0.42, 0.0, 1.0)
    return np.clip(bright * (0.45 + 0.55 * neutral), 0.0, 1.0)


def estimate_sky_mask(frame_linear: np.ndarray) -> np.ndarray:
    r = frame_linear[..., 0]
    g = frame_linear[..., 1]
    b = frame_linear[..., 2]
    blue = (b > g) & (g > r)
    hue_strength = np.clip((b - r) * 2.4, 0.0, 1.0)
    vertical = np.linspace(1.0, 0.0, frame_linear.shape[0], dtype=np.float32)[:, None]
    return blue.astype(np.float32) * hue_strength * vertical


def estimate_clipped_white_mask(frame_linear: np.ndarray, luma: np.ndarray, detail_base: np.ndarray | None = None) -> np.ndarray:
    chroma = compute_chroma(frame_linear)
    detail_base = detail_base if detail_base is not None else cv2.GaussianBlur(np.clip(luma, 0.0, 1.0).astype(np.float32), (0, 0), 1.4)
    detail = np.abs(np.clip(luma, 0.0, 1.0) - detail_base)
    bright = np.clip((luma - 0.72) / 0.22, 0.0, 1.0)
    neutral = 1.0 - np.clip(chroma / np.maximum(luma, 1e-4), 0.0, 1.0)
    flat = 1.0 - np.clip(detail / 0.035, 0.0, 1.0)
    return np.clip(bright * (0.5 + 0.5 * neutral) * flat, 0.0, 1.0)


def apply_near_white_rolloff(luma: np.ndarray, start: float, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return np.ones_like(luma, dtype=np.float32)
    ramp = np.clip((luma - start) / max(1.0 - start, 1e-4), 0.0, 1.0)
    return 1.0 - ramp * strength


def compute_adaptive_highlight_boost(
    base_boost: float,
    clipped_white_ratio: float,
    skin_ratio: float,
    specular_ratio: float,
    sky_ratio: float,
    minimum: float,
    maximum: float,
) -> float:
    adjusted = base_boost
    adjusted -= clipped_white_ratio * 0.45
    adjusted -= skin_ratio * 0.18
    adjusted -= sky_ratio * 0.10
    adjusted += specular_ratio * 0.22
    return float(np.clip(adjusted, minimum, maximum))


class SDRToHDRProcessor:
    def __init__(self, config: ProcessorConfig, enhancer: BaseEnhancer | None = None) -> None:
        self.config = config
        self.enhancer = enhancer or HeuristicEnhancer()
        self.state = TemporalState()
        self.torch_device = self._resolve_torch_device()
        self._rec2020_t: torch.Tensor | None = None
        self._laplacian_kernel_t: torch.Tensor | None = None
        self._sky_gradient_t: torch.Tensor | None = None
        self._sky_gradient_height: int | None = None
        self._scaled_shape_cache: dict[tuple[int, int], tuple[int, int]] = {}
        if torch is not None and self.torch_device is not None:
            self._rec2020_t = torch.from_numpy(REC709_TO_REC2020).to(self.torch_device, dtype=torch.float32)
            self._laplacian_kernel_t = torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                device=self.torch_device,
                dtype=torch.float32,
            )

    def _resolve_torch_device(self) -> str | None:
        if torch is None:
            return None
        if self.config.backend == "numpy":
            return None
        if self.config.backend == "torch-cpu":
            return "cpu"
        if self.config.backend == "cuda":
            return "cuda" if torch.cuda.is_available() else None
        if self.config.backend == "mps":
            return "mps" if torch.backends.mps.is_available() else None
        if platform.system() == "Windows":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return None
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return None

    def _scene_highlight_boost(
        self,
        skin_ratio: float,
        specular_ratio: float,
        sky_ratio: float,
        clipped_white_ratio: float,
        scene_cut: bool,
    ) -> float:
        if not self.config.adaptive_highlight:
            return self.config.highlight_boost
        if self.state.scene_highlight_boost is None or scene_cut:
            self.state.scene_highlight_boost = compute_adaptive_highlight_boost(
                self.config.highlight_boost,
                clipped_white_ratio,
                skin_ratio,
                specular_ratio,
                sky_ratio,
                self.config.adaptive_highlight_min,
                self.config.adaptive_highlight_max,
            )
        return self.state.scene_highlight_boost

    def _tensor_from_rgb(self, frame_rgb: np.ndarray) -> torch.Tensor:
        assert torch is not None
        return torch.from_numpy(frame_rgb).to(self.torch_device, dtype=torch.float32)

    def _torch_compute_luma(self, frame_linear: torch.Tensor) -> torch.Tensor:
        return frame_linear[..., 0] * 0.2627 + frame_linear[..., 1] * 0.6780 + frame_linear[..., 2] * 0.0593

    def _torch_compute_chroma(self, frame_linear: torch.Tensor) -> torch.Tensor:
        return torch.amax(frame_linear, dim=2) - torch.amin(frame_linear, dim=2)

    def _torch_srgb_to_linear(self, frame: torch.Tensor) -> torch.Tensor:
        return torch.where(
            frame <= 0.04045,
            frame / 12.92,
            torch.pow((frame + 0.055) / 1.055, 2.4),
        )

    def _torch_linear_to_pq(self, frame: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp(frame * self.config.peak_nits / 10000.0, 0.0, 1.0)
        power = torch.pow(normalized, _PQ_M1)
        return torch.pow((_PQ_C1 + _PQ_C2 * power) / (1.0 + _PQ_C3 * power), _PQ_M2)

    def _torch_downsample(self, image: torch.Tensor, size: int = 32) -> torch.Tensor:
        assert F is not None
        mode = "area"
        if self.torch_device == "mps":
            height, width = image.shape[-2:]
            if height % size != 0 or width % size != 0:
                mode = "bilinear"
        kwargs = {"size": (size, size), "mode": mode}
        if mode != "area":
            kwargs["align_corners"] = False
        return F.interpolate(image[None, None], **kwargs).squeeze(0).squeeze(0)

    def _torch_blur(self, image: torch.Tensor, kernel: int = 5) -> torch.Tensor:
        assert F is not None
        padding = kernel // 2
        return F.avg_pool2d(image[None, None], kernel, stride=1, padding=padding).squeeze(0).squeeze(0)

    def _torch_resize_map(self, image: torch.Tensor, height: int, width: int) -> torch.Tensor:
        assert F is not None
        return F.interpolate(image[None, None], size=(height, width), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

    def _torch_laplacian(self, image: torch.Tensor) -> torch.Tensor:
        assert F is not None
        assert self._laplacian_kernel_t is not None
        return F.conv2d(image[None, None], self._laplacian_kernel_t[None, None], padding=1).squeeze(0).squeeze(0)

    def _torch_dilate(self, image: torch.Tensor, kernel: int = 3) -> torch.Tensor:
        assert F is not None
        padding = kernel // 2
        return F.max_pool2d(image[None, None], kernel, stride=1, padding=padding).squeeze(0).squeeze(0)

    def _torch_heuristic_maps(self, frame_linear_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        luminance = torch.clamp(self._torch_compute_luma(frame_linear_t), 0.0, 1.0)
        smooth = self._torch_blur(luminance, 11)
        detail = torch.clamp(luminance - smooth, -0.2, 0.2)
        expansion = torch.clamp((luminance - 0.55) / 0.45, 0.0, 1.0)
        local_contrast = torch.clamp(torch.abs(detail) * 6.0, 0.0, 1.0)
        chroma_spread = self._torch_compute_chroma(frame_linear_t)
        protection = torch.clamp(1.0 - chroma_spread * 1.2, 0.0, 1.0)
        return expansion, local_contrast, protection

    def _torch_clipped_white_mask(
        self,
        frame_linear_t: torch.Tensor,
        luma_unit: torch.Tensor,
        detail_base: torch.Tensor | None = None,
    ) -> torch.Tensor:
        chroma = self._torch_compute_chroma(frame_linear_t)
        detail_base = detail_base if detail_base is not None else self._torch_blur(torch.clamp(luma_unit, 0.0, 1.0), 5)
        detail = torch.abs(torch.clamp(luma_unit, 0.0, 1.0) - detail_base)
        bright = torch.clamp((luma_unit - 0.72) / 0.22, 0.0, 1.0)
        neutral = 1.0 - torch.clamp(chroma / torch.clamp(luma_unit, min=1e-4), 0.0, 1.0)
        flat = 1.0 - torch.clamp(detail / 0.035, 0.0, 1.0)
        return torch.clamp(bright * (0.5 + 0.5 * neutral) * flat, 0.0, 1.0)

    def _torch_near_white_rolloff(self, luma_unit: torch.Tensor) -> torch.Tensor:
        if self.config.near_white_rolloff_strength <= 0.0:
            return torch.ones_like(luma_unit)
        ramp = torch.clamp(
            (luma_unit - self.config.near_white_rolloff_start)
            / max(1.0 - self.config.near_white_rolloff_start, 1e-4),
            0.0,
            1.0,
        )
        return 1.0 - ramp * self.config.near_white_rolloff_strength

    def _torch_skin_mask(self, frame_linear_t: torch.Tensor) -> torch.Tensor:
        r = frame_linear_t[..., 0]
        g = frame_linear_t[..., 1]
        b = frame_linear_t[..., 2]
        sum_rgb = torch.clamp(r + g + b, min=1e-6)
        rn = r / sum_rgb
        gn = g / sum_rgb
        skin = (
            (rn > 0.36)
            & (rn < 0.55)
            & (gn > 0.25)
            & (gn < 0.40)
            & (r > b)
            & (g > b * 0.9)
        )
        return skin.to(torch.float32)

    def _torch_subtitle_mask(self, frame_rgb_t: torch.Tensor, luma_unit: torch.Tensor) -> torch.Tensor:
        gray = frame_rgb_t[..., 0] * 0.2990 + frame_rgb_t[..., 1] * 0.5870 + frame_rgb_t[..., 2] * 0.1140
        height, width = gray.shape
        lower_band = torch.zeros_like(gray)
        lower_band[int(height * 0.72) :, :] = 1.0
        bright = (gray > 0.82).to(torch.float32)
        low_chroma = (torch.std(frame_rgb_t, dim=2) < 0.08).to(torch.float32)
        edge = (torch.abs(self._torch_laplacian(gray)) > 0.12).to(torch.float32)
        mask = bright * low_chroma * edge * lower_band * (luma_unit < 0.92).to(torch.float32)
        return self._torch_dilate(mask, 3)

    def _torch_subtitle_mask_fast(self, frame_rgb_t: torch.Tensor, luma_unit: torch.Tensor) -> torch.Tensor:
        assert F is not None
        height, width = luma_unit.shape
        band_start = int(height * 0.72)
        gray = frame_rgb_t[..., 0] * 0.2990 + frame_rgb_t[..., 1] * 0.5870 + frame_rgb_t[..., 2] * 0.1140
        gray_band = gray[band_start:, :]
        luma_band = luma_unit[band_start:, :]
        rgb_band = frame_rgb_t[band_start:, :, :]
        small_width = max(64, width // 2)
        small_height = max(16, gray_band.shape[0] // 2)
        gray_small = F.interpolate(
            gray_band[None, None],
            size=(small_height, small_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        luma_small = F.interpolate(
            luma_band[None, None],
            size=(small_height, small_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        rgb_small = F.interpolate(
            rgb_band.permute(2, 0, 1)[None],
            size=(small_height, small_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
        bright = (gray_small > 0.82).to(torch.float32)
        low_chroma = (torch.std(rgb_small, dim=2) < 0.08).to(torch.float32)
        edge = (torch.abs(gray_small - self._torch_blur(gray_small, 3)) > 0.035).to(torch.float32)
        mask_small = bright * low_chroma * edge * (luma_small < 0.92).to(torch.float32)
        mask_small = self._torch_dilate(mask_small, 3)
        mask_band = F.interpolate(
            mask_small[None, None],
            size=(height - band_start, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        mask = torch.zeros_like(gray)
        mask[band_start:, :] = torch.clamp(mask_band, 0.0, 1.0)
        return mask

    def _get_scaled_shape(self, original_height: int, original_width: int) -> tuple[int, int]:
        cache_key = (original_height, original_width)
        if cache_key not in self._scaled_shape_cache:
            scaled_width = max(64, int(round(original_width * self.config.processing_scale)))
            scaled_height = max(64, int(round(original_height * self.config.processing_scale)))
            self._scaled_shape_cache[cache_key] = (scaled_height, scaled_width)
        return self._scaled_shape_cache[cache_key]

    def _get_sky_gradient(self, height: int) -> torch.Tensor:
        assert torch is not None
        if self._sky_gradient_t is None or self._sky_gradient_height != height:
            self._sky_gradient_t = torch.linspace(1.0, 0.0, height, device=self.torch_device)[:, None]
            self._sky_gradient_height = height
        return self._sky_gradient_t

    def _process_frame_torch(self, frame_bgr8: np.ndarray) -> np.ndarray:
        assert torch is not None
        original_height, original_width = frame_bgr8.shape[:2]
        if self.config.processing_scale < 0.999:
            scaled_height, scaled_width = self._get_scaled_shape(original_height, original_width)
            work_bgr8 = cv2.resize(frame_bgr8, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        else:
            work_bgr8 = frame_bgr8

        frame_rgb = work_bgr8[..., ::-1].astype(np.float32) / 255.0
        frame_rgb_t = self._tensor_from_rgb(frame_rgb)
        frame_linear_t = self._torch_srgb_to_linear(frame_rgb_t)
        luma_t = torch.clamp(self._torch_compute_luma(frame_linear_t), 0.0, 1.0)
        chroma_t = self._torch_compute_chroma(frame_linear_t)
        scene_maps_t = torch.stack([self._torch_downsample(luma_t), self._torch_downsample(chroma_t)]).detach()
        scene_maps = scene_maps_t.cpu().numpy()
        luma_small = scene_maps[0]
        chroma_small = scene_maps[1]
        exposure_mean = float(luma_t.mean().detach().cpu().numpy())
        exposure, scene_cut = self.state.update(
            exposure_mean,
            self.config.scene_smoothing,
            luma_small,
            chroma_small,
            self.config.scene_cut_threshold,
        )
        frame_linear_t = torch.clamp(frame_linear_t * exposure, 0.0, 2.0)

        if isinstance(self.enhancer, HeuristicEnhancer):
            maps_expansion, maps_contrast, maps_protection = self._torch_heuristic_maps(frame_linear_t)
            skin_mask = self._torch_skin_mask(frame_linear_t)
        else:
            frame_linear_np = frame_linear_t.detach().cpu().numpy()
            maps = self.enhancer.estimate(frame_linear_np)
            maps_expansion = torch.from_numpy(maps.expansion).to(self.torch_device, dtype=torch.float32)
            maps_contrast = torch.from_numpy(maps.contrast).to(self.torch_device, dtype=torch.float32)
            maps_protection = torch.from_numpy(maps.protection).to(self.torch_device, dtype=torch.float32)
            skin_mask = torch.from_numpy(estimate_skin_mask(frame_linear_np)).to(self.torch_device, dtype=torch.float32)
        luma_t = torch.clamp(self._torch_compute_luma(frame_linear_t), 0.0, 2.0)
        luma_unit = torch.clamp(luma_t, 0.0, 1.0)
        subtitle_mask = (
            self._torch_subtitle_mask_fast(frame_rgb_t, luma_unit)
            if self.config.fast_mode
            else self._torch_subtitle_mask(frame_rgb_t, luma_unit)
        )
        blur_t = self._torch_blur(torch.clamp(luma_unit, 0.0, 1.0), 5)
        residual_t = torch.abs(luma_unit - blur_t)
        noise_mask = torch.clamp(residual_t / torch.clamp(luma_unit + 0.05, min=0.05) * 6.0, 0.0, 1.0)
        noise_mask = noise_mask * torch.clamp(
            (self.config.shadow_noise_floor - luma_unit) / max(self.config.shadow_noise_floor, 1e-4),
            0.0,
            1.0,
        )
        chroma_t = self._torch_compute_chroma(frame_linear_t)
        neutral = 1.0 - torch.clamp(chroma_t / torch.clamp(luma_unit, min=1e-4), 0.0, 1.0)
        specular_mask = torch.clamp(torch.clamp((luma_unit - 0.58) / 0.42, 0.0, 1.0) * (0.45 + 0.55 * neutral), 0.0, 1.0)
        r = frame_linear_t[..., 0]
        g = frame_linear_t[..., 1]
        b = frame_linear_t[..., 2]
        blue = ((b > g) & (g > r)).to(torch.float32)
        hue_strength = torch.clamp((b - r) * 2.4, 0.0, 1.0)
        vertical = self._get_sky_gradient(frame_linear_t.shape[0])
        sky_mask = blue * hue_strength * vertical
        clipped_white_mask = self._torch_clipped_white_mask(frame_linear_t, luma_unit, detail_base=blur_t)
        highlight_mask = torch.clamp(specular_mask * 0.75 + sky_mask * 0.35 + maps_expansion * 0.25, 0.0, 1.0)
        highlight_mask = highlight_mask * (1.0 - clipped_white_mask * self.config.clipped_white_protection)
        rolloff = self._torch_near_white_rolloff(luma_unit)
        protected = torch.clamp(
            self.config.skin_protection * skin_mask
            + 0.25 * maps_protection
            + self.config.subtitle_protection * subtitle_mask
            + 0.35 * noise_mask,
            0.0,
            1.0,
        )
        protected = torch.clamp(protected + clipped_white_mask * self.config.clipped_white_protection, 0.0, 1.0)

        scene_mean_stats = torch.stack(
            [
                skin_mask.mean(),
                specular_mask.mean(),
                sky_mask.mean(),
                clipped_white_mask.mean(),
                noise_mask.mean(),
            ]
        ).detach().cpu().numpy()
        scene_boost = self._scene_highlight_boost(
            float(scene_mean_stats[0]),
            float(scene_mean_stats[1]),
            float(scene_mean_stats[2]),
            float(scene_mean_stats[3]),
            scene_cut,
        )
        base_boost = scene_boost * (0.75 if scene_cut else 1.0)
        expanded = torch.clamp(frame_linear_t * (1.0 + highlight_mask[..., None] * base_boost * rolloff[..., None]), 0.0, 4.0)
        ai_expanded = torch.clamp(
            frame_linear_t
            * (1.0 + maps_expansion[..., None] * (scene_boost + 0.4) * rolloff[..., None]),
            0.0,
            4.0,
        )
        blended = expanded * (1.0 - self.config.ai_strength) + ai_expanded * self.config.ai_strength
        noise_limited = (
            blended * (1.0 - noise_mask[..., None] * self.config.shadow_rolloff)
            + frame_linear_t * noise_mask[..., None] * self.config.shadow_rolloff
        )
        frame_linear_t = (
            noise_limited * (1.0 - protected[..., None] * self.config.shadow_rolloff)
            + frame_linear_t * protected[..., None] * self.config.shadow_rolloff
        )

        relit_luma_t = torch.clamp(self._torch_compute_luma(frame_linear_t), 0.0, 2.0)
        detail_base = self._torch_blur(relit_luma_t, 5)
        detail = relit_luma_t - detail_base
        boosted_luma = torch.clamp(
            relit_luma_t
            + detail
            * (
                self.config.detail_boost
                * (0.35 + 0.65 * maps_contrast)
                * (1.0 - 0.6 * float(scene_mean_stats[4]))
            ),
            0.0,
            2.0,
        )
        relight = boosted_luma / torch.clamp(relit_luma_t, min=1e-4)
        frame_linear_t = torch.clamp(frame_linear_t * relight[..., None], 0.0, 4.0)
        if work_bgr8.shape[:2] != (original_height, original_width):
            frame_linear_t = F.interpolate(
                frame_linear_t.permute(2, 0, 1)[None],
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)

        assert self._rec2020_t is not None
        frame_2020_t = torch.clamp(torch.matmul(frame_linear_t, self._rec2020_t.T), min=0.0)
        frame_pq_t = self._torch_linear_to_pq(frame_2020_t)
        return torch.clamp(torch.round(frame_pq_t * 65535.0), 0, 65535).to(torch.uint16).cpu().numpy()

    def process_frame(self, frame_bgr8: np.ndarray) -> np.ndarray:
        if self.torch_device is not None:
            return self._process_frame_torch(frame_bgr8)
        original_height, original_width = frame_bgr8.shape[:2]
        if self.config.processing_scale < 0.999:
            scaled_height, scaled_width = self._get_scaled_shape(original_height, original_width)
            work_bgr8 = cv2.resize(frame_bgr8, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        else:
            work_bgr8 = frame_bgr8

        frame_rgb = work_bgr8[..., ::-1].astype(np.float32) / 255.0
        frame_linear = srgb_to_linear(frame_rgb)
        luma = np.clip(compute_luma(frame_linear), 0.0, 1.0)
        chroma = compute_chroma(frame_linear)
        luma_small = downsample_map(luma)
        chroma_small = downsample_map(chroma)
        exposure, scene_cut = self.state.update(
            float(luma.mean()),
            self.config.scene_smoothing,
            luma_small,
            chroma_small,
            self.config.scene_cut_threshold,
        )
        frame_linear = np.clip(frame_linear * exposure, 0.0, 2.0)

        maps = self.enhancer.estimate(frame_linear)
        skin_mask = estimate_skin_mask(frame_linear)
        luma = np.clip(compute_luma(frame_linear), 0.0, 1.5)
        luma_unit = np.clip(luma, 0.0, 1.0)
        subtitle_mask = (
            estimate_subtitle_mask_fast(work_bgr8, luma_unit)
            if self.config.fast_mode
            else estimate_subtitle_mask(work_bgr8, luma_unit)
        )
        blur_luma = cv2.GaussianBlur(luma_unit.astype(np.float32), (0, 0), 1.2)
        noise_mask = estimate_noise_mask(luma_unit, self.config.shadow_noise_floor, blur=blur_luma)
        specular_mask = estimate_specular_mask(frame_linear, luma_unit)
        sky_mask = estimate_sky_mask(frame_linear)
        clipped_white_mask = estimate_clipped_white_mask(frame_linear, luma_unit, detail_base=blur_luma)
        highlight_mask = np.clip(specular_mask * 0.75 + sky_mask * 0.35 + maps.expansion * 0.25, 0.0, 1.0)
        highlight_mask = highlight_mask * (1.0 - clipped_white_mask * self.config.clipped_white_protection)
        rolloff = apply_near_white_rolloff(
            luma_unit,
            self.config.near_white_rolloff_start,
            self.config.near_white_rolloff_strength,
        )
        protected = np.clip(
            self.config.skin_protection * skin_mask
            + 0.25 * maps.protection
            + self.config.subtitle_protection * subtitle_mask
            + 0.35 * noise_mask,
            0.0,
            1.0,
        )
        protected = np.clip(protected + clipped_white_mask * self.config.clipped_white_protection, 0.0, 1.0)

        scene_boost = self._scene_highlight_boost(
            float(np.mean(skin_mask)),
            float(np.mean(specular_mask)),
            float(np.mean(sky_mask)),
            float(np.mean(clipped_white_mask)),
            scene_cut,
        )
        base_boost = scene_boost * (0.75 if scene_cut else 1.0)
        expanded = np.clip(frame_linear * (1.0 + highlight_mask[..., None] * base_boost * rolloff[..., None]), 0.0, 4.0)
        ai_expanded = np.clip(
            frame_linear
            * (1.0 + maps.expansion[..., None] * (scene_boost + 0.4) * rolloff[..., None]),
            0.0,
            4.0,
        )
        blended = expanded * (1.0 - self.config.ai_strength) + ai_expanded * self.config.ai_strength
        noise_limited = (
            blended * (1.0 - noise_mask[..., None] * self.config.shadow_rolloff)
            + frame_linear * noise_mask[..., None] * self.config.shadow_rolloff
        )
        frame_linear = (
            noise_limited * (1.0 - protected[..., None] * self.config.shadow_rolloff)
            + frame_linear * protected[..., None] * self.config.shadow_rolloff
        )

        relit_luma = np.clip(compute_luma(frame_linear), 0.0, 2.0)
        detail_fn = fast_detail_boost if self.config.fast_mode else bilateral_detail_boost
        boosted_luma = detail_fn(
            relit_luma,
            (self.config.detail_boost * (0.35 + 0.65 * maps.contrast)) * (1.0 - 0.6 * float(np.mean(noise_mask))),
        )
        relight = boosted_luma / np.maximum(relit_luma, 1e-4)
        frame_linear = np.clip(frame_linear * relight[..., None], 0.0, 4.0)
        if work_bgr8.shape[:2] != (original_height, original_width):
            frame_linear = cv2.resize(frame_linear, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        frame_2020 = apply_matrix(frame_linear, REC709_TO_REC2020)
        frame_pq = linear_to_pq(frame_2020, self.config.peak_nits)
        frame_16 = np.clip(np.round(frame_pq * 65535.0), 0, 65535).astype(np.uint16)
        return frame_16
