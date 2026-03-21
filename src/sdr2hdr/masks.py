"""Mask estimation functions for protective HDR processing.

These masks identify regions (skin, subtitles, noise, etc.) where AI
enhancement should be suppressed to avoid artifacts.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_luma(frame_linear: np.ndarray) -> np.ndarray:
    from sdr2hdr.constants import LUMA_R, LUMA_G, LUMA_B

    return LUMA_R * frame_linear[..., 0] + LUMA_G * frame_linear[..., 1] + LUMA_B * frame_linear[..., 2]


def compute_chroma(frame_linear: np.ndarray) -> np.ndarray:
    max_rgb = np.max(frame_linear, axis=2)
    min_rgb = np.min(frame_linear, axis=2)
    return max_rgb - min_rgb


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


def estimate_high_chroma_mask(frame_linear: np.ndarray, luma: np.ndarray) -> np.ndarray:
    chroma = compute_chroma(frame_linear)
    saturation = chroma / np.maximum(np.max(frame_linear, axis=2), 1e-4)
    midtone = 1.0 - np.clip(np.abs(luma - 0.45) / 0.45, 0.0, 1.0)
    vivid = np.clip((saturation - 0.18) / 0.42, 0.0, 1.0)
    return np.clip(vivid * (0.35 + 0.65 * midtone), 0.0, 1.0)


def estimate_memory_color_mask(frame_linear: np.ndarray, luma: np.ndarray) -> np.ndarray:
    r = frame_linear[..., 0]
    g = frame_linear[..., 1]
    b = frame_linear[..., 2]
    sum_rgb = np.maximum(r + g + b, 1e-6)
    rn = r / sum_rgb
    gn = g / sum_rgb
    bn = b / sum_rgb
    foliage = ((gn > rn) & (gn > bn) & (gn > 0.34)).astype(np.float32) * np.clip((gn - bn) * 3.2, 0.0, 1.0)
    warm = ((rn > gn) & (gn > bn) & (rn > 0.38)).astype(np.float32) * np.clip((rn - bn) * 2.8, 0.0, 1.0)
    cyan_blue = ((bn > rn) & (bn > 0.32)).astype(np.float32) * np.clip((bn - rn) * 3.0, 0.0, 1.0)
    valid_luma = np.clip((luma - 0.08) / 0.22, 0.0, 1.0) * (1.0 - np.clip((luma - 0.92) / 0.08, 0.0, 1.0))
    return np.clip(np.maximum.reduce([foliage, warm, cyan_blue]) * valid_luma, 0.0, 1.0)


def estimate_clipped_white_mask(frame_linear: np.ndarray, luma: np.ndarray, detail_base: np.ndarray | None = None) -> np.ndarray:
    chroma = compute_chroma(frame_linear)
    detail_base = detail_base if detail_base is not None else cv2.GaussianBlur(np.clip(luma, 0.0, 1.0).astype(np.float32), (0, 0), 1.4)
    detail = np.abs(np.clip(luma, 0.0, 1.0) - detail_base)
    bright = np.clip((luma - 0.72) / 0.22, 0.0, 1.0)
    neutral = 1.0 - np.clip(chroma / np.maximum(luma, 1e-4), 0.0, 1.0)
    flat = 1.0 - np.clip(detail / 0.035, 0.0, 1.0)
    return np.clip(bright * (0.5 + 0.5 * neutral) * flat, 0.0, 1.0)


def build_ai_gate(
    skin_mask: np.ndarray,
    subtitle_mask: np.ndarray,
    noise_mask: np.ndarray,
    clipped_white_mask: np.ndarray,
    high_chroma_mask: np.ndarray,
    memory_color_mask: np.ndarray,
    learned_protection: np.ndarray,
) -> np.ndarray:
    suppression = np.maximum.reduce(
        [
            skin_mask * 0.95,
            subtitle_mask,
            noise_mask * 0.65,
            clipped_white_mask * 0.95,
            high_chroma_mask * 0.75,
            memory_color_mask * 0.85,
            learned_protection * 0.60,
        ]
    )
    return np.clip(1.0 - suppression, 0.0, 1.0)


def apply_near_white_rolloff(luma: np.ndarray, start: float, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return np.ones_like(luma, dtype=np.float32)
    ramp = np.clip((luma - start) / max(1.0 - start, 1e-4), 0.0, 1.0)
    return 1.0 - ramp * strength


def limit_ai_highlight_expansion(
    expansion: np.ndarray,
    luma: np.ndarray,
    clipped_white_mask: np.ndarray,
    rolloff: np.ndarray,
) -> np.ndarray:
    near_white = np.clip((luma - 0.7) / 0.25, 0.0, 1.0)
    suppression = np.clip(clipped_white_mask * 0.85 + near_white * (1.0 - rolloff) * 0.75, 0.0, 0.95)
    return np.clip(expansion * (1.0 - suppression), 0.0, 1.0)


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
