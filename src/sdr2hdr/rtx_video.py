from __future__ import annotations

import numpy as np
import torch

RTX_VIDEO_QUALITY_OPTIONS = ("low", "medium", "high", "ultra")


def ensure_rtx_video_available() -> None:
    try:
        import nvvfx  # noqa: F401
    except ImportError as exc:
        raise ValueError(
            "RTX Video SDK is not installed. Run 'pip install nvidia-vfx' on a machine with an RTX GPU."
        ) from exc
    if not torch.cuda.is_available():
        raise ValueError("RTX Video SDK requires a CUDA-capable NVIDIA GPU.")


class RtxVideoUpscaler:
    """Wraps a single nvvfx.VideoSuperRes session for an entire conversion job.

    NGX bootstraps its feature models on load(). Recreating the session per
    frame or per chunk (as an external-CLI-per-chunk design would) re-triggers
    that bootstrap every time, which is what caused nvngx_update.exe to spawn
    without bound. Loading once and reusing it via run() for every frame keeps
    NGX initialization to a single call for the whole job.
    """

    def __init__(
        self,
        output_width: int,
        output_height: int,
        quality: str = "high",
        device: int = 0,
    ) -> None:
        ensure_rtx_video_available()
        import nvvfx
        from nvvfx.effects import QualityLevel

        quality_map = {
            "low": QualityLevel.LOW,
            "medium": QualityLevel.MEDIUM,
            "high": QualityLevel.HIGH,
            "ultra": QualityLevel.ULTRA,
        }
        if quality not in quality_map:
            raise ValueError(f"Unknown RTX Video quality: {quality}")
        self._sr = nvvfx.VideoSuperRes(quality_map[quality], device=device)
        self._sr.output_width = output_width
        self._sr.output_height = output_height
        self._sr.load()
        self._pinned_u8: torch.Tensor | None = None
        self._pinned_u8_shape: tuple[int, ...] | None = None

    def upscale_tensor(self, frame_bgr8: np.ndarray) -> torch.Tensor:
        """Run SR and return the result as an HWC RGB float32 [0,1] tensor that
        stays on the CUDA device, so the caller can hand it straight to
        SDRToHDRProcessor.process_frame_tensor without a GPU->CPU->GPU roundtrip.
        """
        # Upload as uint8 (4x less PCIe traffic than float32) and do the
        # BGR->RGB flip, float conversion, and /255 normalization on the GPU
        # instead of on the CPU at input resolution.
        shape = frame_bgr8.shape
        if self._pinned_u8 is None or self._pinned_u8_shape != shape:
            self._pinned_u8 = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
            self._pinned_u8_shape = shape
        # Write via the pinned buffer's own numpy view rather than
        # torch.from_numpy(frame_bgr8): the decoder hands back frames from a
        # subprocess pipe buffer that may be non-writable, which torch warns
        # about (harmlessly, since we only read from it) if wrapped directly.
        np.copyto(self._pinned_u8.numpy(), frame_bgr8)
        bgr_u8 = self._pinned_u8.to("cuda", non_blocking=True)
        rgb_u8 = bgr_u8.flip(-1)  # BGR -> RGB, still HWC uint8
        # permute() only changes strides; the SDK requires a genuinely
        # contiguous CHW buffer, hence the explicit .contiguous() here.
        chw = rgb_u8.permute(2, 0, 1).float().div_(255.0).contiguous()
        with torch.inference_mode():
            output = torch.from_dlpack(self._sr.run(chw).image).clone()
        return output.clamp(0.0, 1.0).permute(1, 2, 0)

    def upscale(self, frame_bgr8: np.ndarray) -> np.ndarray:
        rgb_float_t = self.upscale_tensor(frame_bgr8)
        upscaled = rgb_float_t.mul(255.0).round().to(torch.uint8).cpu().numpy()
        return np.ascontiguousarray(upscaled[..., ::-1])

    def close(self) -> None:
        self._sr.close()

    def __enter__(self) -> "RtxVideoUpscaler":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        self.close()
        return False
