from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch

INTEL_SR_DEVICE_OPTIONS = ("AUTO", "CPU", "GPU")
INTEL_SR_MODEL_OPTIONS = ("sr-1032", "sr-1033")

_MODEL_MAP = {
    "sr-1032": ("single-image-super-resolution-1032", 4),
    "sr-1033": ("single-image-super-resolution-1033", 3),
}


def ensure_openvino_available() -> None:
    """Check that the ``openvino`` runtime is importable."""
    try:
        import openvino as ov  # noqa: F401
    except ImportError as exc:
        raise ValueError(
            "OpenVINO is not installed. Run 'pip install openvino' to enable Intel super resolution."
        ) from exc


def _model_cache_dir() -> Path:
    return Path.home() / ".cache" / "sdr2hdr" / "intel_models"


def _find_or_download_model(full_name: str) -> str:
    """Return the path to the model ``.xml`` file.

    Downloads the model from the Open Model Zoo when it is not already cached
    under ``~/.cache/sdr2hdr/intel_models``.
    """
    cache_dir = _model_cache_dir()
    for precision in ("FP16", "FP32"):
        xml = cache_dir / "intel" / full_name / precision / f"{full_name}.xml"
        if xml.exists():
            return str(xml)

    # Attempt automatic download via omz_downloader (ships with openvino-dev).
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "omz_downloader",
                "--name",
                full_name,
                "--output_dir",
                str(cache_dir),
                "--precision",
                "FP16",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            f"Model '{full_name}' not found and omz_downloader is not installed. "
            "Install it with 'pip install openvino-dev' or download the model manually to "
            f"{cache_dir}/intel/{full_name}/FP16/{full_name}.xml"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"Failed to download model '{full_name}': {exc.stderr.strip()}"
        ) from exc

    xml = cache_dir / "intel" / full_name / "FP16" / f"{full_name}.xml"
    if not xml.exists():
        raise ValueError(
            f"Model download succeeded but '{xml}' was not created. "
            "Check the Open Model Zoo output for details."
        )
    return str(xml)


class OpenVINOUpscaler:
    """Wraps an Intel OpenVINO super-resolution model for video frame upscaling.

    Uses Intel's Open Model Zoo ``single-image-super-resolution`` models which
    accept two inputs:

    1. The low-resolution BGR image.
    2. A bicubic interpolation of that image at the target resolution.

    The bicubic input is computed internally so callers only need to supply the
    original decoded frame, exactly like :class:`RtxVideoUpscaler`.
    """

    def __init__(
        self,
        output_width: int,
        output_height: int,
        model: str = "sr-1032",
        device: str = "AUTO",
    ) -> None:
        ensure_openvino_available()
        import openvino as ov

        if model not in _MODEL_MAP:
            raise ValueError(f"Unknown Intel SR model: {model}. Choose from: {', '.join(_MODEL_MAP)}")

        full_name, self._scale_factor = _MODEL_MAP[model]
        model_xml = _find_or_download_model(full_name)

        core = ov.Core()
        ov_model = core.read_model(model_xml)

        # Reshape the model to the actual video dimensions.
        input_h = output_height // self._scale_factor
        input_w = output_width // self._scale_factor
        inputs = ov_model.inputs
        new_shapes: dict[str, list[int]] = {}
        # Input 0 = low-resolution image (1, 3, H/scale, W/scale)
        # Input 1 = bicubic-upscaled  image (1, 3, H, W)
        new_shapes[inputs[0].get_any_name()] = [1, 3, input_h, input_w]
        new_shapes[inputs[1].get_any_name()] = [1, 3, output_height, output_width]
        ov_model.reshape(new_shapes)

        self._compiled = core.compile_model(ov_model, device)
        self._output_width = output_width
        self._output_height = output_height
        self._input_h = input_h
        self._input_w = input_w

    # ------------------------------------------------------------------
    # Public API (mirrors RtxVideoUpscaler)
    # ------------------------------------------------------------------

    def upscale(self, frame_bgr8: np.ndarray) -> np.ndarray:
        """Upscale a BGR uint8 frame and return a BGR uint8 numpy array."""
        import cv2

        lr_blob, bicubic_blob = self._prepare_inputs(frame_bgr8, cv2)
        output = self._infer(lr_blob, bicubic_blob)

        # NCHW BGR float [0,1] → HWC BGR uint8
        out_frame = output[0].transpose(1, 2, 0)
        return np.clip(out_frame * 255.0, 0, 255).astype(np.uint8)

    def upscale_tensor(self, frame_bgr8: np.ndarray) -> torch.Tensor:
        """Upscale and return an HWC RGB float32 ``[0, 1]`` :class:`torch.Tensor`.

        The tensor lives on the CPU.  When the downstream processor runs on an
        Intel XPU it will move the tensor there inside
        ``process_frame_tensor``.
        """
        import cv2

        lr_blob, bicubic_blob = self._prepare_inputs(frame_bgr8, cv2)
        output = self._infer(lr_blob, bicubic_blob)

        # NCHW BGR → HWC RGB float32
        out = output[0].transpose(1, 2, 0)
        out = np.ascontiguousarray(out[..., ::-1])  # BGR → RGB
        return torch.from_numpy(out).clamp(0.0, 1.0)

    def close(self) -> None:
        self._compiled = None  # type: ignore[assignment]

    def __enter__(self) -> "OpenVINOUpscaler":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        self.close()
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_inputs(self, frame_bgr8: np.ndarray, cv2: object) -> tuple[np.ndarray, np.ndarray]:
        resize = getattr(cv2, "resize")
        INTER_AREA = getattr(cv2, "INTER_AREA")
        INTER_CUBIC = getattr(cv2, "INTER_CUBIC")

        lr = resize(frame_bgr8, (self._input_w, self._input_h), interpolation=INTER_AREA)
        lr_blob: np.ndarray = lr.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0

        bicubic = resize(frame_bgr8, (self._output_width, self._output_height), interpolation=INTER_CUBIC)
        bicubic_blob: np.ndarray = bicubic.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0

        return lr_blob, bicubic_blob

    def _infer(self, lr_blob: np.ndarray, bicubic_blob: np.ndarray) -> np.ndarray:
        assert self._compiled is not None
        result = self._compiled({0: lr_blob, 1: bicubic_blob})
        return result[self._compiled.output(0)]
