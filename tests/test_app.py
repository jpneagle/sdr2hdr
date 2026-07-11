import os
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

from sdr2hdr.app import (
    CancelToken,
    ConversionCallbacks,
    ConversionRequest,
    build_request_config,
    build_output_path,
    default_encoder_for_platform,
    is_hardware_encoder_failure,
    is_videotoolbox_failure,
    resolve_model_device,
    resolve_model_backend,
    resolve_output_dimensions,
    validate_request,
    run_conversion,
)
from sdr2hdr.ai import HeuristicEnhancer
from sdr2hdr.io import ffprobe_video, has_expected_hdr_metadata


class AppTests(unittest.TestCase):
    def test_build_output_path_adds_hdr_suffix(self) -> None:
        self.assertEqual(build_output_path("/tmp/example.mp4"), str(Path("/tmp/example_hdr.mp4")))

    def test_build_output_path_converts_transport_stream_extensions_to_mp4(self) -> None:
        self.assertEqual(build_output_path("/tmp/example.m2ts"), str(Path("/tmp/example_hdr.mp4")))
        self.assertEqual(build_output_path("/tmp/example.ts"), str(Path("/tmp/example_hdr.mp4")))

    def test_detects_videotoolbox_failure_message(self) -> None:
        self.assertTrue(is_videotoolbox_failure("Error: cannot create compression session: -12908"))
        self.assertTrue(is_videotoolbox_failure("hevc_videotoolbox failed"))
        self.assertFalse(is_videotoolbox_failure("generic libx265 failure"))

    def test_detects_hardware_encoder_failure_message(self) -> None:
        self.assertTrue(is_hardware_encoder_failure("hevc_videotoolbox failed"))
        self.assertTrue(is_hardware_encoder_failure("OpenEncodeSessionEx failed: unsupported device"))
        self.assertTrue(is_hardware_encoder_failure("Cannot load nvcuda.dll"))
        self.assertTrue(is_hardware_encoder_failure("Error initializing an MFX session for QSV"))
        self.assertFalse(is_hardware_encoder_failure("generic libx265 failure"))

    def test_default_encoder_for_platform(self) -> None:
        self.assertEqual(default_encoder_for_platform("Darwin"), "hevc_videotoolbox")
        self.assertEqual(default_encoder_for_platform("Windows"), "hevc_nvenc")
        self.assertEqual(default_encoder_for_platform("Linux"), "libx265")

    def test_portrait_uses_stronger_default_ai_strength_when_model_path_is_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "in.mp4"
            model_path = Path(temp_dir) / "model.pt"
            input_path.write_bytes(b"")
            model_path.write_bytes(b"")
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(Path(temp_dir) / "out.mp4"),
                preset="portrait",
                model_path=str(model_path),
            )
            config, _, _ = build_request_config(request)
            self.assertEqual(config.ai_strength, 0.25)

    def test_hdr_style_adjusts_temporal_and_shadow_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "in.mp4"
            model_path = Path(temp_dir) / "model.pt"
            input_path.write_bytes(b"")
            model_path.write_bytes(b"")
            natural = ConversionRequest(
                input_path=str(input_path),
                output_path=str(Path(temp_dir) / "natural.mp4"),
                preset="balanced",
                hdr_style="natural",
                model_path=str(model_path),
            )
            night = ConversionRequest(
                input_path=str(input_path),
                output_path=str(Path(temp_dir) / "night.mp4"),
                preset="balanced",
                hdr_style="night",
                model_path=str(model_path),
            )
            natural_config, _, _ = build_request_config(natural)
            night_config, _, _ = build_request_config(night)
            self.assertGreater(natural_config.shadow_lift_limit, night_config.shadow_lift_limit)
            self.assertGreater(night_config.temporal_stability_strength, natural_config.temporal_stability_strength)

    def test_tone_reference_anchors_diffuse_white(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "in.mp4"
            model_path = Path(temp_dir) / "model.pt"
            input_path.write_bytes(b"")
            model_path.write_bytes(b"")
            vivid = ConversionRequest(
                input_path=str(input_path),
                output_path=str(Path(temp_dir) / "vivid.mp4"),
                preset="balanced",
                tone="vivid",
                model_path=str(model_path),
            )
            reference = ConversionRequest(
                input_path=str(input_path),
                output_path=str(Path(temp_dir) / "ref.mp4"),
                preset="balanced",
                tone="reference",
                input_eotf="bt1886",
                model_path=str(model_path),
            )
            vivid_config, _, _ = build_request_config(vivid)
            reference_config, _, _ = build_request_config(reference)
            self.assertIsNone(vivid_config.diffuse_white_nits)
            self.assertEqual(reference_config.diffuse_white_nits, 203.0)
            self.assertEqual(vivid_config.input_eotf, "srgb")
            self.assertEqual(reference_config.input_eotf, "bt1886")

    def test_resolve_output_dimensions_scales_to_even_size(self) -> None:
        info = SimpleNamespace(width=1919, height=1079)
        request = ConversionRequest(
            input_path="/tmp/in.mp4",
            output_path="/tmp/out.mp4",
            output_scale=2.0,
            model_path="model.pt",
        )
        self.assertEqual(resolve_output_dimensions(info, request), (3838, 2158))

    def test_resolve_output_dimensions_uses_explicit_target(self) -> None:
        info = SimpleNamespace(width=1920, height=1080)
        request = ConversionRequest(
            input_path="/tmp/in.mp4",
            output_path="/tmp/out.mp4",
            target_width=3840,
            target_height=2160,
            model_path="model.pt",
        )
        self.assertEqual(resolve_output_dimensions(info, request), (3840, 2160))

    def test_validate_request_rejects_unknown_tone_and_eotf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "in.mp4"
            model_path = Path(temp_dir) / "model.pt"
            input_path.write_bytes(b"")
            model_path.write_bytes(b"")
            base = dict(
                input_path=str(input_path),
                output_path=str(Path(temp_dir) / "out.mp4"),
                preset="portrait",
                model_path=str(model_path),
            )
            with self.assertRaises(ValueError):
                validate_request(ConversionRequest(tone="punchy", **base))
            with self.assertRaises(ValueError):
                validate_request(ConversionRequest(input_eotf="gamma22", **base))
            with self.assertRaises(ValueError):
                validate_request(ConversionRequest(upscale_engine="waifu2x", **base))
            with self.assertRaises(ValueError):
                validate_request(ConversionRequest(scaler="nearest", **base))
            with self.assertRaises(ValueError):
                validate_request(ConversionRequest(output_scale=0.0, **base))
            with self.assertRaises(ValueError):
                validate_request(ConversionRequest(target_width=3839, target_height=2160, **base))
            with self.assertRaises(ValueError):
                validate_request(
                    ConversionRequest(output_scale=2.0, target_width=3840, target_height=2160, **base)
                )

    def test_validate_request_rejects_unknown_rtx_video_quality(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "in.mp4"
            input_path.write_bytes(b"")
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_bytes(b"")
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(Path(temp_dir) / "out.mp4"),
                preset="portrait",
                model_path=str(model_path),
                upscale_engine="rtx-video",
                rtx_video_quality="extreme",
            )
            with self.assertRaises(ValueError):
                validate_request(request)

    def test_validate_request_checks_rtx_video_availability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "in.mp4"
            input_path.write_bytes(b"")
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_bytes(b"")
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(Path(temp_dir) / "out.mp4"),
                preset="portrait",
                model_path=str(model_path),
                upscale_engine="rtx-video",
            )
            with mock.patch("sdr2hdr.app.ensure_rtx_video_available") as ensure_mock:
                validate_request(request)
            ensure_mock.assert_called_once()

    def test_validate_request_requires_model_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "in.mp4"
            input_path.write_bytes(b"")
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(Path(temp_dir) / "out.mp4"),
                preset="portrait",
                model_path=None,
            )
            with self.assertRaises(ValueError):
                validate_request(request)

    def test_validate_request_rejects_non_pt_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "in.mp4"
            model_path = Path(temp_dir) / "model.onnx"
            input_path.write_bytes(b"")
            model_path.write_bytes(b"")
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(Path(temp_dir) / "out.mp4"),
                preset="portrait",
                backend="numpy",
                model_path=str(model_path),
            )
            with self.assertRaises(ValueError):
                validate_request(request)

    def test_validate_request_rejects_missing_model_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "in.mp4"
            input_path.write_bytes(b"")
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(Path(temp_dir) / "out.mp4"),
                preset="portrait",
                model_path=str(Path(temp_dir) / "missing.pt"),
            )
            with self.assertRaises(ValueError):
                validate_request(request)

    def test_resolve_model_device_uses_backend_resolved_device_for_auto(self) -> None:
        request = ConversionRequest(input_path="/tmp/in.mp4", output_path="/tmp/out.mp4", device="auto")
        self.assertEqual(resolve_model_device(request, "mps"), "mps")
        self.assertEqual(resolve_model_device(request, None), "cpu")

    def test_resolve_model_backend_uses_torch_device_for_auto(self) -> None:
        request = ConversionRequest(
            input_path="/tmp/in.mp4",
            output_path="/tmp/out.mp4",
            backend="auto",
            model_path="model.pt",
        )
        self.assertEqual(resolve_model_backend(request, "mps"), "mps")
        self.assertEqual(resolve_model_backend(request, None), "numpy")

    def test_run_conversion_respects_cancel_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_bytes(b"placeholder")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x90:rate=24",
                    "-t",
                    "2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(input_path),
                ],
                check=True,
            )
            token = CancelToken()
            token.cancel()
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(output_path),
                preset="poc",
                encoder="libx265",
                backend="numpy",
                model_path=str(model_path),
            )
            with mock.patch("sdr2hdr.app.TorchMapEnhancer") as enhancer_cls:
                enhancer_cls.return_value = HeuristicEnhancer()
                result = run_conversion(request, callbacks=ConversionCallbacks(), cancel_token=token)
            self.assertTrue(result.cancelled)
            self.assertEqual(result.processed_frames, 0)
            self.assertFalse(output_path.exists())

    def test_cancel_keeps_partial_output_after_some_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_bytes(b"placeholder")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x90:rate=24",
                    "-t",
                    "2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(input_path),
                ],
                check=True,
            )
            token = CancelToken()
            events: list[int] = []

            def on_progress(processed: int, total: int | None, fps: float | None) -> None:
                events.append(processed)
                if processed >= 2:
                    token.cancel()

            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(output_path),
                preset="poc",
                encoder="libx265",
                backend="numpy",
                model_path=str(model_path),
            )
            with mock.patch("sdr2hdr.app.TorchMapEnhancer") as enhancer_cls:
                enhancer_cls.return_value = HeuristicEnhancer()
                result = run_conversion(
                    request,
                    callbacks=ConversionCallbacks(on_progress=on_progress),
                    cancel_token=token,
                )
            self.assertTrue(result.cancelled)
            self.assertGreaterEqual(result.processed_frames, 2)
            self.assertTrue(output_path.exists())
            self.assertTrue(has_expected_hdr_metadata(str(output_path)))

    def test_cancel_can_drop_partial_output_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_bytes(b"placeholder")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x90:rate=24",
                    "-t",
                    "2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(input_path),
                ],
                check=True,
            )
            token = CancelToken()
            token.cancel()
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(output_path),
                preset="poc",
                encoder="libx265",
                backend="numpy",
                model_path=str(model_path),
                keep_partial_output_on_cancel=False,
            )
            with mock.patch("sdr2hdr.app.TorchMapEnhancer") as enhancer_cls:
                enhancer_cls.return_value = HeuristicEnhancer()
                result = run_conversion(request, callbacks=ConversionCallbacks(), cancel_token=token)
            self.assertTrue(result.cancelled)
            self.assertFalse(output_path.exists())

    def test_completed_output_has_expected_hdr_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_bytes(b"placeholder")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x90:rate=24",
                    "-t",
                    "1",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(input_path),
                ],
                check=True,
            )
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(output_path),
                preset="poc",
                encoder="libx265",
                backend="numpy",
                model_path=str(model_path),
                max_frames=12,
            )
            with mock.patch("sdr2hdr.app.TorchMapEnhancer") as enhancer_cls:
                enhancer_cls.return_value = HeuristicEnhancer()
                result = run_conversion(request, callbacks=ConversionCallbacks())
            self.assertFalse(result.cancelled)
            self.assertTrue(has_expected_hdr_metadata(str(output_path)))

    def test_run_conversion_with_rtx_video_upscales_via_one_resident_session(self) -> None:
        import cv2

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_bytes(b"placeholder")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x90:rate=24",
                    "-t",
                    "1",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(input_path),
                ],
                check=True,
            )

            class FakeRtxUpscaler:
                instances: list["FakeRtxUpscaler"] = []

                def __init__(self, output_width: int, output_height: int, quality: str = "high", device: int = 0) -> None:
                    self.output_width = output_width
                    self.output_height = output_height
                    self.quality = quality
                    self.calls = 0
                    FakeRtxUpscaler.instances.append(self)

                def upscale(self, frame_bgr8):
                    self.calls += 1
                    return cv2.resize(frame_bgr8, (self.output_width, self.output_height))

                def close(self) -> None:
                    pass

            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(output_path),
                preset="poc",
                encoder="libx265",
                backend="numpy",
                model_path=str(model_path),
                upscale_engine="rtx-video",
                output_scale=2.0,
                rtx_video_quality="ultra",
                max_frames=6,
            )
            with (
                mock.patch("sdr2hdr.app.TorchMapEnhancer") as enhancer_cls,
                mock.patch("sdr2hdr.app.RtxVideoUpscaler", FakeRtxUpscaler),
            ):
                enhancer_cls.return_value = HeuristicEnhancer()
                result = run_conversion(request, callbacks=ConversionCallbacks())

            self.assertFalse(result.cancelled)
            # Exactly one nvvfx session for the whole job, reused per frame -
            # this is what prevents NGX from re-bootstrapping (and spawning
            # nvngx_update.exe) on every frame or chunk.
            self.assertEqual(len(FakeRtxUpscaler.instances), 1)
            instance = FakeRtxUpscaler.instances[0]
            self.assertEqual((instance.output_width, instance.output_height), (320, 180))
            self.assertEqual(instance.quality, "ultra")
            self.assertEqual(instance.calls, result.processed_frames)
            self.assertGreater(instance.calls, 0)

            output_info = ffprobe_video(str(output_path))
            self.assertEqual((output_info.width, output_info.height), (320, 180))


if __name__ == "__main__":
    unittest.main()
