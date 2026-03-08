import os
import subprocess
import tempfile
import unittest
from pathlib import Path

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
    validate_request,
    run_conversion,
)
from sdr2hdr.io import has_expected_hdr_metadata


class AppTests(unittest.TestCase):
    def test_build_output_path_adds_hdr_suffix(self) -> None:
        self.assertEqual(build_output_path("/tmp/example.mp4"), "/tmp/example_hdr.mp4")

    def test_build_output_path_converts_transport_stream_extensions_to_mp4(self) -> None:
        self.assertEqual(build_output_path("/tmp/example.m2ts"), "/tmp/example_hdr.mp4")
        self.assertEqual(build_output_path("/tmp/example.ts"), "/tmp/example_hdr.mp4")

    def test_detects_videotoolbox_failure_message(self) -> None:
        self.assertTrue(is_videotoolbox_failure("Error: cannot create compression session: -12908"))
        self.assertTrue(is_videotoolbox_failure("hevc_videotoolbox failed"))
        self.assertFalse(is_videotoolbox_failure("generic libx265 failure"))

    def test_detects_hardware_encoder_failure_message(self) -> None:
        self.assertTrue(is_hardware_encoder_failure("hevc_videotoolbox failed"))
        self.assertTrue(is_hardware_encoder_failure("OpenEncodeSessionEx failed: unsupported device"))
        self.assertTrue(is_hardware_encoder_failure("Cannot load nvcuda.dll"))
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
            self.assertEqual(config.ai_strength, 0.35)

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

    def test_run_conversion_respects_cancel_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
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
            )
            result = run_conversion(request, callbacks=ConversionCallbacks(), cancel_token=token)
            self.assertTrue(result.cancelled)
            self.assertEqual(result.processed_frames, 0)
            self.assertFalse(output_path.exists())

    def test_cancel_keeps_partial_output_after_some_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
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
            )
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
                keep_partial_output_on_cancel=False,
            )
            result = run_conversion(request, callbacks=ConversionCallbacks(), cancel_token=token)
            self.assertTrue(result.cancelled)
            self.assertFalse(output_path.exists())

    def test_completed_output_has_expected_hdr_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
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
                max_frames=12,
            )
            result = run_conversion(request, callbacks=ConversionCallbacks())
            self.assertFalse(result.cancelled)
            self.assertTrue(has_expected_hdr_metadata(str(output_path)))


if __name__ == "__main__":
    unittest.main()
